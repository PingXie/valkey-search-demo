#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
import random
import numpy as np

# Valkey imports for both modes
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode

# Google Cloud Vertex AI import
import vertexai
from vertexai.language_models import TextEmbeddingModel

# --- Argument Parsing ---
# Combines arguments for both Valkey and GCP
parser = argparse.ArgumentParser(
    description="Load product data, generate embeddings with Vertex AI, and store in a Valkey server.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--host', type=str, default=os.getenv("VALKEY_HOST", "localhost"), help="IP address or hostname of the Valkey server or a cluster entrypoint.")
parser.add_argument('--port', type=int, default=int(os.getenv("VALKEY_PORT", 6379)), help="Port number of the Valkey server or a cluster entrypoint.")
# ADDED: New flag to control cluster mode
parser.add_argument('--cluster', action='store_true', help="Enable cluster mode for connecting to a Valkey Cluster.")
parser.add_argument('--project', type=str, default=os.getenv("GCP_PROJECT"), help="Your Google Cloud Project ID.")
parser.add_argument('--location', type=str, default="us-central1", help="The GCP region for your Vertex AI job.")
args = parser.parse_args()

# --- Configuration ---
VALKEY_HOST = args.host
VALKEY_PORT = args.port
IS_CLUSTER = args.cluster
GCP_PROJECT = args.project
GCP_LOCATION = args.location
DATA_DIR = "data"
BATCH_SIZE = 100  # Keep batch size small to stay under API token limits
MODEL_NAME = "text-embedding-004"
REGIONS = ["NA", "EU", "ASIA", "LATAM"]
STOP_WORDS = set(["a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "but", "by", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "s", "t", "that", "the", "this", "to", "was", "what", "when", "where", "who", "will", "with", "storage", "ram", "gb", "mah", "mm", "hz", "with", "cm"])

# --- Helper Functions (no changes) ---
def generate_tags(text: str, separator: str = ',') -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\(|\)|,|\.', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    words = re.split(r'[\s-]+', text)
    unique_words = {word for word in words if word and word not in STOP_WORDS and not word.isdigit()}
    return separator.join(sorted(list(unique_words)))

def extract_brand(name: str) -> str:
    if not isinstance(name, str): return "Unknown"
    return name.split(' ')[0]

def clean_numeric(val, to_type=float):
    if not isinstance(val, str): val = str(val)
    numeric_part = re.findall(r'[\d\.]+', val.replace(',', ''))
    try:
        return to_type(numeric_part[0]) if numeric_part else 0
    except (ValueError, IndexError): return 0

# --- 1. Initialize Clients (Valkey and Vertex AI) ---
print("--- Unified Data and Embedding Loader ---")
try:
    print(f"Initializing Vertex AI for project '{GCP_PROJECT}' in '{GCP_LOCATION}'...")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    VECTOR_DIM = 768 # text-embedding-004 model has 768 dimensions
    print(f"Vertex AI model '{MODEL_NAME}' loaded. Vector dimension: {VECTOR_DIM}")

    # --- MODIFIED: Conditional client initialization ---
    if IS_CLUSTER:
        print(f"Connecting to Valkey Cluster at entrypoint {VALKEY_HOST}:{VALKEY_PORT}...")
        startup_nodes = [ClusterNode(host=VALKEY_HOST, port=VALKEY_PORT)]
        r = ValkeyCluster(startup_nodes=startup_nodes)
        mode_message = "Cluster"
    else:
        print(f"Connecting to standalone Valkey server at {VALKEY_HOST}:{VALKEY_PORT}...")
        r = valkey.Valkey(host=VALKEY_HOST, port=VALKEY_PORT)
        mode_message = "Standalone"
    
    r.ping()
    print(f"Successfully connected to Valkey ({mode_message} mode).")

except Exception as e:
    print(f"Error during initialization: {e}")
    print("Please check your GCP project, authentication, and Valkey connection details.")
    exit(1)

# --- 2. Find, Load, and Prepare Data ---
print("\n--- Finding and Preparing Data ---")
csv_path = None
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(".csv"):
            csv_path = os.path.join(root, file)
            break
    if csv_path: break
if not csv_path:
    print(f"Error: No CSV file found in the '{DATA_DIR}' directory or its subdirectories.")
    exit(1)
    
print(f"Found and loading data from: {csv_path}")
df = pd.read_csv(csv_path, index_col=0, on_bad_lines='skip')
df.dropna(subset=['name'], inplace=True)
df = df.fillna('')
print(f"Data prepared. Processing all {len(df)} records.")

# --- 3. Process Data in Batches (Generate Embeddings and Load to Valkey) ---
print("\n--- Generating Embeddings and Loading to Valkey in Batches ---")
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Batches"):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    
    texts_to_embed = []
    for index, row in batch_df.iterrows():
        text = f"Product: {row.get('name', '')}. Brand: {extract_brand(row.get('name', ''))}. Category: {row.get('main_category', '')}, {row.get('sub_category', '')}."
        texts_to_embed.append(text)
        
    response = model.get_embeddings(texts_to_embed)
    embedding_vectors = [item.values for item in response]
    
    # This pipeline logic works for both standalone and cluster clients
    pipe = r.pipeline(transaction=False)
    for (index, row), embedding_vector in zip(batch_df.iterrows(), embedding_vectors):
        product_key = f"product:{index}"
        brand = extract_brand(row['name'])
        region = random.choice(REGIONS)
        combined_text_for_tags = f"{row['name']} {brand} {row['main_category']} {row['sub_category']} {region}"
        
        product_data = {
            'id': index,
            'name': row['name'], 'brand': brand, 'main_category': row['main_category'],
            'sub_category': row['sub_category'], 'link': row['link'], 'image_url': row['image'],
            'rating': clean_numeric(row.get('ratings')),
            'review_count': clean_numeric(row.get('no_of_ratings')),
            'price': clean_numeric(row.get('discount_price')),
            'original_price': clean_numeric(row.get('actual_price')),
            'brand_tags': generate_tags(brand), 'search_tags': generate_tags(combined_text_for_tags),
            'region': region,
            'embedding': np.array(embedding_vector, dtype=np.float32).tobytes()
        }
        pipe.hset(product_key, mapping=product_data)
    
    pipe.execute()

print("\n--- Data loading and embedding generation process finished successfully. ---")

# --- 4. Final Instruction: Create the Full Index ---
print("\nIMPORTANT: Your data is now fully loaded with embeddings.")
print("To enable all search features, connect to your server with valkey-cli and run this single command:")
# NOTE: The FT.CREATE command is the same for both standalone and cluster modes.
create_command = (
    f"FT.CREATE products ON HASH PREFIX 1 \"product:\" SCHEMA "
    f"brand_tags TAG SEPARATOR \",\" "
    f"search_tags TAG SEPARATOR \",\" "
    f"price NUMERIC "
    f"rating NUMERIC "
    f"review_count NUMERIC "
    f"region TAG "
    f"embedding VECTOR HNSW 6 TYPE FLOAT32 DIM {VECTOR_DIM} DISTANCE_METRIC COSINE"
)
print("\n" + create_command + "\n")
