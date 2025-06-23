#!/usr/bin/env python3
import os
import re
import argparse
import sys
import json
import hashlib
import base64
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import random
import numpy as np

# Valkey imports for both modes
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Load product data and generate embeddings, using Vertex AI if configured, otherwise falling back to a local model.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--host', type=str, default=os.getenv("VALKEY_HOST", "localhost"), help="IP address or hostname of the Valkey server or a cluster entrypoint.")
parser.add_argument('--port', type=int, default=int(os.getenv("VALKEY_PORT", 6379)), help="Port number of the Valkey server or a cluster entrypoint.")
parser.add_argument('--cluster', action='store_true', help="Enable cluster mode for connecting to a Valkey Cluster.")
# GCP Project is now optional. If not provided, the script will use the local fallback.
parser.add_argument('--project', type=str, default=os.getenv("GCP_PROJECT"), help="[Optional] Your Google Cloud Project ID. If not set, a local model will be used.")
parser.add_argument('--location', type=str, default="us-central1", help="The GCP region for your Vertex AI job.")
parser.add_argument('--flush', action='store_true', help="Flush all data from the Valkey server before loading new data.")
args = parser.parse_args()

# --- Configuration ---
VALKEY_HOST = args.host
VALKEY_PORT = args.port
IS_CLUSTER = args.cluster
FLUSH_DATA = args.flush
DATA_DIR = "data"
BATCH_SIZE = 100
INDEX_NAME = "products"
DOC_PREFIX = f"product:"
DISTANCE_METRIC = "COSINE"
REGIONS = ["NA", "EU", "ASIA", "LATAM"]
STOP_WORDS = set(["a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "but", "by", "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "s", "t", "that", "the", "this", "to", "was", "what", "when", "where", "who", "will", "with", "storage", "ram", "gb", "mah", "mm", "hz", "with", "cm"])

# --- Dynamic AI Configuration ---
AI_MODE = None
MODEL_NAME = None
VECTOR_DIM = None
model = None # This will hold either the GCP or local model client

if args.project:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    AI_MODE = "GCP"
    GCP_PROJECT = args.project
    GCP_LOCATION = args.location
    MODEL_NAME = "text-embedding-004"
    VECTOR_DIM = 768 # text-embedding-004 model has 768 dimensions
else:
    from sentence_transformers import SentenceTransformer
    AI_MODE = "LOCAL"
    MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_DIM = 384 # all-MiniLM-L6-v2 model has 384 dimensions

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

def generate_avatar_data_uri(user_id: str) -> str:
    m = hashlib.md5()
    m.update(user_id.encode('utf-8'))
    digest = m.digest()
    hue = int(digest[0]) * 360 // 256
    fg_color = f"hsl({hue}, 55%, 50%)"
    bg_color = "hsl(0, 0%, 94%)"
    svg = f'<svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg"><rect width="80" height="80" fill="{bg_color}" />'
    for y in range(5):
        for x in range(3):
            bit_index = (y * 3 + x) % (len(digest) * 8)
            byte_index, inner_bit_index = divmod(bit_index, 8)
            if (digest[byte_index] >> inner_bit_index) & 1:
                svg += f'<rect x="{x*16}" y="{y*16}" width="16" height="16" fill="{fg_color}" />'
                if x < 2:
                    svg += f'<rect x="{(4-x)*16}" y="{y*16}" width="16" height="16" fill="{fg_color}" />'
    svg += '</svg>'
    b64_svg = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64_svg}"

# --- 1. Initialize Clients ---
try:
    print(f"\n--- AI Mode Detected: {AI_MODE} ---")
    if AI_MODE == "GCP":
        print(f"--- Initializing Vertex AI for project '{GCP_PROJECT}' in '{GCP_LOCATION}'...")
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    elif AI_MODE == "LOCAL":
        print(f"--- Initializing local embedding model '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME)
    print(f"✅ AI model '{MODEL_NAME}' loaded. Vector dimension: {VECTOR_DIM}")

    if IS_CLUSTER:
        mode_message = "Cluster"
        print(f"\n--- Connecting to Valkey Cluster at entrypoint {VALKEY_HOST}:{VALKEY_PORT}...")
        startup_nodes = [ClusterNode(host=VALKEY_HOST, port=VALKEY_PORT)]
        r = ValkeyCluster(startup_nodes=startup_nodes, decode_responses=True)
        primary_node_objects = r.get_primaries()
        primary_nodes = [{'host': node.host, 'port': node.port} for node in primary_node_objects]

        if not primary_nodes:
            print("Error: Could not find any primary (master) nodes.", file=sys.stderr)
            exit(1)

        r = ValkeyCluster(startup_nodes=startup_nodes)

    else:
        mode_message = "Standalone"
        print(f"\n--- Connecting to standalone Valkey server at {VALKEY_HOST}:{VALKEY_PORT}...")
        r = valkey.Valkey(host=VALKEY_HOST, port=VALKEY_PORT)
        primary_nodes = [{"host": VALKEY_HOST, "port": VALKEY_PORT}]

    r.ping()
    print(f"✅ Successfully connected to Valkey ({mode_message} mode).")

except Exception as e:
    print(f"Error during initialization: {e}")
    if AI_MODE == "GCP":
        print("Please check your GCP project, authentication, and Valkey connection details.")
    else:
        print("Please check your Valkey connection details and ensure AI libraries are installed.")
    exit(1)

# --- 2. Prepare Nodes for Flushing ---
if FLUSH_DATA:
    print("\n--- Flushing server(s) ...")
    print(f"⚠️  WARNING: This will delete ALL data from the Valkey server! ⚠️")
    print(f"⚠️  WARNING: This operation is irreversible! ⚠️")

    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled by user.")
        exit(0)

    success_count = 0
    error_count = 0
    for node in primary_nodes:
        node_host = node['host']
        node_port = node['port']
        try:
            node_conn = valkey.Valkey(host=node_host, port=node_port)
            node_conn.flushall()
            print(f"✅ Successfully flushed {node_host}:{node_port}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to flush {node_host}:{node_port}. Error: {e}", file=sys.stderr)
            error_count += 1
    print(f"Summary: {success_count} node(s) flushed, {error_count} failed.")


# --- 3. Find, Load, and Prepare Data ---
print("\n--- Finding and Preparing Product Data ---")
REQUIRED_PRODUCT_HEADER = [
    "Unnamed: 0",
    "name", "main_category", "sub_category", "image", "link",
    "ratings", "no_of_ratings", "discount_price", "actual_price"
]
matching_csv_paths = []
print(f"Searching for all product data files in '{DATA_DIR}' ...")

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(".csv"):
            potential_path = os.path.join(root, file)
            try:
                df_header = pd.read_csv(potential_path, nrows=0)
                if list(df_header.columns) == REQUIRED_PRODUCT_HEADER:
                    print(f"Found product data file: {potential_path}")
                    matching_csv_paths.append(potential_path)
            except Exception as e:
                print(f"❌ Could not read header of {potential_path}. Skipping. Error: {e}")

if not matching_csv_paths:
    print(f"❌ Error: No CSV files with the required header were found in '{DATA_DIR}' or its subdirectories.")
    exit(1)

print(f"Loading and combining data from {len(matching_csv_paths)} file(s)...")
try:
    list_of_dfs = [
        pd.read_csv(path, index_col=0, on_bad_lines='skip')
        for path in matching_csv_paths
    ]
    df = pd.concat(list_of_dfs, ignore_index=True)
except Exception as e:
    print(f"❌ Error loading or concatenating CSV files. Details: {e}")
    exit(1)
print(f"✅ Data prepared. Processing all {len(df)} records.")


# --- 4. Process Data in Batches (Generate Embeddings and Load to Valkey) ---
print("\n--- Generating Product Embeddings and Loading to Valkey in Batches ---")
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Batches"):
    batch_df = df.iloc[i:i+BATCH_SIZE]

    texts_to_embed = []
    for index, row in batch_df.iterrows():
        text = f"Product: {row.get('name', '')}. Brand: {extract_brand(row.get('name', ''))}. Category: {row.get('main_category', '')}, {row.get('sub_category', '')}."
        texts_to_embed.append(text)

    if AI_MODE == "GCP":
        response = model.get_embeddings(texts_to_embed)
        embedding_vectors = [item.values for item in response]
    else: # LOCAL mode
        embedding_vectors = model.encode(texts_to_embed, convert_to_numpy=True)

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

print("✅ Data loading and embedding generation process finished successfully.")

# --- 5. Final Instruction: Create the Full Index (VECTOR_DIM is dynamic) ---
print(f"\n--- Preparing index '{INDEX_NAME}'... ---")
# This version is more precise. It attempts to drop the index and will ONLY
# ignore the specific error that Valkey returns when the index doesn't exist.
# Any other error during the drop command will correctly halt the script.
try:
    print(f"Attempting to drop index '{INDEX_NAME}' to ensure a clean slate...")
    r.execute_command("FT.DROPINDEX", INDEX_NAME)
    print(f"✅ Existing index '{INDEX_NAME}' dropped successfully.")
except Exception as e:
    # This is the correct way to handle this: check if the error message
    # indicates the index was not found, which is an expected and safe condition.
    if "Index with name" in str(e):
        print(f"Index '{INDEX_NAME}' did not exist, which is fine.")
    else:
        # If it's a different error, something is wrong, so we re-raise it.
        print(f"❌ An unexpected error occurred while trying to drop the index: {e}")
        raise e

# Now, create the index. This part is guaranteed to run on a clean slate.
try:
    print(f"Creating index '{INDEX_NAME}' with vector dimension {VECTOR_DIM}...")
    command_args = [
        "FT.CREATE", INDEX_NAME,
        "ON", "HASH",
        "PREFIX", "1", DOC_PREFIX,
        "SCHEMA",
        "brand_tags", "TAG", "SEPARATOR", ",",
        "search_tags", "TAG", "SEPARATOR", ",",
        "region", "TAG",
        "price", "NUMERIC",
        "rating", "NUMERIC",
        "review_count", "NUMERIC",
        "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(VECTOR_DIM),
            "DISTANCE_METRIC", DISTANCE_METRIC
    ]
    r.execute_command(*command_args)
    print(f"✅ Index '{INDEX_NAME}' created successfully.")

except Exception as e:
    print(f"❌ Failed to create index '{INDEX_NAME}'.")
    print(f"Details: {e}")
    exit(1)
 
 # --- 6. Create Users ---
print("\n--- Loading Persona Dataset ---")
PERSONAS_CSV_PATH = "data/personas.csv"
try:
    df = pd.read_csv(PERSONAS_CSV_PATH)
    print(f"✅ Successfully loaded {len(df)} personas from CSV.")
except FileNotFoundError:
    print(f"❌ FATAL: The persona database file '{PERSONAS_CSV_PATH}' was not found.")
    exit(1)

print(f"\n--- Generating Persona Embeddings and Storing {len(df)} in Valkey  ---")
pipe = r.pipeline(transaction=False)
for index, persona in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Personas"):
    user_id = persona['id']
    texts_to_embed = []
    texts_to_embed.append( f"User Persona: {persona['bio']} User Interests: {persona['interests_for_embedding']}")

    try:
        if AI_MODE == "GCP":
            response = model.get_embeddings(texts_to_embed)
            embedding_vector = [item.values for item in response][0]
        else: # LOCAL mode
            embedding_vector = model.encode(texts_to_embed, convert_to_numpy=True)
    except Exception as e:
        print(f"WARNING: Could not generate embedding for {user_id}. Using random vector. Details: {e}")
        embedding_vector = np.random.rand(768).astype(np.float32)

    # Prepare data for Valkey Hash. The purchase_history is already a JSON string from the CSV.
    persona_data = {
        "id": user_id,
        "name": persona.get("name", f"User {user_id}"),
        "bio": persona.get("bio", ""),
        "purchase_history": persona.get("purchase_history", "[]"),
        "embedding": np.array(embedding_vector, dtype=np.float32).tobytes(),
        "avatar": generate_avatar_data_uri(user_id)
    }
    pipe.hset(user_id, mapping=persona_data)

try:
    print("Saving personas to Valkey ...")
    pipe.execute()
    print("✅ Successfully stored personas in Valkey.")
except Exception as e:
    print(f"❌ Failed to save personas to Valkey. Error: {e}")
