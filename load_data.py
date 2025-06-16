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
from kaggle.api.kaggle_api_extended import KaggleApi

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
parser.add_argument('--flush', action='store_true', help="Flush all data from the Valkey server before loading new data.")
args = parser.parse_args()

# --- Configuration ---
VALKEY_HOST = args.host
VALKEY_PORT = args.port
IS_CLUSTER = args.cluster
GCP_PROJECT = args.project
GCP_LOCATION = args.location
FLUSH_DATA = args.flush
NUM_PERSONAS_TO_LOAD = 10
DATA_DIR = "data"
BATCH_SIZE = 100  # Keep batch size small to stay under API token limits
MODEL_NAME = "text-embedding-004"
INDEX_NAME = "products"
DOC_PREFIX = f"product:"
VECTOR_DIM = 768 # text-embedding-004 model has 768 dimensions
DISTANCE_METRIC = "COSINE"
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

# --- 1. Initialize Clients (Valkey and Vertex AI) ---
print("========== Unified Data and Embedding Loader ==========")
try:
    print(f"\n--- Initializing Vertex AI for project '{GCP_PROJECT}' in '{GCP_LOCATION}'...")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    print(f"✅ Vertex AI model '{MODEL_NAME}' loaded. Vector dimension: {VECTOR_DIM}")

    # --- MODIFIED: Conditional client initialization ---
    if IS_CLUSTER:
        mode_message = "Cluster"
        print(f"\n--- Connecting to Valkey Cluster at entrypoint {VALKEY_HOST}:{VALKEY_PORT}...")
        startup_nodes = [ClusterNode(host=VALKEY_HOST, port=VALKEY_PORT)]
        print("Discovering primary nodes...")
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
    print("Please check your GCP project, authentication, and Valkey connection details.")
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

    # -------- Universal Execution Logic --------
    success_count = 0
    error_count = 0

    for node in primary_nodes:
        node_host = node['host']
        node_port = node['port']
        try:
            # Create a direct, standard connection to the specific node to flush it
            node_conn = valkey.Valkey(host=node_host, port=node_port)
            node_conn.flushall()
            print(f"✅ Successfully flushed {node_host}:{node_port}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to flush {node_host}:{node_port}. Error: {e}", file=sys.stderr)
            error_count += 1

    # -------- Final Report --------
    print(f"Summary: {success_count} node(s) flushed, {error_count} failed.")

# --- 3. Find, Load, and Prepare Data ---
print("\n--- Finding and Preparing Product Data ---")
# Define the exact header you require, as a list of strings
REQUIRED_PRODUCT_HEADER = [
    "name", "main_category", "sub_category", "image", "link",
    "ratings", "no_of_ratings", "discount_price", "actual_price"
]
matching_csv_paths = []
print(f"Searching for all CSV files in '{DATA_DIR}' with a matching header...")

# Recursively search for all CSV files that match the header
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(".csv"):
            potential_path = os.path.join(root, file)
            print(f"Checking file: {potential_path}")
            try:
                # Read only the header of the CSV to check columns
                df_header = pd.read_csv(potential_path, nrows=0)
                print(f"Found header: {list(df_header.columns)}")

                # Compare the file's columns with your required header
                if list(df_header.columns) == REQUIRED_PRODUCT_HEADER:
                    print(f"Header match found. Adding file to list: {potential_path}")
                    matching_csv_paths.append(potential_path)

            except Exception as e:
                print(f"Could not read header of {potential_path}. Skipping. Error: {e}")

# --- MODIFIED: Check if any matching files were found ---
if not matching_csv_paths:
    print(f"Error: No CSV files with the required header were found in '{DATA_DIR}' or its subdirectories.")
    exit(1)

print(f"\nLoading and combining data from {len(matching_csv_paths)} file(s)...")
try:
    # Create a list of DataFrames by reading each valid CSV
    list_of_dfs = [
        pd.read_csv(path, index_col=0, on_bad_lines='skip')
        for path in matching_csv_paths
    ]

    # Concatenate all DataFrames in the list into a single master DataFrame
    df = pd.concat(list_of_dfs, ignore_index=True)

except Exception as e:
    print(f"Error loading or concatenating CSV files. Details: {e}")
    exit(1)

print(f"✅ Data prepared. Processing all {len(df)} records.")

# --- 4. Process Data in Batches (Generate Embeddings and Load to Valkey) ---
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
    for (index, row), embedding_vectors in zip(batch_df.iterrows(), embedding_vectors):
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
            'embedding': np.array(embedding_vectors, dtype=np.float32).tobytes()
        }
        pipe.hset(product_key, mapping=product_data)

    pipe.execute()

print("✅ Data loading and embedding generation process finished successfully.")

# --- 5. Final Instruction: Create the Full Index ---
print(f"\n--- Preparing index '{INDEX_NAME}'... ---")
try:
    # This works on both client types
    r.ft(INDEX_NAME).info()
    print(f"Index '{INDEX_NAME}' already exists. No action taken.")
except Exception:
    # Any error (likely ResponseError for a non-existent index) means we should try to create it.
    print(f"Index '{INDEX_NAME}' does not exist. Creating it now...")
    try:
        # Define the command arguments. This is the same for both modes.
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

        # Use execute_command. The cluster client will route it to a primary node.
        # The standard client will send it to the connected server.
        r.execute_command(*command_args)

        print(f"✅ Index '{INDEX_NAME}' created successfully.")

    except Exception as e:
        print(f"❌ Failed to create index '{INDEX_NAME}'.")
        print(f"Details: {e}")
        exit(1)

# --- 6. Create Users ---
print("\n--- Loading Persona Dataset ---")
print("Downloading Persona Dataset from Kaggle ...")
try:
    api = KaggleApi()
    api.authenticate()
    # This is the "Persona-Driven Conversations Dataset"
    api.dataset_download_files('sabikasabika/new-conversations', path='./data', unzip=True)
    csv_path = './data/New-Persona-New-Conversations.csv'
    df = pd.read_csv(csv_path)
    # The persona description is in the 'persona' column
    # We drop duplicates to get unique personas
    unique_personas = df['persona'].drop_duplicates()
    
    # Select 10 random personas
    if len(unique_personas) > NUM_PERSONAS_TO_LOAD:
        sampled_personas = unique_personas.sample(n=NUM_PERSONAS_TO_LOAD, random_state=42)
    else:
        sampled_personas = unique_personas
    print(f"✅ Successfully downloaded and sampled {len(sampled_personas)} personas.")
except Exception as e:
    print(f"❌ FATAL: Could not download or process Kaggle dataset. Check API credentials. Error: {e}")
    exit(1)

print(f"Generating embeddings and storing {len(sampled_personas)} personas ...")
user_id_counter = 201
pipe = r.pipeline(transaction=False)

for persona_text in tqdm(sampled_personas, desc="Processing Personas"):
    user_id = f"user:{user_id_counter}"
    
    # Create a descriptive text for embedding
    # We use the full persona description as it's rich with interests
    text_to_embed = []
    text_to_embed.append(f"User persona: {persona_text}")

    # Generate embedding using the known-good model
    response = model.get_embeddings(text_to_embed)
    embedding_vectors = [item.values for item in response]

    # Generate an avatar
    avatar_uri = generate_avatar_data_uri(user_id)

    # Prepare data for Valkey Hash
    persona_data = {
        "id": user_id,
        "name": f"Kaggle Persona {user_id_counter}",
        "bio": persona_text,
        "avatar": avatar_uri,
        "embedding": np.array(embedding_vectors[0], dtype=np.float32).tobytes()
    }
    
    # Add the HSET command to the pipeline
    pipe.hset(user_id, mapping=persona_data)
    user_id_counter += 1

# Execute the pipeline to save all personas
try:
    print("Saving personas to Valkey ...")
    pipe.execute()
    print(f"✅ Successfully stored {user_id_counter - 201} personas in Valkey.")
except Exception as e:
    print(f"❌ Failed to save personas to Valkey. Error: {e}")
