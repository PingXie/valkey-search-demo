#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
from datetime import datetime, timedelta
import base64
import hashlib

# Valkey imports for both modes
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode

# Google Cloud Vertex AI import
import vertexai
from vertexai.language_models import TextEmbeddingModel

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Creates and stores user personas with embeddings and avatars in Valkey.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--host', type=str, default=os.getenv("VALKEY_HOST", "localhost"), help="IP address or hostname of the Valkey server or a cluster entrypoint.")
parser.add_argument('--port', type=int, default=int(os.getenv("VALKEY_PORT", 6379)), help="Port number of the Valkey server or a cluster entrypoint.")
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
MODEL_NAME = "text-embedding-004"

print("--- Valkey Persona Creator with Avatars ---")

# --- NEW: Avatar Generation Function ---
def generate_avatar_data_uri(user_id: str) -> str:
    """
    Generates a unique, deterministic 5x5 SVG identicon and returns it as a Data URI.
    """
    # Use MD5 hash to get a consistent set of bytes for a user_id
    m = hashlib.md5()
    m.update(user_id.encode('utf-8'))
    digest = m.digest()

    # Generate a unique color from the hash
    hue = int(digest[0]) * 360 // 256
    fg_color = f"hsl({hue}, 55%, 50%)"
    bg_color = "hsl(0, 0%, 94%)" # Light grey background

    # Start the SVG string
    svg = f'<svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg"><rect width="80" height="80" fill="{bg_color}" />'

    # Create a 5x5 pattern, taking advantage of symmetry
    for y in range(5):
        for x in range(3): # Only need to compute the first 3 columns
            # Use bits from the hash to decide if a square is drawn
            # The bit position is chosen to create varied patterns
            bit_index = (y * 3 + x) % (len(digest) * 8)
            byte_index = bit_index // 8
            inner_bit_index = bit_index % 8
            
            if (digest[byte_index] >> inner_bit_index) & 1:
                # Draw the square and its horizontal mirror
                svg += f'<rect x="{x*16}" y="{y*16}" width="16" height="16" fill="{fg_color}" />'
                if x < 2: # Don't mirror the center column
                    svg += f'<rect x="{(4-x)*16}" y="{y*16}" width="16" height="16" fill="{fg_color}" />'

    svg += '</svg>'

    # Encode the SVG to Base64 and create the Data URI
    b64_svg = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64_svg}"


# --- Persona Definitions ---
personas = [
    {
        "id": "user:101",
        "name": "Alex Chen",
        "gender": "Non-binary",
        "location": "Seattle, WA",
        "bio": "A 28-year-old software developer and avid PC gamer...",
        "purchase_history": [
            {"product_id": "product:2502", "timestamp": (datetime.now() - timedelta(days=15)).isoformat()},
            {"product_id": "product:4439", "timestamp": (datetime.now() - timedelta(days=40)).isoformat()},
            {"product_id": "product:9583", "timestamp": (datetime.now() - timedelta(days=90)).isoformat()},
        ],
        "interests_for_embedding": "PC gaming, high-performance graphics cards, mechanical keyboards..."
    },
    {
        "id": "user:102",
        "name": "Brenda Garcia",
        "gender": "Female",
        "location": "Denver, CO",
        "bio": "A 34-year-old landscape photographer and weekend hiker...",
        "purchase_history": [
            {"product_id": "product:9146", "timestamp": (datetime.now() - timedelta(days=22)).isoformat()},
            {"product_id": "product:301", "timestamp": (datetime.now() - timedelta(days=60)).isoformat()},
            {"product_id": "product:3611", "timestamp": (datetime.now() - timedelta(days=120)).isoformat()},
        ],
        "interests_for_embedding": "Outdoor photography, hiking, durable tech, portable power banks..."
    }
]


# --- 1. Initialize Clients ---
try:
    print(f"Initializing Vertex AI for project '{GCP_PROJECT}' in '{GCP_LOCATION}'...")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
    print(f"Vertex AI model '{MODEL_NAME}' loaded.")

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
    exit(1)


# --- 2. Process and Store Each Persona ---
print(f"\n--- Generating avatars, embeddings, and storing {len(personas)} personas ---")
for persona in personas:
    # --- Generate all assets for the persona ---
    print(f"Processing data for {persona['name']} ({persona['id']})...")
    
    # 1. Generate the Avatar Data URI
    avatar_uri = generate_avatar_data_uri(persona['id'])
    
    # 2. Generate the embedding
    embedding_text = f"User Bio: {persona['bio']} User Interests: {persona['interests_for_embedding']}"
    response = model.get_embeddings([embedding_text])
    embedding_vector = response[0].values
    
    # 3. Prepare the complete data for the Valkey Hash
    persona_data = {
        "name": persona["name"],
        "gender": persona["gender"],
        "location": persona["location"],
        "bio": persona["bio"],
        "purchase_history": json.dumps(persona["purchase_history"]),
        "embedding": np.array(embedding_vector, dtype=np.float32).tobytes(),
        "avatar": avatar_uri  # ADDED: The new avatar data URI field
    }
    
    # 4. Store the complete persona profile in a Valkey Hash
    r.hset(persona['id'], mapping=persona_data)
    print(f"✅ Successfully stored persona {persona['id']} in Valkey.")

print("\n--- Persona creation process finished successfully. ---")
