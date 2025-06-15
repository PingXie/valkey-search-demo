#!/usr/bin/env python3
import os
import argparse
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Creates a search and vector index on a standalone Valkey server or a Valkey Cluster.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '--host',
    type=str,
    default=os.getenv("VALKEY_HOST", "localhost"),
    help="IP address or hostname of the Valkey server or a cluster entrypoint."
)
parser.add_argument(
    '--port',
    type=int,
    default=int(os.getenv("VALKEY_PORT", 6379)),
    help="Port number of the Valkey server or a cluster entrypoint."
)
# ADDED: New flag to control cluster mode
parser.add_argument(
    '--cluster',
    action='store_true',
    help="Enable cluster mode for connecting to a Valkey Cluster."
)
args = parser.parse_args()

# --- Configuration ---
VALKEY_HOST = args.host
VALKEY_PORT = args.port
IS_CLUSTER = args.cluster
INDEX_NAME = "products"
DOC_PREFIX = f"product:"
VECTOR_DIM = 768
DISTANCE_METRIC = "COSINE"

# --- 1. Connect to Valkey (Standalone or Cluster) ---
print("--- Valkey Index Creation Utility ---")
try:
    # MODIFIED: Conditional client initialization
    if IS_CLUSTER:
        print(f"Connecting to Valkey Cluster via entrypoint {VALKEY_HOST}:{VALKEY_PORT}...")
        startup_nodes = [ClusterNode(host=VALKEY_HOST, port=VALKEY_PORT)]
        r = ValkeyCluster(startup_nodes=startup_nodes, decode_responses=True)
        mode_message = "Cluster"
    else:
        print(f"Connecting to standalone Valkey server at {VALKEY_HOST}:{VALKEY_PORT}...")
        r = valkey.Valkey(host=VALKEY_HOST, port=VALKEY_PORT, decode_responses=True)
        mode_message = "Standalone"

    r.ping()
    print(f"Successfully connected to Valkey ({mode_message} mode).")
except Exception as e:
    print(f"Error connecting to Valkey: {e}")
    exit(1)

# --- 2. Check for Existing Index and Create if Needed ---
print(f"\n--- Checking for index '{INDEX_NAME}'... ---")
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
