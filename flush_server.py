#!/usr/bin/env python3
import os
import sys
import argparse
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode

# -------- Configuration & Argument Parsing --------
parser = argparse.ArgumentParser(
    description="Flushes a standalone Valkey server or all primary nodes in a Valkey Cluster.",
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
    help="Enable cluster mode to discover and flush all primary nodes."
)
parser.add_argument(
    '--yes',
    action='store_true',
    help="Bypass the confirmation prompt. Use with caution."
)
args = parser.parse_args()

VALKEY_HOST = args.host
VALKEY_PORT = args.port
IS_CLUSTER = args.cluster
SKIP_CONFIRMATION = args.yes

# --- Main logic controlled by the --cluster flag ---
if IS_CLUSTER:
    print("--- Valkey Cluster Flush Utility (Cluster Mode) ---")
    
    # -------- 1. Connect to Cluster and Discover Nodes --------
    try:
        startup_nodes = [ClusterNode(host=VALKEY_HOST, port=VALKEY_PORT)]
        r_cluster = ValkeyCluster(startup_nodes=startup_nodes, decode_responses=True)
        r_cluster.ping()
        print(f"Successfully connected to cluster entrypoint at {VALKEY_HOST}:{VALKEY_PORT}")

        print("\n--- Discovering primary nodes... ---")
        primary_node_objects = r_cluster.get_primaries()
        primary_nodes = [{'host': node.host, 'port': node.port} for node in primary_node_objects]

        if not primary_nodes:
            print("Error: Could not find any primary (master) nodes.", file=sys.stderr)
            exit(1)

        print(f"Discovered {len(primary_nodes)} primary nodes to be flushed:")
        for node in primary_nodes:
            print(f"  - {node['host']}:{node['port']}")
            
    except Exception as e:
        print(f"Error: Could not connect to the Valkey cluster or discover nodes.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        exit(1)

    # -------- 2. Cluster Safety Confirmation Prompt --------
    warning_message = "This will delete ALL data from ALL keyspaces on every primary node."
    nodes_to_flush = primary_nodes

else: # Standalone Mode
    print("--- Valkey Flush Utility (Standalone Mode) ---")
    warning_message = f"This will delete ALL data from the Valkey server at {VALKEY_HOST}:{VALKEY_PORT}."
    nodes_to_flush = [{"host": VALKEY_HOST, "port": VALKEY_PORT}]

# -------- Universal Safety Prompt --------
print("\n" + "="*len(warning_message))
print(f"⚠️  WARNING: {warning_message}")
print("This operation is irreversible.")
print("="*len(warning_message) + "\n")

if not SKIP_CONFIRMATION:
    confirm = input("Are you sure you want to proceed? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Operation cancelled by user.")
        exit(0)

# -------- Universal Execution Logic --------
print("\n--- Flushing server(s)... ---")
success_count = 0
error_count = 0

for node in nodes_to_flush:
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
print("\n--- Flush process complete. ---")
print(f"Summary: {success_count} node(s) flushed, {error_count} failed.")
