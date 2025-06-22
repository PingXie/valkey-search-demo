#!/usr/bin/env python3
import os
import base64
import threading
import random
import numpy as np
import argparse
import json
import time

# Valkey imports
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode
from valkey.commands.search.query import Query

# Flask imports
from flask import Flask, render_template, request, redirect, url_for, session, Response, stream_with_context

# --- App Initialization & Argument Parsing ---
app = Flask(__name__)
# Use parse_known_args to ensure Flask's own arguments don't cause an error
parser = argparse.ArgumentParser(description="Valkey VSS Demo with Flask and a configurable AI backend.")
parser.add_argument('--cluster', action='store_true', help="Enable cluster mode for connecting to a Valkey Cluster.")
cli_args, _ = parser.parse_known_args()


# --- CENTRALIZED FLASK CONFIGURATION ---
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-very-secret-key-for-demo-purposes")
app.config['VALKEY_HOST'] = os.getenv("VALKEY_HOST", "localhost")
app.config['VALKEY_PORT'] = int(os.getenv("VALKEY_PORT", 6379))
app.config['VALKEY_IS_CLUSTER'] = cli_args.cluster
app.config['GCP_PROJECT'] = os.getenv("GCP_PROJECT")
app.config['GCP_LOCATION'] = "us-central1"
app.config['PLACEHOLDER_IMAGE_URL']="https://via.placeholder.com/300.png?text=No+Image"

# --- Dynamic AI Configuration ---
ai_client = None
if app.config.get('GCP_PROJECT'):
    from google import genai
    from google.genai import types
    app.config['AI_MODE'] = "GCP"
    app.config['LLM_MODEL_NAME'] = "gemini-1.5-flash-preview-0514"
    app.config['VECTOR_DIM'] = 768
    print(f"--- AI Mode Detected: {app.config['AI_MODE']} ---")
    try:
        print(f"Initializing google.genai client for project '{app.config['GCP_PROJECT']}'...")
        ai_client = genai.Client(
            project=app.config['GCP_PROJECT'],
            location=app.config['GCP_LOCATION']
        )
        print("✅ Google Genai client initialized.")
    except Exception as e:
        print(f"WARNING: Could not initialize Vertex AI client. AI features will be mocked. Details: {e}")
        ai_client = None

else:
    import ollama
    app.config['AI_MODE'] = "LOCAL"
    app.config['LLM_MODEL_NAME'] = "tinyllama"
    app.config['VECTOR_DIM'] = 384
    print(f"--- AI Mode Detected: {app.config['AI_MODE']} ---")
    try:
        print(f"Checking for local Ollama service with model '{app.config['LLM_MODEL_NAME']}'...")
        # The ollama library doesn't need a persistent client, but we can check for connectivity.
        ollama.list()
        ai_client = "Ollama Active" # Use a placeholder string to indicate Ollama is ready
        print("✅ Ollama service detected.")
    except Exception as e:
        print(f"WARNING: Could not connect to Ollama service. AI features will be mocked. Details: {e}")
        ai_client = None

# --- Valkey Client Initialization ---
def get_valkey_connection():
    is_cluster = app.config['VALKEY_IS_CLUSTER']
    host = app.config['VALKEY_HOST']
    port = app.config['VALKEY_PORT']
    print(f"Connecting to Valkey (Cluster mode: {is_cluster})...")
    try:
        if is_cluster:
            client = ValkeyCluster(startup_nodes=[ClusterNode(host=host, port=port)])
        else:
            client = valkey.Valkey(host=host, port=port)
        client.ping()
        print(f"Successfully connected to Valkey.")
        return client
    except Exception as e:
        print(f"FATAL: Could not connect to Valkey. Please check the server. Error: {e}")
        return None

valkey_client = get_valkey_connection()


# --- MMR Reranking Helper (No Changes) ---
def mmr_rerank(query_embedding, candidate_embeddings, lambda_param=0.7, top_n=5):
    """Performs Maximal Marginal Relevance reranking to diversify results."""
    if not candidate_embeddings:
        return []

    selected_indices = []
    candidate_indices = list(range(len(candidate_embeddings)))

    q_emb = query_embedding / np.linalg.norm(query_embedding)
    cand_embs = np.array(candidate_embeddings)
    cand_embs = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)

    relevance_scores = cand_embs @ q_emb

    if candidate_indices:
        best_idx_pos = np.argmax(relevance_scores)
        selected_indices.append(candidate_indices.pop(best_idx_pos))

    while len(selected_indices) < top_n and candidate_indices:
        mmr_scores = {}
        selected_embeddings = cand_embs[selected_indices]

        for i in candidate_indices:
            relevance = relevance_scores[i]
            diversity = np.max(cand_embs[i] @ selected_embeddings.T)
            mmr_scores[i] = lambda_param * relevance - (1 - lambda_param) * diversity

        if not mmr_scores: break
        best_candidate_idx = max(mmr_scores, key=mmr_scores.get)
        selected_indices.append(best_candidate_idx)
        candidate_indices.remove(best_candidate_idx)

    return selected_indices


# --- Data Helpers (No Changes) ---
def get_user_profile(user_id):
    if not user_id:
        return None
    # Assuming user IDs are prefixed for clarity, though not strictly required by HGETALL
    data = valkey_client.hgetall(f"user:{user_id}")
    if not data:
        return None
    return {
        "id":        user_id,
        "name":      data.get(b'name', b'').decode(),
        "bio":       data.get(b'bio', b'').decode(),
        "avatar":    data.get(b'avatar', b'').decode(),
        "embedding": data.get(b'embedding'),
    }

def get_products_by_ids(ids):
    if not ids:
        return []
    pipe = valkey_client.pipeline(transaction=False)
    for pid in ids:
        # Ensure the key is bytes, as hgetall expects byte strings
        key = pid if isinstance(pid, bytes) else pid.encode('utf-8')
        pipe.hgetall(key)
    results = pipe.execute()

    prods = []
    for data in results:
        if not data:
            continue
        thumbnail_url = data.get(b'image_url', b'').decode()
        if not thumbnail_url:
            thumbnail_url = app.config['PLACEHOLDER_IMAGE_URL']
        prods.append({
            "id":    data.get(b'id', b'').decode(),
            "name":  data.get(b'name', b'').decode(),
            "brand": data.get(b'brand', b'').decode(),
            "price": data.get(b'price', b'').decode(),
            "rating": data.get(b'rating', b'').decode(),
            "link":  data.get(b'link', b'').decode(),
            "thumbnail": thumbnail_url,
        })
    return prods


# --- Async LLM Descriptions ---
def get_personalized_descriptions_async(user_profile, products):
    """
    Kicks off a SINGLE background thread that loops through products sequentially
    to generate and cache their descriptions using the configured AI backend.
    """
    def task():
        ## --- DELTA START ---
        if not ai_client:
            print(f"INFO: AI client ({app.config['AI_MODE']}) not available, skipping description generation.")
            return

        for product in products:
            cache_key = f"llm_cache:user:{user_profile['id']}:product:{product['id']}"
            if valkey_client.exists(cache_key):
                print(f"INFO: [Cache Hit] for {cache_key}")
                continue

            print(f"INFO: [Cache Miss] for {cache_key}. Calling {app.config['AI_MODE']}...")
            prompt = (
                f"You are a helpful and persuasive sales assistant. A user named {user_profile['name']} "
                f"is considering the product: '{product['name']}'. Their bio is: '{user_profile['bio']}'. "
                f"Write a short, personalized paragraph for this product that addresses their interests. No markdown."
            )
            desc = None

            try:
                if app.config['AI_MODE'] == "GCP":
                    response = ai_client.generate_content(
                        model=app.config['LLM_MODEL_NAME'],
                        contents=[prompt]
                    )
                    desc = response.text if response and response.text else None
                elif app.config['AI_MODE'] == "LOCAL":
                    response = ollama.generate(
                        model=app.config['LLM_MODEL_NAME'],
                        prompt=prompt
                    )
                    desc = response['response'] if response and response['response'] else None

                if not desc: raise ValueError("Received empty response from LLM")
                print(f"INFO: [{app.config['AI_MODE']} Success] for {cache_key}")

            except Exception as e:
                print(f"WARNING: [{app.config['AI_MODE']} API Call Failed] using mock response. Details: {e}")
                desc = (
                    f"For an individual like {user_profile['name']}, the {product['name']} "
                    f"is a standout choice, aligning perfectly with your unique interests and needs."
                )
            valkey_client.set(cache_key, desc, ex=7200)
        ## --- DELTA END ---

    # Start the single background thread to run the task
    threading.Thread(target=task).start()


# --- Routes (No structural changes, but logic now depends on dynamic config) ---
@app.before_request
def check_connection():
    if not valkey_client:
        return "Error: Could not connect to Valkey. Please check the server and your connection settings.", 503

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uid = request.form.get("user_id")
        # Simple validation, in a real app this would be a proper password check
        if uid and get_user_profile(uid):
            session["user_id"] = uid
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid user ID. Try 101 or 102.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@app.route("/home")
def home():
    uid = session.get("user_id")
    if not uid:
        return redirect(url_for("login"))
    user = get_user_profile(uid)
    # Get a small random sample of products for the homepage
    keys = list(valkey_client.scan_iter("product:*"))
    picks = random.sample(keys, min(5, len(keys)))
    products = get_products_by_ids(picks)
    if user and products:
        get_personalized_descriptions_async(user, products)
    return render_template("home.html", user=user, products=products)

@app.route("/search", methods=["POST"])
def search():
    uid = session.get("user_id")
    if not uid:
        return redirect(url_for("login"))
    user = get_user_profile(uid)
    query_text = request.form.get("query", "").strip()
    if not query_text:
        return redirect(url_for("home"))

    tags = query_text.lower().split()
    tag_filter = " ".join(f"@search_tags:{{{t}}}" for t in tags)

    q_str = f"({tag_filter})=>[KNN 25 @embedding $user_vec]"
    query_obj = (
        Query(q_str)
        .return_fields("id") # Only need the ID for the first pass
        .dialect(2)
    )
    res = valkey_client.ft("products").search(query_obj, {"user_vec": user["embedding"]})

    # Fetch embeddings for the candidates to perform reranking
    candidate_ids = [f"{doc.id}" for doc in res.docs]
    pipe = valkey_client.pipeline(transaction=False)
    for pid in candidate_ids:
        pipe.hget(pid.encode('utf-8'), b"embedding")
    embedding_blobs = pipe.execute()

    # Filter out any products without embeddings and convert blobs to numpy arrays
    valid_candidate_ids = [
        pid for pid, emb in zip(candidate_ids, embedding_blobs) if emb
    ]
    candidate_embs = [
        np.frombuffer(emb, dtype=np.float32)
        for emb in embedding_blobs if emb
    ]

    if valid_candidate_ids and candidate_embs:
        selected_indices = mmr_rerank(
            np.frombuffer(user["embedding"], dtype=np.float32),
            candidate_embs,
            lambda_param=0.7,
            top_n=5
        )
        final_ids = [valid_candidate_ids[i] for i in selected_indices]
        products = get_products_by_ids(final_ids)
    else:
        products = []

    if user and products:
        get_personalized_descriptions_async(user, products)
    return render_template("home.html", user=user, products=products, search_query=query_text)


@app.route("/stream/<cache_key>")
def stream(cache_key):
    """
    This endpoint checks the Valkey cache repeatedly and streams the result
    as soon as it's available using Server-Sent Events (SSE).
    """
    def generate():
        # Try to get the cached result for up to 20 seconds
        retries = 20
        while retries > 0:
            description = valkey_client.get(cache_key)
            if description:
                yield f"data: {json.dumps({'description': description.decode('utf-8')})}\n\n"
                return

            time.sleep(1)
            retries -= 1

        # If the key never appears, send a timeout message
        yield f"data: {json.dumps({'description': 'Could not generate a personalized description at this time.'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/product/<product_id>")
def product_detail(product_id):
    uid = session.get("user_id")
    if not uid:
        return redirect(url_for("login"))
    user = get_user_profile(uid)
    key = f"product:{product_id}"
    items = get_products_by_ids([key])
    if not items:
        return "Not found", 404
    product = items[0]

    cache_key = f"llm_cache:user:{uid}:product:{product_id}"
    desc = valkey_client.get(cache_key)
    if not valkey_client.exists(cache_key):
        get_personalized_descriptions_async(user, [product])

    product['personalized_description'] = desc.decode() if desc else "A great choice that combines quality and value."

    # Find similar products based on product embedding
    ft = valkey_client.ft("products")
    product_emb_bytes = valkey_client.hget(key.encode('utf-8'), "embedding")

    similar_products = []
    if product_emb_bytes:
        q_prod = (Query("*=>[KNN 6 @embedding $product_vec]").return_field("id").dialect(2))
        res_sim = ft.search(q_prod, {"product_vec": product_emb_bytes})
        sim_ids = [f"{d.id}" for d in res_sim.docs if d.id != key][:5] # Exclude self
        similar_products = get_products_by_ids(sim_ids)
        get_personalized_descriptions_async(user, similar_products)

    # Find recommended products based on user embedding
    q_user = Query("*=>[KNN 25 @embedding $user_vec]").return_field("id").dialect(2)
    res_user = ft.search(q_user, {"user_vec": user["embedding"]})

    candidate_ids = []
    candidate_embs = []
    for d in res_user.docs:
        if d.id == key: # Exclude current product from recommendations
            continue
        pid = f"{d.id}"
        e = valkey_client.hget(pid.encode('utf-8'), "embedding")
        if e:
            candidate_ids.append(pid)
            candidate_embs.append(np.frombuffer(e, dtype=np.float32))

    if candidate_ids and candidate_embs:
        selected_indices = mmr_rerank(np.frombuffer(user["embedding"], dtype=np.float32), candidate_embs, top_n=5)
        recommended_ids = [candidate_ids[i] for i in selected_indices]
        recommended_products = get_products_by_ids(recommended_ids)
        get_personalized_descriptions_async(user, recommended_products)
    else:
        recommended_products = []

    return render_template(
        "product_detail.html",
        user=user,
        product=product,
        llm_cache_key=cache_key,
        similar_products=similar_products,
        recommended_products=recommended_products
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
