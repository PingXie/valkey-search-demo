#!/usr/bin/env python3
import os
import json
import base64
import threading
import random
import numpy as np

# Valkey imports
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode
from valkey.commands.search.query import Query

# Flask imports
from flask import Flask, render_template, request, redirect, url_for, session

# Google Cloud Vertex AI import
import vertexai
from vertexai.generative_models import GenerativeModel

# --- App Initialization ---
app = Flask(__name__)

# --- CENTRALIZED FLASK CONFIGURATION ---
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-very-secret-key-for-demo-purposes")
app.config['VALKEY_HOST'] = os.getenv("VALKEY_HOST", "localhost")
app.config['VALKEY_PORT'] = int(os.getenv("VALKEY_PORT", 6379))
app.config['VALKEY_IS_CLUSTER'] = os.getenv("VALKEY_IS_CLUSTER", "false").lower() == "true"
app.config['GCP_PROJECT'] = os.getenv("GCP_PROJECT")
app.config['GCP_LOCATION'] = "us-central1"
# We keep a valid model name here, even if it fails, for future use.
app.config['GEMINI_MODEL_NAME'] = "gemini-1.5-flash-001"
app.config['VECTOR_DIM'] = 768


# --- Reusable Clients ---
def get_valkey_connection():
    is_cluster = app.config['VALKEY_IS_CLUSTER']
    host = app.config['VALKEY_HOST']
    port = app.config['VALKEY_PORT']
    print(f"Connecting to Valkey (Cluster mode: {is_cluster})...")
    if is_cluster:
        startup_nodes = [ClusterNode(host=host, port=port)]
        return ValkeyCluster(startup_nodes=startup_nodes)
    else:
        return valkey.Valkey(host=host, port=port)

def get_gemini_model():
    try:
        if not app.config['GCP_PROJECT']:
            print("WARNING: GCP_PROJECT environment variable not set. Gemini client will not be initialized.")
            return None
        vertexai.init(project=app.config['GCP_PROJECT'], location=app.config['GCP_LOCATION'])
        return GenerativeModel(app.config['GEMINI_MODEL_NAME'])
    except Exception as e:
        print(f"WARNING: Could not initialize Vertex AI. AI features will be disabled. Details: {e}")
        return None

valkey_client = get_valkey_connection()
gemini_model = get_gemini_model() # Will be None if initialization fails


# --- Helper Functions ---
def get_user_profile(user_id):
    if not user_id: return None
    user_key = f"user:{user_id}"
    user_data = valkey_client.hgetall(user_key)
    if user_data:
        return {
            "id": user_id,
            "name": user_data.get(b'name', b'').decode('utf-8'),
            "bio": user_data.get(b'bio', b'').decode('utf-8'),
            "avatar": user_data.get(b'avatar', b'').decode('utf-8'),
            "embedding": user_data.get(b'embedding'),
        }
    return None

def get_products_by_ids(product_ids):
    if not product_ids: return []
    pipe = valkey_client.pipeline(transaction=False)
    for pid in product_ids:
        key = pid if isinstance(pid, bytes) else pid.encode('utf-8')
        pipe.hgetall(key)
    results = pipe.execute()
    
    products = []
    for data in results:
        if data:
            img_blob = data.get(b'image_blob')
            img_uri = ""
            if img_blob:
                b64_img = base64.b64encode(img_blob).decode('utf-8')
                img_uri = f"data:image/png;base64,{b64_img}"
            products.append({
                "id": data.get(b'id', b'').decode('utf-8'),
                "name": data.get(b'name', b'').decode('utf-8'),
                "brand": data.get(b'brand', b'').decode('utf-8'),
                "price": data.get(b'price', b'').decode('utf-8'),
                "rating": data.get(b'rating', b'').decode('utf-8'),
                "link": data.get(b'link', b'').decode('utf-8'),
                "thumbnail": img_uri
            })
    return products

def get_personalized_descriptions_async(user_profile, products):
    def generate_and_cache():
        # If the gemini model failed to initialize, don't even try to run.
        if not gemini_model:
            print("INFO: Gemini client not available. Skipping personalized description generation.")
            return

        for product in products:
            product_id_num = product['id']
            cache_key = f"llm_cache:user:{user_profile['id']}:product:{product_id_num}"
            if valkey_client.exists(cache_key):
                print(f"INFO: CACHE HIT for {cache_key}")
                continue
            
            print(f"INFO: CACHE MISS for {cache_key}. Attempting to call Gemini...")
            try:
                prompt = (
                    f"You are a helpful and persuasive sales assistant. A user named {user_profile['name']} "
                    f"is looking at the product: '{product['name']}'. The user's bio is: '{user_profile['bio']}'. "
                    f"Based on their bio, write a short, exciting, personalized one-paragraph sales pitch for this product that directly addresses their interests. Do not use markdown."
                )
                response = gemini_model.generate_content(prompt)
                
                if response and response.text:
                    description = response.text
                    print(f"INFO: Successfully received response from Gemini for {cache_key}")
                else:
                    raise ValueError("Received an empty response from Gemini.")

            except Exception as e:
                # --- MODIFIED BEHAVIOR ---
                # If the API call fails, log the error and create a mock response.
                print(f"WARNING: Gemini API call failed. Details: {e}")
                print(f"INFO: Generating a mock personalized description for demo purposes.")
                description = (
                    f"For a savvy individual like {user_profile['name']}, the {product['name']} stands out as a premier choice. "
                    f"Its robust feature set aligns perfectly with your interests, ensuring it will "
                    f"not only meet but exceed your expectations for quality and performance."
                )
            
            # Cache the result (either real or mock) with a 2-hour TTL
            valkey_client.set(cache_key, description, ex=7200)
            print(f"INFO: Cached description for {cache_key}")

    thread = threading.Thread(target=generate_and_cache)
    thread.start()

# --- All Routes remain the same as the last correct version ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")
        if user_id and password and get_user_profile(user_id):
            session["user_id"] = user_id
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid user ID. Try 101 or 102.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@app.route("/home")
def home():
    user_id = session.get("user_id")
    if not user_id: return redirect(url_for("login"))
    user_profile = get_user_profile(user_id)
    all_product_keys = [key for key in valkey_client.scan_iter("product:*")]
    sample_size = min(5, len(all_product_keys))
    random_product_keys = random.sample(all_product_keys, sample_size)
    products = get_products_by_ids(random_product_keys)
    if user_profile and products:
        get_personalized_descriptions_async(user_profile, products)
    return render_template("home.html", user=user_profile, products=products)

@app.route("/search", methods=["POST"])
def search():
    user_id = session.get("user_id")
    if not user_id: return redirect(url_for("login"))
    user_profile = get_user_profile(user_id)
    query_text = request.form.get("query", "").strip()
    if not query_text: return redirect(url_for("home"))

    keywords = query_text.lower().split()
    tag_filter = " ".join([f"@search_tags:{{{keyword}}}" for keyword in keywords])
    query_string = f"({tag_filter})=>[KNN 5 @embedding $user_vec]"
    
    query = (
        Query(query_string)
        .return_field("id")
        .dialect(2)
        .sort_by("__embedding_score")
    )
    
    query_params = {"user_vec": user_profile["embedding"]}
    try:
        results = valkey_client.ft("products").search(query, query_params)
        product_ids = [doc.id for doc in results.docs]
        products = get_products_by_ids(product_ids)
    except Exception as e:
        print(f"Search query failed: {e}")
        products = []

    if user_profile and products:
        get_personalized_descriptions_async(user_profile, products)
    return render_template("home.html", user=user_profile, products=products, search_query=query_text)

@app.route("/product/<product_id>")
def product_detail(product_id):
    user_id = session.get("user_id")
    if not user_id: return redirect(url_for("login"))
    user_profile = get_user_profile(user_id)
    product_key = f"product:{product_id}"

    product_list = get_products_by_ids([product_key])
    if not product_list: return "Product not found", 404
    product = product_list[0]
    
    cache_key = f"llm_cache:user:{user_id}:product:{product_id}"
    personalized_description_bytes = valkey_client.get(cache_key)
    if personalized_description_bytes:
        product['personalized_description'] = personalized_description_bytes.decode('utf-8')
    else:
        print(f"INFO: No cached description found for {cache_key}. Using default text.")
        product['personalized_description'] = "A great choice for anyone looking for quality and value."

    ft = valkey_client.ft("products")
    product_embedding_bytes = valkey_client.hget(product_key.encode('utf-8'), "embedding")
    
    # Check if embedding exists before trying to query with it
    similar_products = []
    if product_embedding_bytes:
        q_product = (Query("*=>[KNN 6 @embedding $product_vec]").return_field("id").dialect(2))
        results_prod_similar = ft.search(q_product, {"product_vec": product_embedding_bytes})
        similar_prod_ids = [doc.id for doc in results_prod_similar.docs if doc.id != product_id][:5]
        similar_products = get_products_by_ids(similar_prod_ids)

    q_user = (Query("*=>[KNN 6 @embedding $user_vec]").return_field("id").dialect(2))
    results_user_similar = ft.search(q_user, {"user_vec": user_profile["embedding"]})
    recommended_prod_ids = [doc.id for doc in results_user_similar.docs if doc.id != product_id][:5]
    recommended_products = get_products_by_ids(recommended_prod_ids)
    
    return render_template(
        "product_detail.html",
        user=user_profile,
        product=product,
        similar_products=similar_products,
        recommended_products=recommended_products
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
