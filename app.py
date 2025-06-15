#!/usr/bin/env python3
import os
import json
import base64
import threading
import numpy as np
import valkey
from valkey.cluster import ValkeyCluster, ClusterNode
from flask import Flask, render_template, request, redirect, url_for, session

# --- AI and Helper Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- App Initialization ---
app = Flask(__name__)

# --- CENTRALIZED FLASK CONFIGURATION ---
# Load configuration from environment variables into Flask's app.config
# This is the "Flask way" to handle configuration.
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-very-secret-key-for-demo-purposes")
app.config['VALKEY_HOST'] = os.getenv("VALKEY_HOST", "localhost")
app.config['VALKEY_PORT'] = int(os.getenv("VALKEY_PORT", 6379))
app.config['VALKEY_IS_CLUSTER'] = os.getenv("VALKEY_IS_CLUSTER", "false").lower() == "true"
app.config['GCP_PROJECT'] = os.getenv("GCP_PROJECT")
app.config['GCP_LOCATION'] = "us-central1"
app.config['GEMINI_MODEL_NAME'] = "gemini-1.5-flash-001"
app.config['VECTOR_DIM'] = 768


# --- Reusable Clients ---
def get_valkey_connection():
    """Gets a Valkey connection, supporting both standalone and cluster modes."""
    # CHANGED: Reads from app.config instead of global variables
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
    """Initializes and returns the Gemini model client."""
    # CHANGED: Reads from app.config instead of global variables
    vertexai.init(project=app.config['GCP_PROJECT'], location=app.config['GCP_LOCATION'])
    return GenerativeModel(app.config['GEMINI_MODEL_NAME'])

valkey_client = get_valkey_connection()
gemini_model = get_gemini_model()

# --- The rest of the script (all routes and helpers) remains exactly the same. ---
# --- They will now implicitly use the clients that were configured above. ---

def get_user_profile(user_id):
    """Retrieves a user profile hash from Valkey."""
    if not user_id:
        return None
    user_key = f"user:{user_id}"
    user_data = valkey_client.hgetall(user_key)
    # Manually decode text fields, leave embedding and avatar as is
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
    """Retrieves multiple product hashes from Valkey."""
    pipe = valkey_client.pipeline(transaction=False)
    for pid in product_ids:
        pipe.hgetall(pid)
    results = pipe.execute()
    
    products = []
    for data in results:
        if data:
            # Convert binary image blob to a Data URI for direct HTML rendering
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
    """
    A function designed to run in a background thread to call Gemini
    and cache the results without blocking the main web request.
    """
    def generate_and_cache():
        for product in products:
            product_id_num = product['id']
            cache_key = f"llm_cache:user:{user_profile['id']}:product:{product_id_num}"
            
            # 1. Check cache first
            if valkey_client.exists(cache_key):
                print(f"CACHE HIT for {cache_key}")
                continue
            
            print(f"CACHE MISS for {cache_key}. Calling Gemini...")
            # 2. If not in cache, call Gemini
            try:
                prompt = (
                    f"You are a helpful and persuasive sales assistant. "
                    f"A user named {user_profile['name']} is looking at the following product: '{product['name']}'. "
                    f"The user's bio is: '{user_profile['bio']}'. "
                    f"Based on the user's bio, write a short, exciting, and personalized one-paragraph sales pitch for this specific product that directly addresses their interests. "
                    f"Do not use markdown."
                )
                response = gemini_model.generate_content(prompt)
                
                # 3. Cache the result with a 2-hour TTL
                if response and response.text:
                    valkey_client.set(cache_key, response.text, ex=7200) # ex=7200 seconds -> 2 hours
                    print(f"SUCCESS: Cached Gemini response for {cache_key}")
            except Exception as e:
                print(f"ERROR: Failed to call Gemini or cache result for {cache_key}. Details: {e}")

    # Start the background thread
    thread = threading.Thread(target=generate_and_cache)
    thread.start()

# --- Routes ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")
        
        # In this demo, we accept any password but check if the user exists
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
    if not user_id:
        return redirect(url_for("login"))
    
    user_profile = get_user_profile(user_id)
    
    # Get 5 random product keys from the database
    # To do this robustly in a cluster, we get all keys and sample them.
    # Note: In a massive database, a different approach like a dedicated set would be better.
    all_product_keys = [key for key in valkey_client.scan_iter("product:*")]
    sample_size = min(5, len(all_product_keys))
    random_product_keys = random.sample(all_product_keys, sample_size)
    
    products = get_products_by_ids(random_product_keys)
    
    # In the background, start generating personalized descriptions for these products
    if user_profile and products:
        get_personalized_descriptions_async(user_profile, products)
    
    return render_template("home.html", user=user_profile, products=products)

@app.route("/search", methods=["POST"])
def search():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    
    user_profile = get_user_profile(user_id)
    query_text = request.form.get("query", "").strip()
    
    if not query_text:
        return redirect(url_for("home"))

    # Build the hybrid query
    # 1. Prepare tag filters from keywords
    keywords = query_text.lower().split()
    tag_filter = " ".join([f"@search_tags:{{{keyword}}}" for keyword in keywords])
    
    # 2. Prepare the full hybrid query string
    # We are finding the 5 products that best match the tags AND are closest to the user's interest vector
    query_string = f"({tag_filter})=>[KNN 5 @embedding $user_vec]"
    
    # 3. Define the query object
    query = (
        valkey.search.Query(query_string)
        .return_fields("id", "name", "brand", "price", "rating", "link")
        .dialect(2)
        .sort_by("__embedding_score") # Sort by vector similarity score
    )
    
    # 4. Execute the query with the user's embedding as a parameter
    query_params = {"user_vec": user_profile["embedding"]}
    # Use the ft() method on the client to specify the index name
    results = valkey_client.ft("products").search(query, query_params)
    
    # 5. Process results
    product_ids = [f"product:{doc.id}" for doc in results.docs]
    products = get_products_by_ids(product_ids)
    
    # 6. Start the async LLM task for the search results
    if user_profile and products:
        get_personalized_descriptions_async(user_profile, products)
        
    return render_template("home.html", user=user_profile, products=products, search_query=query_text)

@app.route("/product/<product_id>")
def product_detail(product_id):
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    
    user_profile = get_user_profile(user_id)
    product_key = f"product:{product_id}"

    # Get the main product's details
    product_list = get_products_by_ids([product_key])
    if not product_list:
        return "Product not found", 404
    product = product_list[0]
    
    # Get the cached personalized description
    cache_key = f"llm_cache:user:{user_id}:product:{product_id}"
    personalized_description_bytes = valkey_client.get(cache_key)
    if personalized_description_bytes:
        product['personalized_description'] = personalized_description_bytes.decode('utf-8')
    else:
        # You could optionally trigger a blocking call here if you want to guarantee a description
        product['personalized_description'] = "A great choice for anyone looking for quality and value."

    # --- Find Similar Products (Two ways) ---
    
    # 1. Similarity based on the current product's vector
    product_embedding = valkey_client.hget(product_key, "embedding")
    # Use a generic ft() object from the main client
    ft = valkey_client.ft("products")
    
    q_product = (
        valkey.search.Query("*=>[KNN 6 @embedding $product_vec]")
        .return_field("id")
        .dialect(2)
    )
    results_prod_similar = ft.search(q_product, {"product_vec": product_embedding})
    # Exclude the product itself from its own similarity list
    similar_prod_ids = [f"product:{doc.id}" for doc in results_prod_similar.docs if doc.id != product_id][:5]
    similar_products = get_products_by_ids(similar_prod_ids)

    # 2. Similarity based on the user's vector (Personalized Recommendations)
    q_user = (
        valkey.search.Query("*=>[KNN 6 @embedding $user_vec]")
        .return_field("id")
        .dialect(2)
    )
    results_user_similar = ft.search(q_user, {"user_vec": user_profile["embedding"]})
    # Exclude the current product if it appears in the list
    recommended_prod_ids = [f"product:{doc.id}" for doc in results_user_similar.docs if doc.id != product_id][:5]
    recommended_products = get_products_by_ids(recommended_prod_ids)
    
    return render_template(
        "product_detail.html",
        user=user_profile,
        product=product,
        similar_products=similar_products,
        recommended_products=recommended_products
    )

if __name__ == "__main__":
    # Note: For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(debug=True, host='0.0.0.0', port=5001)
