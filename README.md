# **Valkey-Powered Personalized Product Search Demo**

## **Introduction**

This project is a fully-functional web application that demonstrates a modern, AI-powered e-commerce experience running entirely on your local machine. It showcases how **Valkey** and its **Vector Search** module can serve as a high-performance, multi-modal backend for complex applications.

The application allows users to log in as different "personas" and receive personalized product recommendations and descriptions based on their interests. These recommendations are generated through a hybrid search that combines traditional keyword filtering with advanced vector similarity search.

**Key features demonstrated:**
* **100% Local AI Stack:** All AI features run locally using Ollama and the sentence-transformers library.
* **Hybrid Search:** Combining keyword (tag) search with vector similarity search.  
* **Personalization:** Using user profile embeddings to tailor search results.  
* **AI-Powered Content:** Leveraging Google's Gemini models to generate personalized sales pitches.  
* **High-Performance Caching:** Using Valkey to cache LLM responses, dramatically reducing latency.  
* **Real-time UI Updates:** Using Server-Sent Events (SSE) to push AI-generated content to the browser without a page refresh.

## **Prerequisites**

Before you begin, ensure you have the following installed on your system:

* [Docker](https://docs.docker.com/get-docker/)  
* [Python 3.10+](https://www.python.org/downloads/)  
* Google Cloud SDK (gcloud)

## **Setup and Running the Demo**

Follow these steps to get the application running.

### **Step 1: Start the Valkey Server**

We will use the official valkey/valkey-bundle Docker image, which comes with the Vector Search module pre-installed. This command starts a Valkey container, names it valkey-demo, and maps the default port 6379\.

\# We use \--rm to automatically remove the container when it's stopped, keeping things clean.  
```bash
docker run -d --rm --name valkey-demo -p 6379:6379 valkey/valkey-bundle
```
**Verify that the container is running:**
```bash
docker ps
```

You should see valkey-demo in the list. To test the connection, run:
```bash
docker exec valkey-demo valkey-cli PING
```

The server should reply with PONG.

### **Step 2: Set Up the Python Environment**

It is highly recommended to use a Python virtual environment to manage dependencies.

\# Create a virtual environment named 'venv'  
```bash
python3 -m venv venv
```
\# Activate the virtual environment  
```bash
source venv/bin/activate
```

\# Now, install all required Python packages from the requirements file  
```bash
pip install -r requirements.txt
```

### **Step 3: Install Ollama and Download AI Models**

This demo uses Ollama to run a small Large Language Model on your machine.

**1\. Install Ollama**

For other operating systems, please follow the [Ollama installation instructions](https://ollama.com/docs/install).
\# Install Ollama on Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**2\. Download the LLM Model**

For this demo, we use tinyllama, a 1.1B parameter model that is very fast on CPUs. Open a terminal and run:
```bash
ollama pull tinyllama
```

### **Step 4: Run the Data Loading Script**

This script generates vector embeddings locally using the `sentence-transformers` library and populates the Valkey instance with the product and user data.
The script supports connecting to both standalone and cluster Valkey servers usiddng the \--cluster flag.

\# Load product data from the included CSV and generate embeddings  
\# Add \--cluster if applicable  
```bash
# This may take a few minutes as it processes data and generates embeddings
python3 load_data.py
```
### **Step 5: Run the Web Application**

Finally, run the application.

\# Run the Flask development server  
\# Add the \-- \--cluster flag if connecting to a Valkey Cluster  
```bash
flask run --host=0.0.0.0 --port=5001
```

## **Accessing the Demo**

Your Flask server is now running on port 5001\.

#### **Option A: Direct Access (If running locally)**

If you are running everything on your local laptop, simply open your browser and go to:

* **http://localhost:5001**

#### **Option B: SSH Tunnel (Recommended for GCE VMs)**

To securely access the app running on your GCE VM from your laptop's browser, use an SSH tunnel. This forwards a port from your laptop to the VM.

1. **Open a *new* local terminal window** (keep the Flask server running in the other one).  
2. Run the following command, replacing the user and IP with your own:  
```bash
   ssh -L 8080:localhost:5001 your_user@your_vm_ip
```
3. Now, open the browser **on your laptop** and go to:  
   * **http://localhost:8080**

You will see the login page for the demo application.

## **Stopping the Demo**

1. **Stop the Flask Server:** Go to the terminal where Flask is running and press CTRL+C.  
2. **Stop the Valkey Container:**  
```bash
    docker stop valkey-demo
```
   *(Since we started it with \--rm, it will be automatically removed when stopped).*
