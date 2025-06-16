# **Valkey-Powered Personalized Product Search Demo**

## **Introduction**

This project is a fully-functional web application that demonstrates a modern, AI-powered e-commerce experience. It showcases how **Valkey** and its **Vector Search** module can serve as a high-performance, multi-modal backend for complex applications.

The application allows users to log in as different "personas" and receive personalized product recommendations and descriptions based on their interests. These recommendations are generated through a hybrid search that combines traditional keyword filtering with advanced vector similarity search.

**Key features demonstrated:**

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

We will use the official valkey/valkey-extensions Docker image, which comes with the Vector Search module pre-installed. This command starts a Valkey container, names it valkey-demo, and maps the default port 6379\.

\# We use \--rm to automatically remove the container when it's stopped, keeping things clean.  
```
docker run -d --rm --name valkey-demo -p 6379:6379 valkey/valkey-extensions
```
**Verify that the container is running:**
```
docker ps
```

You` should see valkey-demo in the list. To test the connection, run:
```
docker exec valkey-demo valkey-cli PING
```

The server should reply with PONG.

### **Step 2: Set Up the Python Environment**

It is highly recommended to use a Python virtual environment to manage dependencies.

\# Create a virtual environment named 'venv'  
```
python3 -m venv venv
```
\# Activate the virtual environment  
\# On macOS/Linux/Fish Shell:  
```
source venv/bin/activate
```

\# Now, install all required Python packages from the requirements file  
```
pip install -r requirements.txt
```

### **Step 3: Configure Google Cloud and Load Data**

The application requires access to Google Cloud for its AI features.

**1\. Authenticate for Application Services**

This command will open a browser window for you to log in. It configures "Application Default Credentials," which our Python libraries use to authenticate automatically.

```
gcloud auth login
gcloud auth application-default login
```

**2\. Set Your Project ID**

Set your GCP Project ID as an environment variable. The scripts will use this to connect to the correct project.

\# In Bash/Zsh  
```
export GCP_PROJECT="your-gcp-project-id"
```

\# In Fish Shell  
\# set -x GCP_PROJECT "your-gcp-project-id"

*(Replace your-gcp-project-id with your actual project ID)*

**3\. Run the Data Loading Scripts**

You must run the script below  in order to populate the Valkey database. The scripts support connecting to both standalone and cluster Valkey servers using the \--cluster flag.

\# Load product data from the included CSV and generate embeddings  
\# Add \--cluster if applicable  
```
python3 load_data.py --project $GCP\_PROJECT
```
### **Step 4: Run the Web Application**

Finally, set the required Flask environment variables and run the application.

\# In Bash/Zsh  
```
export FLASK_APP="app.py"  
export FLASK_SECRET_KEY="a-very-strong-and-random-secret-key-12345"
```

\# In Fish Shell  
\# set \-x FLASK\_APP "app.py"  
\# set \-x FLASK\_SECRET\_KEY "a-very-strong-and-random-secret-key-12345"

\# Run the Flask development server  
\# Add the \-- \--cluster flag if connecting to a Valkey Cluster  
```
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
   ssh \-L 8080:localhost:5001 your\_user@your\_vm\_ip

3. Now, open the browser **on your laptop** and go to:  
   * **http://localhost:8080**

You will see the login page for the demo application.

## **Stopping the Demo**

1. **Stop the Flask Server:** Go to the terminal where Flask is running and press CTRL+C.  
2. **Stop the Valkey Container:**  
```
    docker stop valkey-demo
```
   *(Since we started it with \--rm, it will be automatically removed when stopped).*
