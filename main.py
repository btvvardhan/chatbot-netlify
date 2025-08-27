# File: main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from app.rag_pipeline import (
    load_local_docs,
    chunk_texts,
    get_embeddings,
    create_vector_store,
    generate_response_from_query
)

# --- 1. SET UP THE FLASK APP ---
app = Flask(__name__)
CORS(app)  # This allows web pages to call your API

# --- 2. PREPARE THE RAG PIPELINE (RUNS ONCE AT STARTUP) ---
print("🚀 Initializing RAG pipeline... This may take a moment.")
documents = load_local_docs()
text_chunks = chunk_texts(documents)
embeddings = get_embeddings(text_chunks)
vector_store = create_vector_store(embeddings)
print("✅ RAG pipeline is ready and waiting for questions!")

# --- 3. CREATE THE API ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat():
    # Get the user's question from the request
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Use the RAG pipeline to get an answer
    system_prompt = data.get('system_prompt', "You are a helpful assistant.")
    response = generate_response_from_query(query, vector_store, text_chunks, system_prompt)

    # Send the answer back
    return jsonify({"response": response})

# --- 4. START THE SERVER ---
if __name__ == '__main__':
    # This makes the server run on http://localhost:5000
    app.run(host='0.0.0.0', port=5000)