import os
import json
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_pipeline import (
    load_local_docs, chunk_texts, get_embeddings, 
    create_vector_store, generate_response_from_query
)

load_dotenv()

text_chunks = None
vector_store = None

def initialize_rag():
    global text_chunks, vector_store
    if vector_store is not None:
        return

    print("🚀 Initializing RAG pipeline (cold start)...")
    documents = load_local_docs(folder_path="documents")
    text_chunks = chunk_texts(documents)
    embeddings = get_embeddings(text_chunks)
    vector_store = create_vector_store(embeddings)
    print("✅ RAG pipeline initialized successfully!")

def handler(event, context):
    initialize_rag()
    if vector_store is None:
        return {'statusCode': 500, 'body': json.dumps({"error": "RAG pipeline could not be initialized."})}

    if event['httpMethod'] != 'POST':
        return {'statusCode': 405, 'body': 'Method Not Allowed'}

    try:
        body = json.loads(event.get('body', '{}'))
        query = body.get('query')
        system_prompt = body.get('system_prompt', "You are a helpful assistant.")
        if not query:
            return {'statusCode': 400, 'body': json.dumps({"error": "Query is required"})}
        
        response_text = generate_response_from_query(query, vector_store, text_chunks, system_prompt)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"response": response_text})
        }
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({"error": str(e)})}