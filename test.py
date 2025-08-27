# File: test.py
from app.rag_pipeline import (
    load_local_docs, 
    chunk_texts, 
    get_embeddings, 
    create_vector_store,
    generate_response_from_query
)

# --- SETUP PHASE ---
# This part runs all the functions we've built so far
print("Initializing RAG pipeline...")
my_documents = load_local_docs()
my_chunks = chunk_texts(my_documents)
my_embeddings = get_embeddings(my_chunks)
vector_store = create_vector_store(my_embeddings)
print("--- Pipeline Initialized ---")

# --- QUERY PHASE ---
# Now, let's ask a question
user_query = "What is the context window of Gemini 1.5 Pro?"
system_prompt = "You are a helpful AI assistant."

print(f"\nUser Question: {user_query}")

# Call our final function to get the answer
final_answer = generate_response_from_query(user_query, vector_store, my_chunks, system_prompt)

print("\n--- Chatbot Answer ---")
print(final_answer)