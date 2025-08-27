# File: app/rag_pipeline.py

import os
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai # <-- ADD THIS
import numpy as np                 # <-- AND THIS
import faiss                       # <-- AND THIS
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def load_local_docs(folder_path="documents"):
    """
    Scans a folder, reads .txt and .pdf files, and returns their content
    as a list of strings.
    """
    documents_content = []
    print(f"Reading files from: {os.path.abspath(folder_path)}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        content = ""
        try:
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            # --- THIS IS THE NEW LOGIC FOR PDFs ---
            elif filename.endswith('.pdf'):
                with open(file_path, 'rb') as f: # 'rb' is for "read binary"
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        # Extract text from each page and add it to the content
                        content += (page.extract_text() or "") + " "

            if content.strip():
                documents_content.append(content)
                print(f"  - Loaded and processed {filename}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return documents_content

def chunk_texts(texts, chunk_size=1000, chunk_overlap=100):
    """
    Takes a list of document texts and splits them into smaller,
    overlapping chunks.
    """
    all_chunks = []
    for text in texts:
        start = 0
        while start < len(text):
            end = start + chunk_size
            all_chunks.append(text[start:end])
            # The overlap helps keep context between chunks
            start += chunk_size - chunk_overlap 
    return all_chunks

def get_embeddings(texts):
    """
    Takes a list of text chunks and converts them into numerical vectors
    using the Gemini embedding model.
    """
    model = 'models/embedding-001'
    print(f"Generating embeddings for {len(texts)} chunks...")

    # Call the Gemini API to get the embeddings
    return genai.embed_content(model=model,
                             content=texts,
                             task_type="retrieval_document")['embedding']



def create_vector_store(embeddings):
    """
    Creates a FAISS vector store to index the embeddings for fast searching.
    """
    # The dimension of our vectors is the length of the first embedding
    dimension = len(embeddings[0])

    # Create a simple FAISS index. IndexFlatL2 is a basic but effective index
    # that performs an exact search.
    index = faiss.IndexFlatL2(dimension)

    # Add our embeddings to the index. They need to be in a NumPy array.
    index.add(np.array(embeddings).astype('float32'))

    print(f"Vector store created with {index.ntotal} vectors.")
    return index


def generate_response_from_query(query, vector_store, text_chunks, system_prompt):
    """
    The main RAG function. It takes a query, retrieves relevant documents,
    and generates a final answer.
    """
    # 1. Embed the user's query
    query_embedding = genai.embed_content(
        model='models/embedding-001',
        content=query,
        task_type="retrieval_query"  # Use "retrieval_query" for user questions
    )['embedding']

    # 2. Search the vector store to find the most relevant chunks
    top_k = 5  # Retrieve the top 5 most relevant chunks
    # The search returns distances and indices of the nearest vectors
    _, indices = vector_store.search(np.array([query_embedding]).astype('float32'), top_k)

    # Get the actual text chunks using the indices
    retrieved_docs = [text_chunks[i] for i in indices[0]]

    # 3. Build the prompt for the generative model
    model = genai.GenerativeModel('gemini-1.5-flash')
    context = "\n---\n".join(retrieved_docs)

    # This is the prompt engineering part
    prompt = f"""{system_prompt}

Use the following context to answer the question. If the answer is not in the context, say you don't know.

**Context:**
{context}

**Question:**
{query}
"""

    # 4. Generate the final answer
    response = model.generate_content(prompt)
    return response.text