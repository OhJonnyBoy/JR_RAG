import os
import sys
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion import *
from config import *

# --- CONFIGURATION ---

def run_rag_analysis(job_description, return_chunks=False, status_callback=None):
    """
    This is the core engine. Your Streamlit UI will call this function.
    """
    if status_callback:
        status_callback("Initializing embedding model...")
    encoder = SentenceTransformer(EMBED_MODEL)
    if status_callback:
        status_callback("Loading vector index...")
    index, chunks = create_or_load_index(encoder,CHUNK_SIZE)

 
    print(f"The number of chunks created is : {len(chunks)}")

    # 4. Search: Find the top 3 chunks most similar to the Job Description
    if status_callback:
        status_callback("Searching matching chunks...")
    query_embedding = encoder.encode([job_description])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k=NUM_CHUNKS)
    
    # Retrieve the text of the matching chunks
    selected_chunks = [chunks[i] for i in indices[0]]
    context = "\n---\n".join(selected_chunks)

    
    # 5. The "Brain": Send to Ollama
    prompt = f"""
    You are a Senior Engineering Manager. Use the CV CONTEXT below to analyze 
    if the candidate is a good fit for the JOB DESCRIPTION.
    
    CV CONTEXT:
    {context}
    
    JOB DESCRIPTION:
    {job_description}
    
    Provide a concise analysis of matches and missing gaps.
    """
    
    if status_callback:
        status_callback("Generating analysis with LLM...")
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    result = response["response"]
    if status_callback:
        status_callback("Completed")
    if return_chunks:
        return result, selected_chunks
    return result

# --- THE "MAIN" CHECK ---
# This block allows you to test the file directly: 'python rag_cv.py'
if __name__ == "__main__":
    print("--- Starting Local RAG Test (No LangChain) ---")
    test_jd = "Looking for a Senior Manager with AWS and Python experience."
    
    print("Enter text, then EOF (Ctrl+Z then Enter on Windows):")
    test_jd = sys.stdin.read()
    print ("==========================")
    print (test_jd)    
    print ("==========================")
    
    try:
        report = run_rag_analysis(test_jd)
        print("\n--- LLM ANALYSIS ---\n")
        print(report)
    except Exception as e:
        print(f"Error during test: {e}")