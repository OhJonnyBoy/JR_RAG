import os
import sys
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion import *
from config import *

# --- CONFIGURATION ---


def _cosine_similarity_scores(query_embedding, candidate_embeddings):
    """Return cosine similarity scores between one query and many candidates."""
    query = np.array(query_embedding, dtype="float32")
    candidates = np.array(candidate_embeddings, dtype="float32")

    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    # Avoid divide-by-zero for empty/problematic embeddings.
    denom = np.maximum(query_norm * candidate_norms, 1e-12)
    return np.dot(candidates, query) / denom


def _build_tailored_cv_prompt(job_description, selected_chunks, max_bullets_per_role):
    context = "\n---\n".join(selected_chunks)
    return f"""
    You are an expert executive resume writer.
    Using ONLY the CV CONTEXT provided, create a tailored two-page CV draft for this JOB DESCRIPTION.

    Rules:
    - Do not invent skills, titles, dates, or achievements.
    - Keep content concise and ATS-friendly.
    - Prioritize measurable impact and leadership outcomes.
    - Use a maximum of {max_bullets_per_role} bullets per role.
    - Preserve concrete technologies and business results.

    CV CONTEXT:
    {context}

    JOB DESCRIPTION:
    {job_description}

    Output this exact structure:
    1) PROFESSIONAL SUMMARY
    2) CORE SKILLS
    3) PROFESSIONAL EXPERIENCE
    4) EDUCATION & CERTIFICATIONS
    """

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

    # 4. Search: Find the top chunks most similar to the Job Description
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
    
    Provide a concise analysis of matches and missing gaps.  Give a rating of bad, fair, good or excellent as a match for job
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


def run_tailored_cv_generation(
    job_description,
    target_word_budget=TARGET_CV_WORD_BUDGET,
    max_bullets_per_role=MAX_BULLETS_PER_ROLE,
    return_chunks=False,
    status_callback=None,
):
    """
    Build a JD-tailored CV draft from the most relevant CV chunks.
    """
    if status_callback:
        status_callback("Initializing embedding model...")
    encoder = SentenceTransformer(EMBED_MODEL)

    if status_callback:
        status_callback("Loading vector index...")
    _, chunks = create_or_load_index(encoder, CHUNK_SIZE)

    if status_callback:
        status_callback("Scoring CV chunks against job description...")
    query_embedding = encoder.encode(job_description)
    chunk_embeddings = encoder.encode(chunks)
    scores = _cosine_similarity_scores(query_embedding, chunk_embeddings)

    ranked_indices = np.argsort(-scores)
    selected_chunks = []
    selected_word_count = 0

    for idx in ranked_indices:
        chunk = chunks[int(idx)]
        chunk_words = len(chunk.split())
        if selected_word_count + chunk_words > target_word_budget:
            continue
        selected_chunks.append(chunk)
        selected_word_count += chunk_words

    # Ensure at least one chunk is selected for short/strict budgets.
    if not selected_chunks and len(chunks) > 0:
        selected_chunks.append(chunks[int(ranked_indices[0])])
        selected_word_count = len(selected_chunks[0].split())

    prompt = _build_tailored_cv_prompt(
        job_description=job_description,
        selected_chunks=selected_chunks,
        max_bullets_per_role=max_bullets_per_role,
    )

    if status_callback:
        status_callback("Generating tailored CV draft with LLM...")
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    tailored_cv = response["response"]

    if status_callback:
        status_callback("Completed")

    if return_chunks:
        return tailored_cv, selected_chunks, selected_word_count
    return tailored_cv

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