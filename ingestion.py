import faiss
import os
from PyPDF2 import PdfReader
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import config as cfg
import pickle
from cv_chunker import chunk_cv

INDEX_PATH  = os.path.join(cfg.DATA_DIR, "cv_vectors.index")
CHUNKS_PATH = os.path.join(cfg.DATA_DIR, "cv_chunks.pkl")
CV_PATH = os.path.join(cfg.DATA_DIR,"Jonathan_Hutchinson_SeniorEngineeringLeader.pdf")

def logDate(message = ""):
    now = datetime.now()

    # save the date time into a function variable
    if not hasattr(logDate, "prevDateTime"):
        logDate.prevDateTime = now   # initialize once

    #calculate the time difference between the current time and the previous time
    diff = now - logDate.prevDateTime
    #print(f"{now.strftime('%H:%M:%S')} [message] :time difference: {diff.total_seconds()} seconds")

    # update the previous time to the current time
    logDate.prevDateTime = now


def chunk_text(text, size=100):
    """Splits text into chunks so the LLM gets specific context."""
    get_pdf_text(CV_PATH)
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def get_pdf_text(path):
    """Extracts raw text from the PDF file."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def save_vector_db(index, chunks):
    """Saves both the FAISS index and the text chunks."""
    print ("inside of save_vector_db")
    if not os.path.exists(cfg.DATA_DIR):
        os.makedirs(cfg.DATA_DIR)

    # Save the numbers (FAISS)
    faiss.write_index(index, INDEX_PATH)
    
    # Save the words (Pickle)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Success: Vector DB and Chunks saved to {cfg.DATA_DIR}")

def load_vector_db():
    """Loads the numbers and words from your data directory."""
    print ("inside of load_vector_db")

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    
    return None, None

def create_or_load_index(encoder, chunk_size=cfg.CHUNK_SIZE):
    """Function to either generate embeddings for a list of text chunks or to read in a previously stored version"""
    logDate("before load_vector_db")
    index, chunks = load_vector_db()
    logDate("after load_vector_db")

    if index is None or chunks is None:
        print("Vector database not found. Initializing ingestion...")
        logDate("entering code to recreated vector ...")
        # 1. Read in the CV from pdf into text object
        raw_text = get_pdf_text(CV_PATH)

        #chunks = chunk_text(raw_text,chunk_size)
        chunks = chunk_cv(CV_PATH)

        # 2. Generate Embeddings (The Vector Math)
        #encoder = SentenceTransformer(MODEL_NAME)
        chunk_embeddings = encoder.encode(chunks)
    
        # 3. Build the FAISS Index (In-Memory for simplicity)
        dimension = chunk_embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension) # Use Euclidean distance
        index.add(np.array(chunk_embeddings).astype('float32'))

        save_vector_db(index, chunks)
        logDate("completed recreating vector")

    else:
        print("Vector database found so using version on disk")

    return index, chunks

def testImport():
    print  ("inside of ingestion.import")

if __name__ == "__main__":
    print("--- Starting Local RAG Test (No LangChain) ---")
    test_jd = "Looking for a Senior Manager with AWS and Python experience."
    
