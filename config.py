import os

SECRET_CODE = "logic is working"
DATA_DIR = "data"
EMBED_MODEL = 'all-MiniLM-L6-v2'
MODEL_NAME = "llama3"

# DOCX Export Settings
EXPORTS_DIR = "exports"
INDEX_PATH  = os.path.join(DATA_DIR, "cv_vectors.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "cv_chunks.pkl")
CV_PATH = os.path.join(DATA_DIR,"Jonathan_Hutchinson_SeniorEngineeringLeader.pdf")

CHUNK_SIZE=500
NUM_CHUNKS=40
TARGET_CV_WORD_BUDGET=1000
MAX_BULLETS_PER_ROLE=4