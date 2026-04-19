import os

SECRET_CODE = "logic is working"
DATA_DIR = "data"
EMBED_MODEL = 'all-MiniLM-L6-v2'
MODEL_NAME = "llama3"

MODEL_NAME = "llama3"
INDEX_PATH  = os.path.join(DATA_DIR, "cv_vectors.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "cv_chunks.pkl")
CV_PATH = os.path.join(DATA_DIR,"Jonathan_Hutchinson_SeniorEngineeringLeader.pdf")

CHUNK_SIZE=500
NUM_CHUNKS=6