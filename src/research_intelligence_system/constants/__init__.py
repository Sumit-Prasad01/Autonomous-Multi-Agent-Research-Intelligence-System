import os
from pathlib import Path

PDF_PATH = "data/"


HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"
# DATA_PATH = f'data/{file_name}.pdf'
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

MODEL = "llama-3.1-8b-instant"
