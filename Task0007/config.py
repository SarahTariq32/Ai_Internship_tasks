import os
from dotenv import load_dotenv

load_dotenv()

# API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

# Chunking
PARENT_CHUNK_SIZE = 1000
CHILD_CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 3
NUM_MULTI_QUERIES = 2

# Conversation
N_MESSAGES = 5
HISTORY_STRATEGY = 2

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_CONTEXT_PARENTS = 10

# Fixed chunking

FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 100

PIPELINE = "multi"

# fixed
# parent
# multi