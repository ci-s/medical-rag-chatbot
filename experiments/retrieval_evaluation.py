import sys
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.pdf_utils import read_pdf
from core.embedding import embed_chunks
from services.retrieval import FaissService
from services.splitter import TextSplitter
from core.pdf_utils import get_document
from services.chunking import chunk_document
from eval.retrieval import evaluate_source

# Setup
data_path = "../data"
file_name = "MNL_VA_Handbuch_vaskulaere_Neurologie_221230.pdf"

file_path = os.path.join(data_path, file_name)
pages = list(range(7, 109))

method_args = {
    "size": {"chunk_size": 512},
    "section": {"toc_pages": [2, 3]},
    "semantic": {},  # set NOMIC_API_KEY
}

for method, args in method_args.items():
    print(f"Method: {method}")
    chunks = chunk_document(method=method, file_path=file_path, pages=pages, **args)

    embeddings = embed_chunks([chunk.text for chunk in chunks], task_type="search_document")

    faiss_service = FaissService()
    faiss_service.create_index(embeddings)

    stats = evaluate_source("Handbuch", chunks, faiss_service, top_k=3)
    print(stats)
