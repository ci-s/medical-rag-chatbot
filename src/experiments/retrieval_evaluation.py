import sys
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import FaissService
from core.chunking import chunk_document
from eval.retrieval import evaluate_source

from settings.settings import settings
from settings import get_page_types

file_path = os.path.join(settings.data_path, settings.file_name)
pages, _, _, _ = get_page_types()

document = get_document(file_path, pages)

method_args = {
    "size": {"chunk_size": 512},
    # "section": {"toc_pages": [2, 3]},
    # "semantic": {},  # set NOMIC_API_KEY
}

for method, args in method_args.items():
    print(f"Method: {method}")
    chunks = chunk_document(method=method, document=document, pages=pages, **args)

    faiss_service = FaissService()
    faiss_service.create_index(chunks)

    stats = evaluate_source("Handbuch", faiss_service, top_k=3, text_only=True, include_context=False)
    print(stats)

# Questions from Handbuch: 69
# 0.2391304347826087

# Questions from Handbuch: 18
# 0.7777
