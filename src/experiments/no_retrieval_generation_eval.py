import sys
import os
import json
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import (
    FaissService,
    retrieve_table_by_summarization,
    gather_chunks_orderly,
    reorder_flowchart_chunks,
)
from core.chunking import chunk_document, load_saved_chunks, save_chunks, tables_to_chunks
from core.generation import describe_table_for_generation
from domain.document import ChunkType, Chunk

from settings.settings import settings
from settings import get_page_types, config
from eval.generation import evaluate_single, evaluate_source, evaluate_ragas

# TODO: do it somewhere else in init
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Phase 2 Generation")

file_path = os.path.join(settings.data_path, settings.file_name)

text_pages, _, table_pages, _ = get_page_types()
pages = sorted(text_pages + table_pages)
print(f"Number of pages: {len(pages)}")

document = get_document(file_path, pages)


result_dicts = []

# Some config params are not valid for this case, so we set them correctly or to None here
config.chunk_method = None
config.chunk_size = None
config.surrounding_chunk_length = None
config.optimization_method = None
config.use_original_query_only = True
config.top_k = None

faiss_service = None
# avg_score, all_feedbacks = evaluate_ragas("Handbuch", faiss_service)
avg_score, all_feedbacks = evaluate_source("Handbuch", faiss_service, document, use_references_directly=True)
tim = int(time.time())

result_dict = {
    "config": config.model_dump(),
    "settings": settings.model_dump(mode="json"),
    "all_feedbacks": [fb.to_dict() for fb in all_feedbacks],
    "avg_score": avg_score,
}

output_file = f"generation_eval_{tim}_{config.experiment_name}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dict, file, indent=4, ensure_ascii=False)

with mlflow.start_run():
    mlflow.log_params(config.model_dump())
    mlflow.log_params(settings.model_dump(mode="json"))
    mlflow.log_params({"num_questions": len(all_feedbacks)})
    mlflow.log_metric("avg_score", avg_score)
    mlflow.set_tag("name", config.experiment_name)

    mlflow.log_artifact(output_path)

result_dicts.append(result_dict)
print("Results are saved in: ", output_path)
