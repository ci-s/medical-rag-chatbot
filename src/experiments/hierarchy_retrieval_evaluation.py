import sys
import os
import json
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import (
    HierarchicalFaissService,
)
from core.chunking import load_saved_chunks
from eval.retrieval import evaluate_source


from settings.settings import settings
from settings import get_page_types, config


import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Phase 2 Retrieval")


file_path = os.path.join(settings.data_path, settings.file_name)

text_pages, _, table_pages, _ = get_page_types()
pages = sorted(text_pages + table_pages)
print(f"Number of pages: {len(pages)}")

document = get_document(file_path, pages)
method = "section_and_size"
result_dicts = []
for optim_method in [
    None,
    # "hypothetical_document",
    # "decomposing",
    # "paraphrasing",
    # "stepback",
]:  # , "hypothetical_document", "decomposing", "paraphrasing", "stepback"
    if optim_method:
        config.optimization_method = optim_method
        config.use_original_query_only = False
    else:
        config.optimization_method = None
        config.use_original_query_only = True

    chunks_saved = True
    if chunks_saved:
        print("Chunks are already saved. Loading them.")
        all_chunks = load_saved_chunks(config.saved_chunks_path)
    else:
        raise ValueError("Chunks are not saved. Please set chunks_saved to True.")

    # faiss_service = FaissService()
    faiss_service = HierarchicalFaissService()
    faiss_service.create_index(all_chunks)
    overall_stat, all_stats = evaluate_source("Handbuch", faiss_service)

    result_dict = {
        "config": config.model_dump(),
        "settings": settings.model_dump(mode="json"),
        "all_stats": [stat.to_dict() for stat in all_stats],
        "overall": overall_stat.to_dict(),
    }

    output_file = f"retrieval_eval_{int(time.time())}.json"
    output_path = os.path.join(settings.results_path, output_file)
    with open(output_path, "w") as file:
        json.dump(result_dict, file, indent=4, ensure_ascii=False)

    with mlflow.start_run():
        mlflow.log_params(config.model_dump())
        mlflow.log_params(settings.model_dump(mode="json"))
        mlflow.log_params({"num_questions": len(all_stats)})
        mlflow.log_metric("avg recall", overall_stat.recall)
        mlflow.set_tag("name", config.experiment_name)

        mlflow.log_artifact(output_path)

    print("Results are saved in: ", output_path)
