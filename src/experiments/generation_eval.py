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

chunks_saved = True
transform_for_generation = False

if chunks_saved:
    _all_chunks = load_saved_chunks(config.saved_chunks_path)
    all_chunks = reorder_flowchart_chunks(_all_chunks)

    if transform_for_generation:
        for text, chunk in all_chunks:
            if chunk.type == ChunkType.TABLE:
                chunk.text = describe_table_for_generation(chunk, document)

        save_chunks(all_chunks)
else:
    chunks = chunk_document(method=config.chunk_method, document=document, pages=pages, chunk_size=config.chunk_size)

    with open(settings.table_texts_path, "r") as file:
        tables = json.load(file)
        table_chunks = tables_to_chunks(tables)

    _all_chunks = gather_chunks_orderly(chunks, table_chunks)
    all_chunks = []
    for chunk in _all_chunks:
        if chunk.type == ChunkType.TEXT:
            all_chunks.append((chunk.text, chunk))
        elif chunk.type == ChunkType.TABLE:
            summary = retrieve_table_by_summarization(chunk, document)
            if transform_for_generation:
                chunk.text = describe_table_for_generation(chunk, document)
            all_chunks.append((summary, chunk))
        else:
            raise ValueError(f"Chunk type {chunk.type} is not implemented yet.")
    save_chunks(all_chunks)

faiss_service = FaissService()
faiss_service.create_index(all_chunks)
print("Total chunks: ", len(faiss_service.chunks))

chunk_configurations = [
    ("size", 1024, 1),
    # ("size", 256, 3),
    # ("section_and_size", 256, 2),
    # ("section_and_size", 256, 3),
    # ("section_and_size", 1024, 1),
    # ("section_and_size", 256, 3),
]

for custom_config in chunk_configurations:
    result_dicts = []
    config.chunk_method = custom_config[0]
    config.chunk_size = custom_config[1]
    config.surrounding_chunk_length = custom_config[2]
    # config.reasoning = True

    for optim_method in [
        "hypothetical_document",
    ]:  # None, "hypothetical_document", "decomposing", "paraphrasing", "stepback"
        if optim_method:
            config.optimization_method = optim_method
            config.use_original_query_only = False
        else:
            config.optimization_method = None
            config.use_original_query_only = True

        # avg_score, all_feedbacks = evaluate_ragas("Handbuch", faiss_service)
        avg_score, all_feedbacks = evaluate_source("Handbuch", faiss_service, document)
        tim = int(time.time())

        result_dict = {
            "config": config.model_dump(),
            "settings": settings.model_dump(mode="json"),
            "all_feedbacks": [fb.to_dict() for fb in all_feedbacks],
            "avg_score": avg_score,
        }

        output_file = f"generation_eval_{tim}_{config.experiment_name}_{str(optim_method)}.json"
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
