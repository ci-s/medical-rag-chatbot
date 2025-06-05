import sys
import os
import json
import time
from statistics import mean

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from core.document import get_document
from services.retrieval import (
    FaissService,
    retrieve_table_by_summarization,
    gather_chunks_orderly,
    reorder_flowchart_chunks,
    create_flowchart_chunks,
)
from core.chunking import chunk_document, load_saved_chunks, save_chunks, tables_to_chunks
from core.generation import describe_table_for_generation
from domain.document import ChunkType, Chunk

from settings.settings import settings
from settings import get_page_types, config, VIGNETTE_COLLECTION
from eval.combined import evaluate_single_combined, EvalResult

from eval.generation import evaluate_ragas


# TODO: do it somewhere else in init
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


ragas = config.ragas
if ragas:
    mlflow.set_experiment("Ragas Generation and Retrieval")
else:
    mlflow.set_experiment("Final Generation and Retrieval")


file_path = os.path.join(settings.data_path, settings.file_name)

text_pages, _, table_pages, _ = get_page_types()
pages = sorted(text_pages + table_pages)
print(f"Number of pages: {len(pages)}")

document = get_document(file_path, pages)

if not config.saved_chunks_path:
    raise ValueError("This setting requires manual configuration: transform_for_generation")

transform_for_generation = False

if config.saved_chunks_path:
    all_chunks = load_saved_chunks(config.saved_chunks_path)

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
    flowchart_directory = os.path.join(settings.data_path, "flowcharts")
    fchunks = create_flowchart_chunks(flowchart_directory)
    for fchunk in fchunks:
        all_chunks.append((fchunk.text, fchunk))

    all_chunks = reorder_flowchart_chunks(all_chunks)
    save_chunks(all_chunks)

faiss_service = FaissService()
faiss_service.create_index(all_chunks)
print("Total chunks: ", len(faiss_service.chunks))


def evaluate_source_combined(
    source: str,
    faiss_service: FaissService,
    document,
) -> tuple[float, float, float, list[EvalResult]]:
    """
    Evaluate both retrieval and generation for a given source using the combined evaluation function.
    """
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != source:
                continue

            eval_result = evaluate_single_combined(vignette, question, faiss_service, document)
            all_feedbacks.append(eval_result)

    try:
        avg_score = mean([float(eval_result.score) for eval_result in all_feedbacks if eval_result.score is not None])
        avg_recall = mean(
            [
                float(eval_result.retrieval_recall)
                for eval_result in all_feedbacks
                if eval_result.retrieval_recall is not None
            ]
        )
        avg_precision = mean(
            [
                float(eval_result.retrieval_precision)
                for eval_result in all_feedbacks
                if eval_result.retrieval_precision is not None
            ]
        )
    except Exception as e:
        print("Trouble calculating average scores: ", e)
        avg_score = 0
    return avg_score, avg_recall, avg_precision, all_feedbacks


# Run combined evaluation
source = "Handbuch"

if ragas:
    avg_scores, avg_recall, avg_precision, all_feedbacks = evaluate_ragas("Handbuch", faiss_service, document)
    print("Average scores for Ragas: ", avg_scores)
else:
    avg_score, avg_recall, avg_precision, all_feedbacks = evaluate_source_combined(source, faiss_service, document)

# Save results
tim = int(time.time())
output_file = f"generation_retrieval_eval_{tim}_{config.experiment_name}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(
        {
            "avg_generation_score": avg_score if not ragas else avg_scores,
            "avg_retrieval_recall": avg_recall,
            "avg_retrieval_precision": avg_precision,
            "all_feedbacks": [feedback.to_dict() for feedback in all_feedbacks],
        },
        file,
        indent=4,
        ensure_ascii=False,
    )

with mlflow.start_run():
    mlflow.log_params(config.model_dump())
    mlflow.log_params(settings.model_dump(mode="json"))
    mlflow.log_params({"num_questions": len(all_feedbacks)})

    if ragas:
        for metric, avg_score in avg_scores.items():
            mlflow.log_metric(metric, avg_score)
    else:
        mlflow.log_metric("avg_generation_score", avg_score)

    mlflow.log_metric("avg_retrieval_recall", avg_recall)
    mlflow.log_metric("avg_retrieval_precision", avg_precision)
    mlflow.set_tag("name", config.experiment_name)

    mlflow.log_artifact(output_path)

print("Results are saved in: ", output_path)
