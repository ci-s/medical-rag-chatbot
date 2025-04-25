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
    create_flowchart_chunks,
)
from core.chunking import chunk_document, load_saved_chunks, tables_to_chunks, save_chunks
from eval.retrieval import evaluate_source
from domain.document import ChunkType


from settings.settings import settings
from settings import get_page_types, config


import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Phase 2 Retrieval")


#    - Take footnotes into account when relevant.
# prompt = """You'll be given a page containing a flowchart from a medical document that clinicians use to make decisions.

#     Your task is to generate a detailed description in German that maximizes retrieval effectiveness. Your response should follow these guidelines:

#     - Provide a concise yet informative overview of what the flowchart represents.
#     - Include key medical concepts and terms clinicians might search for.
#     - Use synonyms and alternative phrasing to capture diverse query formulations.

#     Your response must follow this JSON format strictly:

#     {
#         "description": "<Your overview in German in one single string>"
#     } <END OF JSON>

#     Do not say anything else. Make sure the response is a valid JSON. Stop immediately at <END OF JSON>.\n
# """


file_path = os.path.join(settings.data_path, settings.file_name)


# if config.filter_questions:
#     text_pages, _, table_pages, _ = get_page_types()
#     pages = []
#     if "Text" in config.filter_questions:
#         pages.extend(text_pages)
#     if "Table" in config.filter_questions:
#         pages.extend(table_pages)
#     # No need for flowchart pages because they are processed as images
#     pages = sorted(pages)
# else:
#     pages = list(range(7, 109))
text_pages, _, table_pages, _ = get_page_types()
pages = sorted(text_pages + table_pages)
print(f"Number of pages: {len(pages)}")

document = get_document(file_path, pages)
method_args = {
    # "semantic": {},  # set NOMIC_API_KEY
    # "section": {"toc": None},
    # "section_and_size": {"toc": None, "chunk_size": config.chunk_size},
    "size": {"chunk_size": config.chunk_size},
    # "size": {}
}


result_dicts = []
for method, args in method_args.items():
    for optim_method in [
        # None,
        "hypothetical_document",
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

        config.chunk_method = method

        chunks_saved = True
        if chunks_saved:
            print("Chunks are already saved. Loading them.")
            all_chunks = load_saved_chunks(config.saved_chunks_path)
            # all_chunks = reorder_flowchart_chunks(_all_chunks)
        else:
            print(f"Method: {method}")
            chunks = chunk_document(method=method, document=document, pages=pages, **args)

            with open(settings.table_texts_path, "r") as file:
                tables = json.load(file)
            table_chunks = tables_to_chunks(tables)

            _all_chunks = gather_chunks_orderly(chunks, table_chunks)
            all_chunks = []
            for chunk in _all_chunks:
                if chunk.type == ChunkType.TEXT:
                    all_chunks.append((chunk.text, chunk))
                elif chunk.type == ChunkType.TABLE:
                    all_chunks.append((retrieve_table_by_summarization(chunk, document), chunk))
                else:
                    raise ValueError(f"Chunk type {chunk.type} is not implemented yet.")

            save_chunks(all_chunks)

        # flowchart_directory = os.path.join(settings.data_path, "flowcharts")
        # fchunks = create_flowchart_chunks(flowchart_directory)
        # for fchunk in fchunks:
        #     all_chunks.append((fchunk.text, fchunk))
        # save_chunks(all_chunks, output_filename="flowchart_longer_hori.json")

        faiss_service = FaissService()
        faiss_service.create_index(all_chunks)
        print("Total chunks: ", len(faiss_service.chunks))
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
