import yaml
from statistics import mean
from typing import Literal

from domain.document import Chunk
from domain.evaluation import Stats
from services.retrieval import FaissService, retrieve
from settings import VIGNETTE_COLLECTION
from .retrieval_metrics import recall, precision


def get_references_w_id(vignette_id, question_id) -> list[int]:
    return VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id).get_question(question_id).get_reference()


def get_references(query: str) -> list[int]:
    # Assuming there are no duplicate questions
    for vignette in VIGNETTE_COLLECTION.vignettes:
        for question in vignette.get_questions():
            if query == question.get_question():
                return question.get_reference()


def evaluate_single(query: str, retrieved_passages: list[Chunk]) -> Stats | None:
    reference_pages = get_references(query)
    print("Reference pages are: ", reference_pages)
    print(
        "Retrieved pages are: ",
        [
            str(retrieved_passage.start_page) + "-" + str(retrieved_passage.end_page)
            for retrieved_passage in retrieved_passages
        ],
    )
    return recall(retrieved_passages, reference_pages)


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    text_only: bool = False,
) -> int:
    all_stats = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != source:
                continue

            if text_only and question.text_only:
                retrieved_documents = retrieve(vignette, question, faiss_service)
                all_stats.append(evaluate_single(question.get_question(), retrieved_documents))
            elif not text_only:
                retrieved_documents = retrieve(vignette, question, faiss_service)
                all_stats.append(evaluate_single(question.get_question(), retrieved_documents))
            else:
                pass

    print(f"Questions from {source}: {len([all_stats for s in all_stats if s is not None])}")

    return mean([stat.pct for stat in all_stats if stat is not None])
