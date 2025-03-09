from statistics import mean
from typing import Literal

from domain.document import Chunk
from domain.vignette import Question
from domain.evaluation import Stats
from services.retrieval import FaissService, retrieve
from settings import VIGNETTE_COLLECTION
from .retrieval_metrics import recall, precision


def get_references_w_id(vignette_id, question_id) -> list[int]:
    return VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id).get_question(question_id).get_reference_pages()


def get_references(query: str) -> list[int]:
    # Assuming there are no duplicate questions
    for vignette in VIGNETTE_COLLECTION.vignettes:
        for question in vignette.get_questions():
            if query == question.get_question():
                return question.get_reference_pages()


def evaluate_single(question: Question, retrieved_passages: list[Chunk]) -> Stats:
    reference_pages = get_references(question.get_question())
    return Stats(
        question_id=question.get_id(),
        recall=recall(retrieved_passages, reference_pages),
        precision=precision(retrieved_passages, reference_pages),
        retrieved_documents=retrieved_passages,
    )


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
) -> tuple[Stats, list[Stats]]:
    all_stats = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            print(f"Processing question {question.get_id()} from {source}")
            if question.get_source() != source:
                continue

            retrieved_documents = retrieve(vignette, question, faiss_service)
            all_stats.append(evaluate_single(question, retrieved_documents))

    print(f"Questions from {source}: {len([all_stats for s in all_stats if s is not None])}")
    return Stats(
        question_id=-1,
        recall=mean([stat.recall for stat in all_stats if stat is not None]),
        precision=mean([stat.precision for stat in all_stats if stat is not None]),
    ), all_stats
