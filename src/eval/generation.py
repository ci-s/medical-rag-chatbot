from statistics import mean
from typing import Literal
import json

from services.retrieval import FaissService, retrieve
from core.model import generate_response
from core.question_answering import create_question_prompt_w_docs
from domain.evaluation import Feedback
from domain.document import Chunk
from settings import VIGNETTE_COLLECTION
from .generation_metrics import llm_as_a_judge, faithfulness, answer_relevance
from .retrieval_metrics import context_relevance
from parsing import Answer, parse_with_retry


class RAGASResult:
    def __init__(
        self,
        question_id: int,
        retrieved_documents: list[Chunk],
        generated_answer: str,
        llm_as_a_judge_param: Feedback | None,
        faithfulness_param: Feedback | None,
        answer_relevance_param: Feedback | None,
        context_relevance_param: Feedback | None,
    ):
        self.question_id = question_id
        self.retrieved_documents = retrieved_documents
        self.generated_answer = generated_answer
        self.llm_as_judge = llm_as_a_judge_param
        self.faithfulness = faithfulness_param
        self.answer_relevance = answer_relevance_param
        self.context_relevance = context_relevance_param

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "generated_answer": self.generated_answer,
            "llm_as_judge": self.llm_as_judge.to_dict() if self.llm_as_judge else None,
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "answer_relevance": self.answer_relevance.to_dict() if self.answer_relevance else None,
            "context_relevance": self.context_relevance.to_dict() if self.context_relevance else None,
        }


def evaluate_single(
    vignette_id: int,
    question_id: int,
    faiss_service: FaissService,
) -> Feedback:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    question = vignette.get_question(question_id)

    retrieved_documents = retrieve(vignette, question, faiss_service)
    system_prompt, user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)

    generated_answer = generate_response(system_prompt, user_prompt)
    generated_answer = parse_with_retry(Answer, generated_answer)

    return llm_as_a_judge(vignette, question, generated_answer.answer, retrieved_documents)


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    text_only: bool = False,
) -> tuple[int, list[Feedback]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != source:
                continue
            if text_only and question.text_only:
                all_feedbacks.append(evaluate_single(vignette.get_id(), question.get_id(), faiss_service))
            elif not text_only:
                all_feedbacks.append(evaluate_single(vignette.get_id(), question.get_id(), faiss_service))
            else:
                pass

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    try:
        avg_score = mean([float(feedback.score) for feedback in all_feedbacks if feedback.score is not None])
    except Exception as e:
        print("Trouble calculating average score: ", e)
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks


def evaluate_single_w_ragas(
    vignette_id: int,
    question_id: int,
    faiss_service: FaissService,
) -> RAGASResult:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    question = vignette.get_question(question_id)

    retrieved_documents = retrieve(vignette, question, faiss_service)
    system_prompt, user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)

    generated_answer = generate_response(system_prompt, user_prompt)
    generated_answer = parse_with_retry(Answer, generated_answer)

    return RAGASResult(
        question_id,
        retrieved_documents,
        generated_answer,
        llm_as_a_judge(vignette, question, generated_answer, retrieved_documents),
        # faithfulness(vignette, question, generated_answer, retrieved_documents),
        # answer_relevance(vignette, question, generated_answer, retrieved_documents),
        # context_relevance(vignette, question, generated_answer, retrieved_documents),
        None,
        None,
        None,
    )


def compute_average_scores(all_feedbacks: list[RAGASResult], score_keys: list[str]) -> dict:
    """
    Computes average scores for given score keys in all_feedbacks.

    Parameters:
        all_feedbacks (list): List of feedback objects containing different score categories.
        score_keys (list): List of score keys to compute averages for.

    Returns:
        dict: Dictionary with average scores for each key.
    """
    avg_scores = {}

    for key in score_keys:
        scores = [
            float(getattr(ragas_result, key).score)
            for ragas_result in all_feedbacks
            if getattr(ragas_result, key) is not None
        ]
        avg_scores[key] = mean(scores) if scores else 0.0  # Avoid StatisticsError

    return avg_scores


def evaluate_ragas(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    text_only: bool = False,
) -> tuple[int, list[RAGASResult]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != source:
                continue
            if text_only and question.text_only:
                all_feedbacks.append(evaluate_single_w_ragas(vignette.get_id(), question.get_id(), faiss_service))
            elif not text_only:
                all_feedbacks.append(evaluate_single_w_ragas(vignette.get_id(), question.get_id(), faiss_service))
            else:
                pass

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    score_keys = ["llm_as_judge", "faithfulness", "answer_relevance", "context_relevance"]
    avg_scores = compute_average_scores(all_feedbacks, score_keys)
    print("Trouble calculating average scores")
    # Add validation to score for integer between 1 and 5
    return avg_scores, all_feedbacks


def evaluate_ragas_qids(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    question_ids: list[int],
) -> tuple[int, list[RAGASResult]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_id() not in question_ids:
                continue

            all_feedbacks.append(evaluate_single_w_ragas(vignette.get_id(), question.get_id(), faiss_service))

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    score_keys = ["llm_as_judge", "faithfulness", "answer_relevance", "context_relevance"]
    avg_scores = compute_average_scores(all_feedbacks, score_keys)
    # Add validation to score for integer between 1 and 5
    return avg_scores, all_feedbacks
