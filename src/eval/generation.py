from statistics import mean
from typing import Literal
import json

from services.retrieval import FaissService, retrieve
from core.model import generate_response, create_question_prompt_w_docs
from domain.evaluation import Feedback
from settings import VIGNETTE_COLLECTION
from .generation_metrics import llm_as_a_judge, faithfulness, answer_relevance, context_relevance


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
    user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)

    generated_answer = generate_response(user_prompt)
    # return llm_as_a_judge(vignette, question, generated_answer, retrieved_documents)
    return faithfulness(vignette, question, generated_answer, retrieved_documents)


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
    except:
        print("Trouble calculating average score")
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks
