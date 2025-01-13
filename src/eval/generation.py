from statistics import mean
from typing import Literal
import json

from services.retrieval import FaissService, retrieve
from core.model import generate_response, create_question_prompt_w_docs
from domain.evaluation import Feedback
from settings import VIGNETTE_COLLECTION
from prompts import GENERATION_EVALUATION_PROMPT


def evaluate_single(vignette_id: int, question_id: int, faiss_service: FaissService, top_k: int = 3) -> Feedback:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    background = vignette.get_background()

    question = vignette.get_question(question_id)
    answer = question.get_answer()

    print(f"Question {question_id}: {question.get_question()}")
    if background is None or answer is None:
        print(f"Either background or answer not found for vignette id {vignette_id} and question id {question_id}")
        return None

    retrieved_documents = retrieve(question.get_question(), faiss_service, top_k=top_k)
    user_prompt = create_question_prompt_w_docs(retrieved_documents, vignette, question)

    response = generate_response(user_prompt)

    eval_prompt = GENERATION_EVALUATION_PROMPT.format(
        instruction=f"""        
            Related information:\n{"".join([f"{docu}\n" for docu in retrieved_documents])}
                    
            Background:\n{background}
            Question:\n{question.get_question()}
            """,
        response=response,
        reference_answer=answer,
    )

    eval_result = generate_response(eval_prompt)
    if not isinstance(eval_result, dict):
        raise TypeError(f"Expected eval_result to be a dictionary after parsing. Got -> {eval_result}")
    return Feedback(eval_result["feedback"], eval_result["score"])


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    top_k: int = 3,
    text_only: bool = False,
) -> int:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        for question in vignette.get_questions():
            if question.get_source() != source:
                continue
            if text_only and question.text_only:
                all_feedbacks.append(evaluate_single(vignette.get_id(), question.get_id(), faiss_service, top_k))
            elif not text_only:
                all_feedbacks.append(evaluate_single(vignette.get_id(), question.get_id(), faiss_service, top_k))
            else:
                pass

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    try:
        avg_score = mean([int(feedback.score) for feedback in all_feedbacks if feedback.score is not None])
    except:
        print("Trouble calculating average score")
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks
