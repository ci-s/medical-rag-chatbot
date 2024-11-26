from statistics import mean
from typing import Literal
import json

from services.retrieval import FaissService, retrieve
from core.model import generate_response
from domain.evaluation import Feedback
from settings import VIGNETTE_YAML
from prompts import GENERATION_EVALUATION_PROMPT, create_question_prompt


def evaluate_single(vignette_id: int, question: str, faiss_service: FaissService, top_k: int = 3) -> Feedback:
    questions = VIGNETTE_YAML["vignettes"][vignette_id].get("questions", None)

    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    background = None
    answer = None
    for question_obj in questions:
        if question == question_obj["question"]:
            background = VIGNETTE_YAML["vignettes"][vignette_id]["background"]
            answer = question_obj["answer"]

    if background is None or answer is None:
        print(f"Question not found in vignette {vignette_id}")
        return None

    retrieved_documents = retrieve(question, faiss_service, top_k=top_k)
    prompt = create_question_prompt(retrieved_documents, background, question)

    response = generate_response(prompt)

    eval_prompt = GENERATION_EVALUATION_PROMPT.format(
        instruction=f"""        
            Related information:\n{''.join([f"{docu}\n" for docu in retrieved_documents])}
                    
            Background:\n{background}
            Question:\n{question}
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
) -> int:
    all_feedbacks = []

    for vignette in VIGNETTE_YAML["vignettes"]:
        for question in vignette["questions"]:
            if question["source"] != source:
                continue

            all_feedbacks.append(evaluate_single(vignette["id"], question["question"], faiss_service, top_k))

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    try:
        avg_score = mean([int(feedback.score) for feedback in all_feedbacks if feedback.score is not None])
    except:
        print("Trouble calculating average score")
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks
