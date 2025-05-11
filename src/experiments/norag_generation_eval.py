import sys
import os
import json
import time
from typing import Literal
from statistics import mean

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from settings.settings import settings
from settings import config, get_page_types

from core.model import generate_response
from domain.evaluation import Feedback
from settings import VIGNETTE_COLLECTION
from parsing import get_format_instructions, parse_with_retry, Answer
from prompts import QUESTION_PROMPT
from core.generation import create_user_question_prompt
from core.utils import replace_abbreviations
from prompts import GENERATION_EVALUATION_PROMPT
from parsing import Feedback as Feedback_Parsing
from domain.vignette import Question, Vignette
from domain.document import Document
from core.document import get_document

import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Phase 1 Generation")
NORAG_USER_PROMPT = """
    
    {user_prompt}
    
    Answer:\n
"""


def llm_as_a_judge(vignette: Vignette, question: Question, generated_answer: str, document: Document) -> Feedback:
    eval_prompt = GENERATION_EVALUATION_PROMPT.format(
        reference_pages=" ".join(
            [document.get_processed_content(page_number) for page_number in question.get_reference_pages()]
        ),
        question=f"""
            Background:\n{vignette.background}
            Question:\n{question.get_question()}
            """,
        reference_answer=question.get_answer(),
        generated_answer=generated_answer,
    )

    eval_result = generate_response(eval_prompt)
    try:
        eval_result = parse_with_retry(Feedback_Parsing, eval_result)
    except Exception as e:
        print("Problematic parsing in llm as a judge:", e)
        raise e

    return Feedback(question.get_id(), eval_result.feedback, eval_result.score, generated_answer)


def evaluate_single(vignette_id: int, question_id: int, document: Document) -> Feedback:
    vignette = VIGNETTE_COLLECTION.get_vignette_by_id(vignette_id)
    questions = vignette.get_questions()
    print(f"Questions in vignette {vignette_id}: {len(questions)}")
    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    question = vignette.get_question(question_id)

    user_prompt = create_user_question_prompt(vignette, question)

    user_prompt = NORAG_USER_PROMPT.format(
        user_prompt=user_prompt,
    )
    user_prompt, _ = replace_abbreviations(user_prompt)
    system_prompt = QUESTION_PROMPT.format(format_instructions=get_format_instructions(Answer))

    print("User prompt: ", user_prompt)
    print("System prompt: ", system_prompt)
    generated_answer = generate_response(user_prompt, system_prompt)
    generated_answer = parse_with_retry(Answer, generated_answer)

    return llm_as_a_judge(vignette, question, generated_answer.answer, document)


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    document: Document,
) -> tuple[int, list[Feedback]]:
    all_feedbacks = []

    for vignette in VIGNETTE_COLLECTION.get_vignettes():
        print(f"Vignette: {vignette.get_id()}")
        for question in vignette.get_questions():
            print(f"Question: {question.get_id()}")
            if question.get_source() != source:
                continue
            all_feedbacks.append(evaluate_single(vignette.get_id(), question.get_id(), document))

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    try:
        avg_score = mean([float(feedback.score) for feedback in all_feedbacks if feedback.score is not None])
    except Exception as e:
        print("Trouble calculating average score: ", e)
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks


file_path = os.path.join(settings.data_path, settings.file_name)

if config.filter_questions:
    if config.filter_questions == ["Text"]:
        pages, _, _, _ = get_page_types()
    elif config.filter_questions == ["Table"]:
        _, _, pages, _ = get_page_types()
    elif config.filter_questions == ["Flowchart"]:
        _, pages, _, _ = get_page_types()
    else:
        raise ValueError("Multiple filter_questions value is not configured for page types yet")
else:
    pages = list(range(7, 109))

document = get_document(file_path, pages)
result_dict = []


avg_score, all_feedbacks = evaluate_source("Handbuch", document)
tim = int(time.time())

result_dict = {
    "config": config.model_dump(),
    "settings": settings.model_dump(mode="json"),
    "all_feedbacks": [fb.to_dict() for fb in all_feedbacks],
    "avg_score": avg_score,
}


output_file = f"norag_generation_eval_{tim}.json"
output_path = os.path.join(settings.results_path, output_file)
with open(output_path, "w") as file:
    json.dump(result_dict, file, indent=4, ensure_ascii=False)


with mlflow.start_run():
    mlflow.log_params(config.model_dump())
    mlflow.log_params(settings.model_dump(mode="json"))
    mlflow.log_params({"num_questions": len(all_feedbacks)})
    mlflow.log_metric("avg_score", avg_score)
    mlflow.set_tag("NoRAG", "All eval")

    mlflow.log_artifact(output_path)

print("Results are saved in: ", output_path)
