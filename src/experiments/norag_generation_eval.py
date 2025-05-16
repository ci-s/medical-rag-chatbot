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
from parsing import get_format_instructions, parse_with_retry, Answer, ReasoningAnswer, ThinkingAnswer
from prompts import QUESTION_PROMPT, QUESTION_PROMPT_w_THINKING, QUESTION_PROMPT_w_REASONING
from core.generation import create_user_question_prompt
from core.utils import replace_abbreviations
from prompts import GENERATION_EVALUATION_PROMPT
from parsing import Feedback as Feedback_Parsing
from domain.vignette import Question, Vignette
from domain.document import Document
from core.document import get_document
from eval.generation import llm_as_a_judge

import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Final Generation and Retrieval")

# IMPORTANT: This script was revised to include reasoning and thinking

NORAG_USER_PROMPT = """
    
    {user_prompt}
    
    Answer:\n
"""


def evaluate_single(vignette_id: int, question_id: int, document: Document) -> tuple[Feedback, str | None]:
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

    if config.reasoning:
        system_prompt = QUESTION_PROMPT_w_REASONING.format(format_instructions=get_format_instructions(ReasoningAnswer))
    elif config.thinking:
        system_prompt = QUESTION_PROMPT_w_THINKING.format(format_instructions=get_format_instructions(ThinkingAnswer))
    else:
        system_prompt = QUESTION_PROMPT.format(format_instructions=get_format_instructions(Answer))

    generated_answer = generate_response(user_prompt, system_prompt)

    if config.thinking:
        parsed_response = parse_with_retry(ThinkingAnswer, generated_answer)
        generated_answer = parsed_response.answer
        reasoning = parsed_response.thinking
    elif config.reasoning:
        parsed_response = parse_with_retry(ReasoningAnswer, generated_answer)
        generated_answer = parsed_response.answer
        reasoning = parsed_response.reasoning
    else:
        parsed_response = parse_with_retry(Answer, generated_answer)
        generated_answer = parsed_response.answer
        reasoning = None

    return llm_as_a_judge(vignette, question, generated_answer, document), reasoning


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
            feedback, reasoning = evaluate_single(vignette.get_id(), question.get_id(), document)
            feedback_dict = feedback.to_dict()
            if reasoning is not None:
                feedback_dict["reasoning"] = reasoning
            all_feedbacks.append(feedback_dict)

    print(f"Questions from {source}: {len([all_feedbacks for s in all_feedbacks if s is not None])}")

    try:
        avg_score = mean([float(feedback["score"]) for feedback in all_feedbacks if feedback["score"] is not None])
    except Exception as e:
        print("Trouble calculating average score: ", e)
        avg_score = 0
    # Add validation to score for integer between 1 and 5
    return avg_score, all_feedbacks


file_path = os.path.join(settings.data_path, settings.file_name)

text_pages, _, table_pages, _ = get_page_types()
pages = sorted(text_pages + table_pages)
print(f"Number of pages: {len(pages)}")

document = get_document(file_path, pages)
result_dict = []


avg_score, all_feedbacks = evaluate_source("Handbuch", document)
tim = int(time.time())

result_dict = {
    "config": config.model_dump(),
    "settings": settings.model_dump(mode="json"),
    "all_feedbacks": all_feedbacks,
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
