import yaml
from statistics import mean
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


from services.retrieval import FaissService, retrieve
from core.ollama import generate_response
from domain.evaluation import Feedback
from settings.settings import settings
from prompts import EVALUATION_PROMPT_TEMPLATE, create_question_prompt

ypath = settings.vignettes_path
with open(ypath, "r") as file:
    vignette_yaml = yaml.safe_load(file)

llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0, format="json")
chain = llm | JsonOutputParser()


def evaluate_single(vignette_id: int, question: str, faiss_service: FaissService, top_k: int = 3) -> Feedback:
    questions = vignette_yaml["vignettes"][vignette_id].get("questions", None)

    if questions is None:
        print(f"Vignette with id {vignette_id} not found")
        return None

    background = None
    answer = None
    for question_obj in questions:
        if question == question_obj["question"]:
            background = vignette_yaml["vignettes"][vignette_id]["background"]
            answer = question_obj["answer"]

    if background is None or answer is None:
        print(f"Question not found in vignette {vignette_id}")
        return None

    retrieved_documents = retrieve(question, faiss_service, top_k=3)
    prompt = create_question_prompt(retrieved_documents, background, question)

    response = generate_response(prompt)

    eval_prompt = EVALUATION_PROMPT_TEMPLATE.format_messages(
        instruction=f"""        
            Related information:\n{''.join([f"{docu}\n" for docu in retrieved_documents])}
                    
            Background:\n{background}
            Question:\n{question}
            """,
        response=response,
        reference_answer=answer,
    )
    eval_result = chain.invoke(eval_prompt)
    return Feedback(eval_result["feedback"], eval_result["score"])


def evaluate_source(
    source: Literal["Handbuch", "Antibiotika"],
    faiss_service: FaissService,
    top_k: int = 3,
) -> int:
    all_feedbacks = []

    for vignette in vignette_yaml["vignettes"]:
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
