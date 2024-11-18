import yaml
from statistics import mean
from typing import Literal

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


from services.retrieval import FaissService, retrieve
from core.ollama import generate_response
from domain.evaluation import Feedback
from settings.settings import settings

ypath = settings.vignettes_path
with open(ypath, "r") as file:
    vignette_yaml = yaml.safe_load(file)

llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0, format="json")
chain = llm | JsonOutputParser()


def create_prompt(retrieved_documents, background, query):
    return f"""
        You are a helpful assistant for a clinician. You will be given some information and you need to provide an answer to the question asked by the clinician based on the provided information. If you don't know the answer, you can say "I don't know" or request clarification/more information. This is for professional use only, not for patient advice.
    
        Related information:\n{''.join([f"{document}\n" for document in retrieved_documents])}
        
        Background:\n{background}
        Question:\n{query}
        
        Answer:\n
    """


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should be a JSON as follows: {{'feedback': "write a feedback for criteria", "score": "an integer number between 1 and 5"}}
4. Please do not generate any other opening, closing, and explanations. Be sure to output a valid JSON.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Output:"""

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)


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
    prompt = create_prompt(retrieved_documents, background, question)

    response = generate_response(prompt)

    eval_prompt = evaluation_prompt_template.format_messages(
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
