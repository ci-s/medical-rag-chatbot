from domain.vignette import Question, Vignette
from domain.document import Chunk, ChunkType, Document
from domain.evaluation import Feedback, StatementResult, AnswerRelevanceResult
from core.model import generate_response
from core.embedding import embed_chunks
from prompts import (
    GENERATION_EVALUATION_PROMPT,
    EXTRACT_STATEMENTS_PROMPT,
    FAITHFULNESS_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
)
from parsing import (
    Feedback as Feedback_Parsing,
    Statements,
    ResultsResponse,
    AnswerRelevanceResultResponse,
    parse_with_retry,
)
import numpy as np

from core.chunking import load_saved_chunks
from settings.settings import config


def get_generated_flowchart_page_description(page_number: int, chunk_path: str | None = None) -> str:
    print("Using chunk text for page: ", page_number)
    if chunk_path:
        all_chunks = load_saved_chunks(chunk_path)
    else:
        all_chunks: tuple[str, list[Chunk]] = load_saved_chunks(config.saved_chunks_path)
    for text, chunk in all_chunks:
        if chunk.start_page == page_number:
            if chunk.type != ChunkType.FLOWCHART:
                raise ValueError(
                    f"Chunk is not flowchart! It is {chunk.type} pages {chunk.start_page}-{chunk.end_page}"
                )
            return chunk.text


def llm_as_a_judge(vignette: Vignette, question: Question, generated_answer: str, document: Document) -> Feedback:
    eval_prompt = GENERATION_EVALUATION_PROMPT.format(
        reference_pages=" ".join(
            [
                document.get_processed_content(page_number)
                if document.get_processed_content(page_number)
                else get_generated_flowchart_page_description(page_number)
                for page_number in question.get_reference_pages()
            ]
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


def extract_statements_from_answer(vignette: Vignette, question: Question, generated_answer: str) -> list[str]:
    user_prompt = f"""
    Background: {vignette.get_background()}
    Question: {question.get_question()}
    Answer: {generated_answer}
    """
    response = generate_response(user_prompt, EXTRACT_STATEMENTS_PROMPT)
    try:
        statements = parse_with_retry(Statements, response)
        return statements.statements
    except Exception as e:
        print("Problematic parsing:", e)
        raise e


def faithfulness(
    vignette: Vignette, question: Question, generated_answer: str, retrieved_documents: list[Chunk]
) -> StatementResult:
    statements = extract_statements_from_answer(vignette, question, generated_answer)
    user_prompt = f"""
    Related information: {"\n".join([retrieved_doc.text for retrieved_doc in retrieved_documents])}
    Background:{vignette.get_background()}
    Question: {question.get_question()}
    Statements:\n{"\n".join(["Statement: " + statement for statement in statements])}
    """
    response = generate_response(user_prompt, FAITHFULNESS_PROMPT)
    try:
        result_response = parse_with_retry(ResultsResponse, response)
        results = result_response.results
    except Exception as e:
        print("Problematic parsing:", e)
        raise e

    supported_statements = sum(1 for result in results if result.verdict == "yes")
    total_statements = len(statements)

    if total_statements == 0:
        print("No statements extracted.")
        faithfulness_score = 0
    else:
        faithfulness_score = supported_statements / total_statements

    return StatementResult(
        statements=statements,
        explanations=[result.explanation for result in results],
        verdicts=[result.verdict for result in results],
        question_id=question.get_id(),
        score=faithfulness_score,
        feedback="",
        generated_answer=generated_answer,
    )


def calculate_similarity(question: str, generated_questions: list[str]):
    question_vec = embed_chunks(question, task_type="search_query")
    gen_question_vec = embed_chunks(generated_questions, task_type="search_query")
    norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
    return np.dot(gen_question_vec, question_vec.T) / norm  # TODO: try wo reshape


def answer_relevance(
    vignette: Vignette, question: Question, generated_answer: str, retrieved_documents: list[Chunk]
) -> AnswerRelevanceResult:
    user_prompt = f"""
        Background: {vignette.get_background()}
        
        Answer: {generated_answer}
    """

    generated_questions = []

    response = generate_response(user_prompt, ANSWER_RELEVANCE_PROMPT)
    try:
        response = parse_with_retry(AnswerRelevanceResultResponse, response)
    except Exception as e:
        print("Bad parsing in answer_relevance:", e)
        raise e

    generated_questions.extend(response.questions)
    is_noncommittal = bool(response.noncommittal)

    cosine_sim = calculate_similarity(question.get_question(), generated_questions)
    if is_noncommittal:
        score = 0
    else:
        score = cosine_sim.mean()

    feedback_text = f"Generated questions: {generated_questions}, Noncommittal: {is_noncommittal}"

    return AnswerRelevanceResult(
        answer=generated_answer,
        generated_questions=generated_questions,
        noncommittal=is_noncommittal,
        question_id=question.get_id(),
        score=score,
        feedback=feedback_text,
        generated_answer=generated_answer,
    )
