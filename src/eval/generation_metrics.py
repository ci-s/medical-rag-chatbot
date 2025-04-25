import json

from domain.vignette import Question, Vignette
from domain.document import Chunk, ChunkType, Document
from domain.evaluation import Feedback, StatementResult, AnswerRelevanceResult
from core.model import generate_response
from core.embedding import embed_chunks
from prompts import GENERATION_EVALUATION_PROMPT
from parsing import (
    Feedback as Feedback_Parsing,
    Statements,
    ResultsResponse,
    AnswerRelevanceResultResponse,
    parse_with_retry,
)
import numpy as np

from core.chunking import Chunk, load_saved_chunks
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
    system_prompt = """
    You are given three pieces of text: a background, a question and an answer. The background provides important medical context, while the answer might contain crucial information, such as specific values, guidelines, recommended treatments or other key data points. Your task is to extract short, fully understandable facts that is conveyed by the answer related to the background.

    The output should contain statements that:
        - Directly link the information provided in the answer to the clinical details in the background.
        - Avoid extracting information from the background that does not directly relate to the answer.
        - Do not generate vague statements i.e. "The patient should be treat with X." cannot be verified.
        - Ensure no pronouns are used in each statement.
        
    Only provide the json and say nothing else.
    
    Here is an example:
    
    Background: Es erfolgt die Vorstellung eines 81-jährigen Patienten unter dem Verdacht
    auf einen Schlaganfall. Der Patient sei etwa 45 Minuten vor Vorstellung am Boden
    liegend vorgefunden worden und habe nicht mehr aufstehen können. Zuletzt wohlauf
    (last seen well) war der Patient 20h vor Vorstellung. An Vorerkrankungen ist ein
    Bluthochdruck, ein Vorhofflimmern unter Antikoagulation mit Apixaban und eine
    Hypercholisterinämie bekannt. In der körperlichen Untersuchung ist die Patientin
    wach und bietet eine nicht-flüssige Aphasie mit hochgradiger brachiofaziale Hemiparese
    rechts und rechtsseitg positivem Babinski-Zeichen (NIHSS 12). Der Blutdruck liegt
    bei 167/87. Eine multimodale CT Bildgebung zeigt ein keine größere Infarktdemarkation.
    Die A. Cerebri media links ist verschlossen mit nach geschaltetem Perfusionsdefizit.
    Ein Notfalllabor ergibt einen INR von 1,2, eine aPTT von 28. Die Thrombozyten
    liegen bei 189.000/ µl.
    Question: Welche Maßnahme sollte nun erfolgen?
    Answer: Mechanische Rekanalisation / endovaskuläre Thrombektomie
    {
        "statements": [
            "Mechanical recanalization / thrombectomy is needed due to left middle cerebral artery occlusion and perfusion deficit.",
            "The intervention is recommended based on acute symptoms, including severe hemiparesis and non-fluent aphasia.",
            "The patient qualifies for the mechanical recanalization procedure due to the absence of major infarction on CT imaging."
        ]
        }
    
    Do not say anything else. Make sure the response is a valid JSON.
    """

    user_prompt = f"""
    Background: {vignette.get_background()}
    Question: {question.get_question()}
    Answer: {generated_answer}
    """
    response = generate_response(system_prompt, user_prompt)
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
    system_prompt = """
    Consider the given context and following statements, then determine whether they are supported by the information present in the related information or background. Provide a brief explanation for each statement before arriving at the verdict (yes/no). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

    Example:
    Related information:
    The Earth revolves around the Sun and completes one full orbit in approximately 365.25 days.
    Cheese is a dairy product made by curdling milk using bacteria or enzymes.
    Tigers are carnivorous animals, known for their distinctive orange and black stripes.

    Background: 
    A zoo exhibit is being designed to educate visitors about animal habitats and diets, focusing on Asian carnivores and their roles in ecosystems.

    Question:
    Because tigers are primarily found in Asia, they are considered for the exhibit. Is there anything that would make them unsuitable?
    
    Statements:
    1. "Tigers are herbivores."
    2. "The Earth takes about 365 days to orbit the Sun."
    3. "Cheese is a natural food source for tigers in the wild."
    4. "Tigers often exist in Asia."

    Output:
    {
        "results": [
            {"statement": "Tigers are herbivores.", "verdict": "no", "explanation": "The related information explicitly states that tigers are carnivorous animals."},
            {"statement": "The Earth takes about 365 days to orbit the Sun.", "verdict": "yes", "explanation": "The related information confirms that the Earth completes an orbit in approximately 365.25 days."},
            {"statement": "Cheese is a natural food source for tigers in the wild.", "verdict": "no", "explanation": "There is no evidence that it is a food source for tigers."},
            {"statement": "Tigers often exist in Asia.", "verdict": "yes", "explanation": "The question itself mentions that tigers are primarily found in Asia."}
        ]
    }
    
    Do not say anything else. Make sure the response is a valid JSON.\n
    """
    user_prompt = f"""
    Related information: {"\n".join([retrieved_doc.text for retrieved_doc in retrieved_documents])}
    Background:{vignette.get_background()}
    Question: {question.get_question()}
    Statements:\n{"\n".join(["Statement: " + statement for statement in statements])}
    """
    response = generate_response(system_prompt, user_prompt)
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
    gen_question_vec = embed_chunks(generated_questions, task_type="search_document")
    norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
    return np.dot(gen_question_vec, question_vec.T) / norm  # TODO: try wo reshape


def answer_relevance(
    vignette: Vignette, question: Question, generated_answer: str, retrieved_documents: list[Chunk]
) -> AnswerRelevanceResult:
    system_prompt = """
        Generate 3 different questions for the given answer and the background and identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers. Do not deviate from the specified format.
        
        ## Examples
        Background:
        Es erfolgt die Vorstellung eines 72-jährigen Patienten unter dem Verdacht auf einen Schlaganfall. Der Patient wurde etwa 1 Stunde vor der Aufnahme von seiner Ehefrau bewusstlos auf dem Boden gefunden. Bei der Anamnese berichtet die Ehefrau, dass der Patient seit mehreren Jahren an Vorhofflimmern leidet und mit Apixaban behandelt wird. In der körperlichen Untersuchung zeigt der Patient eine vollständige Hemiparese der rechten Körperhälfte und eine globale Aphasie. Der Blutdruck liegt bei 200/100, die Herzfrequenz ist unregelmäßig bei 110/min.
        Eine cCT-Bildgebung zeigt einen frühzeitigen Verschluss der linken A. cerebri media, ohne Hinweise auf eine intrakranielle Blutung. Ein Notfalllabor ergibt einen INR von 2,0, eine aPTT von 35 und Thrombozyten von 150.000/µl. Die Blutzuckerwerte liegen bei 140 mg/dl.
        
        Answer: Die Empfehlung ist eine sofortige intravenöse Thrombolyse mit Alteplase, falls keine Kontraindikationen vorliegen, und eine mechanische Thrombektomie aufgrund des Verschlusses der A. cerebri media.
        {
            "questions": ["Welche therapeutische Maßnahme sollte zur Wiederherstellung der Durchblutung in diesem Fall erfolgen?", 
                "Welche Behandlung ist erforderlich, um die Durchblutung in diesem Fall wiederherzustellen?", 
                "Welche Therapieoptionen kommen zur Wiederherstellung der zerebralen Durchblutung in Betracht?"],
            "noncommittal": 0
        }

        Do not say anything else. Make sure the response is a valid JSON.
    """

    user_prompt = f"""
        Background: {vignette.get_background()}
        
        Answer: {generated_answer}
    """

    generated_questions = []

    response = generate_response(system_prompt, user_prompt)
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
