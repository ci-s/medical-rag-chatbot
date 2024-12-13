import requests
import json

from settings.settings import settings
from settings import LLM
from prompts.question import QUESTION_PROMPT
from domain.vignette import Vignette
from .utils import replace_abbreviations

MAX_NEW_TOKENS = 1024  # TODO: .env?


def create_question_prompt(retrieved_documents, vignette: Vignette, question_id: int):
    documents_str = "".join([f"{document}\n" for document in retrieved_documents])

    preceding_questions = vignette.get_preceding_questions(question_id)
    preceding_qa_str = ""
    for q in preceding_questions:
        preceding_qa_str += f"Question: {q.get_question()}\nAnswer: {q.get_answer()}\n\n"

    prompt = QUESTION_PROMPT.format(
        retrieved_documents=documents_str,
        background=vignette.background,
        context=vignette.context,
        preceding_question_answer_pairs=preceding_qa_str,
        query=vignette.get_question(question_id).question,
    )
    prompt, replaced_count = replace_abbreviations(prompt)

    print("Prompt:", prompt)
    print(f"Number of abbreviations replaced: {replaced_count}")
    return prompt


class PromptFormat_llama3:
    description = "Llama3-instruct models"

    def __init__(self):
        pass

    def default_system_prompt(self):
        return """Assist users with tasks and answer questions to the best of your knowledge. Provide helpful and informative responses.Make sure to return a valid JSON."""

    def first_prompt(self):
        return (
            """<|start_header_id|>system<|end_header_id|>\n\n"""
            + """<|system_prompt|><|eot_id|>"""
            + """<|start_header_id|>user<|end_header_id|>\n\n"""
            + """<|user_prompt|><|eot_id|>"""
            + """<|start_header_id|>assistant<|end_header_id|>"""
        )

    def subs_prompt(self):
        return (
            """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""
            + """<|user_prompt|><|eot_id|>"""
            + """<|start_header_id|>assistant<|end_header_id|>"""
        )

    def stop_conditions(self, tokenizer):
        return [tokenizer.eos_token_id, tokenizer.single_id("<|eot_id|>"), tokenizer.single_id("<|start_header_id|>")]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


def format_prompt(prompt_format: PromptFormat_llama3, user_prompt, first=True):
    if first:
        return (
            prompt_format.first_prompt()
            .replace("<|system_prompt|>", prompt_format.default_system_prompt())
            .replace("<|user_prompt|>", user_prompt)
        )
    else:
        return prompt_format.subs_prompt().replace("<|user_prompt|>", user_prompt)


prompt_format = PromptFormat_llama3()


def generate_response(prompt: str) -> str:
    if settings.inference_location == "local":
        if settings.inference_type == "exllama":
            return LLM.generate(prompt=prompt, max_new_tokens=MAX_NEW_TOKENS, add_bos=True)
        elif settings.inference_type == "ollama":
            # url = "http://localhost:11434/api/generate"
            # model_name = "llama3.1"  #:8b-instruct-q4_0
            # stream = False
            # data = {
            #     "model": model_name,
            #     "prompt": prompt,
            #     "stream": stream,
            #     "parameter": ["temperature", 0],
            #     # "options": {"seed": 42},
            #     # "format": "json",
            #     # "raw": True,
            # }  # "raw": True, "seed", 123
            # response = requests.post(url, json=data)

            # return response.json()["response"]
            return LLM.invoke(prompt)["output"]  # TODO: messages?
        else:
            raise ValueError("Invalid inference type")
    elif settings.inference_location == "remote":
        url = "http://localhost:8082/generate"
        headers = {"Content-Type": "application/json"}

        formatted_prompt = format_prompt(prompt_format, prompt, first=True)
        data = {"prompt": formatted_prompt, "max_new_tokens": MAX_NEW_TOKENS}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        try:
            parsed_response = response.json()  # Automatically parses JSON
            # Check if the parsed response contains another JSON-encoded string
            if isinstance(parsed_response, str):
                return json.loads(parsed_response)  # Parse inner JSON
            return parsed_response
        except json.JSONDecodeError:
            # If the response isn't valid JSON, return raw text
            return response.text
