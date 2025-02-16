import requests
import json

from settings.settings import config
from settings import LLM


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


def format_prompt(prompt_format: PromptFormat_llama3, user_prompt, system_prompt=None, first=True):
    if system_prompt is None:
        system_prompt = prompt_format.default_system_prompt()
    if first:
        return (
            prompt_format.first_prompt()
            .replace("<|system_prompt|>", system_prompt)
            .replace("<|user_prompt|>", user_prompt)
        )
    else:
        return prompt_format.subs_prompt().replace("<|user_prompt|>", user_prompt)


prompt_format = PromptFormat_llama3()


def generate_response(user_prompt: str, system_prompt: str = None, max_new_tokens: int = 1024) -> str:
    if config.inference_location == "local":
        if config.inference_type == "exllama":
            formatted_prompt = format_prompt(prompt_format, user_prompt, system_prompt, first=True)
            return LLM.generate(prompt=formatted_prompt, max_new_tokens=config.max_new_tokens, add_bos=True)
        elif config.inference_type == "ollama":
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
            return LLM.invoke(user_prompt)["output"]  # TODO: messages?
        else:
            raise ValueError("Invalid inference type")
    elif config.inference_location == "remote":
        url = "http://localhost:8082/generate"
        headers = {"Content-Type": "application/json"}

        formatted_prompt = format_prompt(prompt_format, user_prompt, system_prompt, first=True)
        data = {"prompt": formatted_prompt, "max_new_tokens": max_new_tokens}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_text = response.text.encode("utf-8").decode("utf-8").strip()
        return response_text
        # try:
        #     parsed_response = json.loads(response_text)  # Parse JSON
        #     if isinstance(parsed_response, str):
        #         return json.loads(parsed_response.strip())  # Handle double-encoded JSON
        #     return parsed_response
        # except json.JSONDecodeError:
        #     print("Failed to parse response as JSON")
        #     return response_text
