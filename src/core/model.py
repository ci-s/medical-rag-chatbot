import requests
from settings.settings import settings
from settings import LLM

MAX_NEW_TOKENS = 250 # TODO: .env?

def generate_response(prompt: str) -> str:
    if settings.inference_type == "exllama":
        return LLM.generate(prompt = prompt, max_new_tokens = MAX_NEW_TOKENS, add_bos = True)
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
        return LLM.invoke(prompt)["output"] # TODO: messages?
    else:
        raise ValueError("Invalid inference type")
