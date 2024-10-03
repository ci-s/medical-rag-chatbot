import requests

url = "http://localhost:11434/api/generate"
model_name = "llama3.1"  #:8b-instruct-q4_0
stream = False


def generate_response(query: str) -> str:
    data = {
        "model": model_name,
        "prompt": query,
        "stream": stream,
        "parameter": ["temperature", 0],
    }  # "raw": True, "seed", 123
    response = requests.post(url, json=data)

    # print("Prompt token count: ", response.json()["prompt_eval_count"])
    # print("Response token count: ", response.json()["eval_count"])

    return response.json()["response"]
