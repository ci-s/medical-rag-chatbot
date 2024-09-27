import requests

url = "http://localhost:11434/api/generate"
model_name = "llama3.1"
stream = False


def generate_response(query: str) -> str:
    data = {"model": model_name, "prompt": query, "stream": stream, "parameter": ["temperature", 0]}
    response = requests.post(url, json=data)
    return response.json()["response"]
