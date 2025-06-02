from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi import FastAPI
import uvicorn

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator

import sys
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from settings.settings import settings

MAX_NEW_TOKENS = 250
config = ExLlamaV2Config(settings.llm_path)

config.arch_compat_overrides()
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, max_seq_len = 16384, lazy = True)
model.load_autosplit(cache, progress = False)
tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)
generator.warmup()

class PromptFormat_llama3:

    description = "Llama3-instruct models"

    def __init__(self):
        pass

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             tokenizer.single_id("<|eot_id|>"),
             tokenizer.single_id("<|start_header_id|>")]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True
    

prompt_format = PromptFormat_llama3()
add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
stop_conditions = prompt_format.stop_conditions(tokenizer)

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    max_new_tokens = request_dict.pop("max_new_tokens")
    response = generator.generate(prompt = prompt, max_new_tokens = max_new_tokens, add_bos = add_bos, stop_conditions=stop_conditions, completion_only=True, seed=15)
    return JSONResponse(response)

if __name__ == "__main__":
    port_number = 8080
    uvicorn.run(app, host="0.0.0.0", port=port_number)
