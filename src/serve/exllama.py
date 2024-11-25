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

# Figure out temperature setting
# settings = ExLlamaV2Sampler.Settings()
# settings.temperature = 0.85

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

    def default_system_prompt(self):
        return \
            """Assist users with tasks and answer questions to the best of your knowledge. Provide helpful and informative """ + \
            """responses. Be conversational and engaging. If you are unsure or lack knowledge on a topic, admit it and try """ + \
            """to find the answer or suggest where to find it. Keep responses concise and relevant. Follow ethical """ + \
            """guidelines and promote a safe and respectful interaction."""

    def first_prompt(self):
        return \
            """<|start_header_id|>system<|end_header_id|>\n\n""" + \
            """<|system_prompt|><|eot_id|>""" + \
            """<|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>"""

    def subs_prompt(self):
        return \
            """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             tokenizer.single_id("<|eot_id|>"),
             tokenizer.single_id("<|start_header_id|>")]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True
    
def format_prompt(prompt_format: PromptFormat_llama3, user_prompt, first=True):

    if first:
        return prompt_format.first_prompt() \
            .replace("<|system_prompt|>", prompt_format.default_system_prompt()) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return prompt_format.subs_prompt() \
            .replace("<|user_prompt|>", user_prompt)


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
    # Read contract
    prompt = request_dict.pop("prompt")
    max_new_tokens = request_dict.pop("max_new_tokens")
    response = generator.generate(prompt = prompt, max_new_tokens = max_new_tokens, add_bos = add_bos, stop_conditions=stop_conditions, completion_only=True)
    return JSONResponse(response)

if __name__ == "__main__":
    port_number = 8080
    uvicorn.run(app, host="0.0.0.0", port=port_number)
