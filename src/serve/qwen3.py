from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi import FastAPI
import uvicorn
import base64
from io import BytesIO

from PIL import Image
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

import sys
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

# from settings.settings import settings

from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnCurlyBrace(StoppingCriteria):
    def __init__(self, tokenizer, stop_token="}"):
        self.stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)[-1]

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] == self.stop_token_id


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def format_prompt(prompt: str, image_input: str | None, system_prompt: str | None) -> list:
    if image_input:
        image_data = base64.b64decode(image_input)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        return [
            {"role": "system", "content": "You are a helpful assistant." if not system_prompt else system_prompt},
            {
                "role": "user",
                "content": prompt,
            }
        ]

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    system_prompt = request_dict.pop("system_prompt", None)
    image_input = request_dict.pop("image_input", None)
    max_new_tokens = request_dict.pop("max_new_tokens")
    
    
    formatted_messages = format_prompt(prompt, image_input, system_prompt)
    print(f"Formatted messages: {formatted_messages}")
    text = tokenizer.apply_chat_template(
        formatted_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if image_input:
        image_inputs, _ = process_vision_info(formatted_messages)
        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            #videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to("cuda")
    
    # try:
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        #stopping_criteria=stopping_criteria
    )
    # except TimeoutException:
    #     print("Generation took too long.")
    #     return JSONResponse({"error": "Timeout during generation"}, status_code=500)
    # finally:
    #     signal.alarm(0)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return JSONResponse(output_text)

if __name__ == "__main__":
    port_number = 8082
    uvicorn.run(app, host="0.0.0.0", port=port_number)
