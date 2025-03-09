from pydantic import BaseModel, ValidationError
import json
from core.model import generate_response

SYSTEM_PROMPT = "You are an helpful AI assistant that assists with formats. Please provide a JSON instance that satisfies the constraints laid out in the Instructions. Do not say anthing else and make sure to return a valid JSON at all times."

FIX_OUTPUT_PROMPT = (
    "Instructions:\n--------------\n{instructions}\n--------------\n"
    "Completion:\n--------------\n{completion}\n--------------\n"
    "\nAbove, the Completion did not satisfy the constraints given in the Instructions.\n"
    "Error:\n--------------\n{error}\n--------------\n"
    "\nPlease try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:"
)


def try_parse_json(text: str) -> dict | None:
    """Extract JSON from text and parse it safely."""

    try:
        parsed_response = json.loads(text)
        if isinstance(parsed_response, str):
            return json.loads(parsed_response.strip())
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"Failed to parse response as JSON in try_parse_json function: {e}")
        return text.strip()


def prepare_json_response(text: str) -> str:
    text = text.strip()
    text = text.strip("` ")
    return text.strip()


def try_parse_result(response: str, model: type[BaseModel]) -> tuple[BaseModel | None, str | None]:
    response = prepare_json_response(response)
    json_object = try_parse_json(response)

    if json_object is None:
        print(f"Failed to extract JSON from response: {response}")
        return None, "Failed to extract JSON from response."

    try:
        return model.model_validate(json_object), None
    except ValidationError as e:
        return None, repr(e)


def get_format_instructions(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    schema.pop("type", None)
    schema_str = json.dumps(schema, indent=2)
    pretext = "The output should be formatted as a JSON instance that conforms to the JSON schema below."
    return f"{pretext}\n\n```\n{schema_str}\n```"


def parse_with_retry(model: type[BaseModel], response: str, max_retries: int = 4) -> BaseModel | None:
    for retries in range(max_retries):
        parsed, error_message = try_parse_result(response, model)
        if parsed:
            return parsed

        print(f"Failed to parse response, will try again. The error: {error_message}")
        user_prompt = FIX_OUTPUT_PROMPT.format(
            instructions=get_format_instructions(model), completion=response, error=error_message
        )
        response = generate_response(SYSTEM_PROMPT, user_prompt, max_new_tokens=2048)

    parsed, _ = try_parse_result(response, model)
    return parsed
