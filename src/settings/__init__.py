import pandas as pd
import os
import json

from settings.settings import settings
from domain.vignette import VignetteCollection


def setup_model():
    if settings.inference_type == "exllama":
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator

        config = ExLlamaV2Config(settings.llm_path)

        config.arch_compat_overrides()
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len=16384, lazy=True)
        model.load_autosplit(cache, progress=False)
        tokenizer = ExLlamaV2Tokenizer(config)

        generator = ExLlamaV2DynamicGenerator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
        )
        generator.warmup()

        return generator
    elif settings.inference_type == "ollama":
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", temperature=0, format="json")
        return llm | JsonOutputParser()
    else:
        raise ValueError("Invalid inference type")


def get_abbreviation_dict():
    abbreviations = pd.read_csv(settings.abbreviations_csv_path)
    return dict(zip(abbreviations["Abbreviation"], abbreviations["Meaning"]))


def get_page_types():
    # TODO: Define enum for page types and two files (handbuch and antibiotika)
    with open(
        os.path.join(settings.data_path, settings.page_types_json_path),
        "r",
    ) as file:
        page_types = json.load(file)
    all_pages = list(range(7, 109))

    text_pages = []
    for p in all_pages:
        if p not in page_types["flowchart"] and p not in page_types["table"] and p not in page_types["visual"]:
            text_pages.append(p)

    return text_pages, page_types["flowchart"], page_types["table"], page_types["visual"]


if settings.inference_location == "local":
    LLM = setup_model()
elif settings.inference_location == "remote":
    LLM = None  # TODO


ABBREVIATION_DICT = get_abbreviation_dict()

VIGNETTE_COLLECTION = VignetteCollection()
VIGNETTE_COLLECTION.load_from_yaml(settings.vignettes_path)
VIGNETTE_COLLECTION.label_text_only_questions(get_page_types()[0])
