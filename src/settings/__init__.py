import pandas as pd
import os
import json

from .settings import settings, config
from domain.vignette import VignetteCollection


def setup_model():
    if config.inference_type == "exllama":
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator

        config = ExLlamaV2Config(config.llm_path)

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
    elif config.inference_type == "ollama":
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
        if (
            p not in page_types["flowchart"]
            and p not in page_types["table"]
            and p not in page_types["visual"]
            and p not in page_types["exclude"]
        ):
            text_pages.append(p)

    return text_pages, page_types["flowchart"], page_types["table"], page_types["visual"]


if config.inference_location == "local":
    LLM = setup_model()
elif config.inference_location == "remote":
    LLM = None  # TODO


ABBREVIATION_DICT = get_abbreviation_dict()

VIGNETTE_COLLECTION = VignetteCollection()

if config.filter_questions:
    if config.filter_questions_based_on == "pages":
        if config.filter_questions == ["Text"]:
            pages, _, _, _ = get_page_types()
        elif config.filter_questions == ["Table"]:
            _, _, pages, _ = get_page_types()
        elif config.filter_questions == ["Flowchart"]:
            _, pages, _, _ = get_page_types()
        elif config.filter_questions == ["Text", "Table"]:  # problem, sorted?
            text_pages, _, table_pages, _ = get_page_types()
            pages = text_pages + table_pages
            print("Text and Table pages: ", pages)
        else:
            raise ValueError("Some multiple filter_questions value are not configured for page types yet")
        VIGNETTE_COLLECTION.load_from_yaml(
            settings.vignettes_path, filter_categories=config.filter_questions, filter_pages=pages
        )
    elif config.filter_questions_based_on == "categories":
        VIGNETTE_COLLECTION.load_from_yaml(
            settings.vignettes_path, filter_categories=config.filter_questions, filter_pages=None
        )
# VIGNETTE_COLLECTION.label_text_only_questions(get_page_types()[0])
