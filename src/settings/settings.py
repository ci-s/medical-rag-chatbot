from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import yaml
import os

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field


class Settings(BaseSettings):
    abbreviations_csv_path: Path

    saved_document_path: Path
    data_path: Path
    prompt_path: Path
    file_name: str
    raw_file_name: str

    vignettes_path: Path

    page_types_json_path: Path
    config_path: Path

    results_path: Path
    headings_json_path: Path
    table_texts_path: Path


load_dotenv()
settings = Settings()


class Config(BaseModel):
    inference_type: str = Field(..., description="Type of inference: exllama or ollama")
    inference_location: str = Field(..., description="Location of inference: remote or local")
    llm_path: str | None = None
    filter_questions: list[str] | None = None
    filter_questions_based_on: str | None = None
    replace_abbreviations: bool = True
    inject_whitespace: bool = True
    chunk_method: str = Field(..., description="Method to chunk the text: size, section or semantic")
    top_k: int = 5
    experiment_name: str = "AAA"
    include_context: bool = True
    include_preceding_question_answers: bool = True
    max_new_tokens: int = 250
    optimization_method: (
        Literal[
            "hypothetical_document",
            "stepback",
            "decomposing",
            "paraphrasing",
        ]
        | None
    ) = None
    use_original_query_only: bool = True  # overides use_original_along_with_optimized
    use_original_along_with_optimized: bool = False
    most_relevant_chunk_first: bool = True
    summarize_retrieved_documents: bool = False
    match_chunk_similarity_threshold: int = 97
    chunk_size: int = 512
    surrounding_chunk_length: int = 0

    reasoning: bool = False
    thinking: bool = False

    saved_chunks_path_raw: str | None = None

    ragas: bool = False

    following_flowchart: bool = False
    flowchart_page: int | None = None

    def dump(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(self.model_dump(), file, default_flow_style=False, allow_unicode=True)

    @property
    def saved_chunks_path(self):
        if self.saved_chunks_path_raw:
            return os.path.join(settings.data_path, self.saved_chunks_path_raw)
        else:
            None


with open(settings.config_path, "r", encoding="utf-8") as file:
    raw_config = yaml.safe_load(file)

config = Config(**raw_config)
