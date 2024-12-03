from pathlib import Path
from dotenv import load_dotenv

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    abbreviations_replaced: bool
    abbreviations_csv_path: Path

    saved_document_path: Path
    data_path: Path
    prompt_path: Path
    file_name: str

    vignettes_path: Path

    llm_path: Path
    inference_type: str
    inference_location: str

    page_types_json_path: Path


load_dotenv()
settings = Settings()
