from nomic import embed
from typing import Literal
import numpy as np


def embed_chunks(
    chunks: list[str] | str,
    task_type=Literal["search_document", "search_query"],
    model="nomic-embed-text-v1.5",
    inference_mode="local",
):
    if isinstance(chunks, str):
        chunks = [chunks]
    embed_res = embed.text(texts=chunks, model=model, task_type=task_type, inference_mode=inference_mode)
    return np.array(embed_res["embeddings"])
