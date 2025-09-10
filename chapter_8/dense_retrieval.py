#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   dense_retrieval.py
@Time    :   2025/09/09 11:53:23
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import os
import faiss
import cohere
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from typing import Sequence, Tuple, Self
from dotenv import load_dotenv


class Embedding:
    def __init__(self):
        load_dotenv(Path(__file__).with_name(".env"))
        self._client = cohere.ClientV2(api_key=os.getenv("COHERE_KEY"))

    def __call__(self, text: Sequence[str]) -> npt.NDArray:
        return np.array(
            self._client.embed(
                texts=text,
                input_type="search_document",
                model="embed-v4.0",
                embedding_types=["float"],
            ).embeddings.float_
        )


class Index:
    def __init__(self, shape: int, embedding_model: Embedding):
        self._index = faiss.IndexFlatL2(shape)
        self._embedding_model = embedding_model

    def __iadd__(self, embeddings: npt.NDArray) -> Self:
        self._index.add(np.float32(embeddings))
        return self

    def __call__(
        self, query: Sequence[str], n_results: int, return_distances: bool = False
    ) -> Tuple:
        query_embed = self._embedding_model(query)[0]
        dists, sim_items = self._index.search(np.float32([query_embed]), n_results)
        return (dists, sim_items) if return_distances else sim_items


def main() -> None:
    text = Path(__file__).with_name("interstellar.txt").read_text()
    sentences = [s.strip(" \n") for s in text.split(".")]
    print(sentences)

    # 1. Embeds the text
    emb_model = Embedding()
    embeds = emb_model(sentences)
    print(embeds.shape)

    # 2. Create a search index
    index = Index(embeds.shape[1], embedding_model=emb_model)
    index += embeds

    # 3. Query the index
    dists, sim_items = index(
        query=["how precise was the science"], n_results=3, return_distances=True
    )

    df = pd.DataFrame(
        data={"sentences": np.array(sentences)[sim_items[0]], "distance": dists[0]}
    )
    print(df.head())


if __name__ == "__main__":
    main()
