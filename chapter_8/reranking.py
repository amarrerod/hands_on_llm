#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   reranking.py
@Time    :   2025/09/11 10:42:57
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
from typing import Sequence, Tuple, Self, List
from dotenv import load_dotenv


class CoClient:
    def __init__(self):
        load_dotenv(Path(__file__).with_name(".env"))
        self._client = cohere.ClientV2(api_key=os.getenv("COHERE_KEY"))

    def embed(self, text: Sequence[str]) -> npt.NDArray:
        return np.array(
            self._client.embed(
                texts=text,
                input_type="search_document",
                model="embed-v4.0",
                embedding_types=["float"],
            ).embeddings.float_
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int = 3,
        return_docs: bool = True,
    ) -> List[Tuple[float, str]]:
        reranked = self._client.rerank(
            model="rerank-v3.5", query=query, documents=documents, top_n=top_n
        ).results

        return (
            list((score, documents[idx]) for (_, idx), (_, score) in reranked)
            if return_docs
            else reranked
        )


class Index:
    def __init__(self, shape: int, embedding_model: CoClient):
        self._index = faiss.IndexFlatL2(shape)
        self._embedding_model = embedding_model

    def __iadd__(self, embeddings: npt.NDArray) -> Self:
        self._index.add(np.float32(embeddings))
        return self

    def __call__(
        self, query: Sequence[str], n_results: int, return_distances: bool = False
    ) -> Tuple:
        query_embed = self._embedding_model.embed(query)[0]
        dists, sim_items = self._index.search(np.float32([query_embed]), n_results)
        return (dists, sim_items) if return_distances else sim_items


def main() -> None:
    text = Path(__file__).with_name("interstellar.txt").read_text()
    sentences = [s.strip(" \n") for s in text.split(".")]
    print(sentences)

    # 1. Embeds the text
    co_client = CoClient()
    embeds = co_client.embed(sentences)
    print(embeds.shape)

    # 2. Create a search index
    index = Index(embeds.shape[1], embedding_model=co_client)
    index += embeds

    # 3. Query the index
    dists, sim_items = index(
        query=["how precise was the science"], n_results=3, return_distances=True
    )

    # 4. Rerank the documents
    reranked = co_client.rerank(
        "how precise was the science", documents=sentences, top_n=3, return_docs=True
    )
    for i, (score, doc) in enumerate(reranked):
        print(i, score, doc)

    df = pd.DataFrame(
        data={"sentences": np.array(sentences)[sim_items[0]], "distance": dists[0]}
    )
    print(df.head())


if __name__ == "__main__":
    main()
