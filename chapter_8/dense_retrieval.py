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

import cohere
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
import faiss


def search(
    query: str,
    n_results: int,
    co: cohere.ClientV2,
    index: faiss.IndexFlatL2,
) -> Tuple:
    query_embed = co.embed(
        texts=[query],
        input_type="search_query",
        model="embed-v4.0",
        embedding_types=["float"],
    ).embeddings.float_[0]
    dists, sim_items = index.search(np.float32([query_embed]), n_results)
    print(f"Query: {query}\nNN: {sim_items}")
    return dists, sim_items


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    co = cohere.ClientV2(api_key=os.getenv("COHERE_KEY"))
    text = Path(__file__).with_name("interstellar.txt").read_text()

    sentences = [s.strip(" \n") for s in text.split(".")]
    print(sentences)
    # 1. Embed the text chunks
    embeds = np.array(
        co.embed(
            texts=sentences,
            input_type="search_document",
            model="embed-v4.0",
            embedding_types=["float"],
        ).embeddings.float_
    )
    print(embeds.shape)

    # 2. Create a search index
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(np.float32(embeds))

    dists, sim_items = search(
        query="how precise was the science", n_results=3, co=co, index=index
    )
    df = pd.DataFrame(
        data={"sentences": np.array(sentences)[sim_items[0]], "distance": dists[0]}
    )
    print(df.head())


if __name__ == "__main__":
    main()
