#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   reduce_df.py
@Time    :   2025/09/05 11:10:13
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from pathlib import Path
import torch

_DF_FILENAME = Path(__file__).with_name("arxiv_clustered.csv")
_EMBDS = Path(__file__).with_name("embds.pt")


def reduce_df(
    dataset_fn: str = "maartengr/arxiv_nlp",
) -> tuple[pd.DataFrame, torch.Tensor]:
    if _DF_FILENAME.is_file() and _EMBDS.is_file():
        embds = torch.load(_EMBDS, weights_only=False)
        return pd.read_csv(_DF_FILENAME), embds
    dataset = load_dataset(dataset_fn)["train"]
    abstracts, titles = dataset["Abstracts"], dataset["Titles"]

    embd_model = SentenceTransformer(
        "thenlper/gte-small"
    )  # Qwen/Qwen3-Embedding-4B Works better
    embds = embd_model.encode(abstracts, show_progress_bar=True)
    torch.save(embds, _EMBDS)

    umap_model = UMAP(n_components=2, metric="cosine", random_state=42)
    reduced_embds = umap_model.fit_transform(embds)
    hdbscan_model = HDBSCAN(
        min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
    ).fit(reduced_embds)
    clusters = hdbscan_model.labels_

    df = pd.DataFrame(reduced_embds, columns=["x0", "x1"])
    df["title"] = titles
    df["cluster"] = [str(c) for c in clusters]
    df.to_csv(_DF_FILENAME, index=False)

    return df, embds
