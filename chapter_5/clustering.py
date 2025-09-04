#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   clustering.py
@Time    :   2025/09/04 13:30:47
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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DF_FILENAME = Path(__file__).with_name("arxiv_clustered.csv")


if __name__ == "__main__":
    if not DF_FILENAME.is_file():
        dataset = load_dataset("maartengr/arxiv_nlp")["train"]
        abstracts, titles = dataset["Abstracts"], dataset["Titles"]
        embd_model = SentenceTransformer(
            "thenlper/gte-small"
        )  # Qwen/Qwen3-Embedding-4B Works better
        embds = embd_model.encode(abstracts, show_progress_bar=True)

        print(embds.shape)
        umap_model = UMAP(n_components=2, metric="cosine", random_state=42)
        reduced_embds = umap_model.fit_transform(embds)

        hdbscan_model = HDBSCAN(
            min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
        ).fit(reduced_embds)

        clusters = hdbscan_model.labels_
        print(f"{len(set(clusters))} clusters created.")
        df = pd.DataFrame(reduced_embds, columns=["x0", "x1"])
        df["title"] = titles
        df["cluster"] = [str(c) for c in clusters]
        df.to_csv(DF_FILENAME, index=False)
    else:
        df = pd.read_csv(DF_FILENAME)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(df[df.cluster == -1], x="x0", y="x1", alpha=0.05, s=2, c="grey")
    sns.scatterplot(
        df[df.cluster != -1],
        x="x0",
        y="x1",
        hue="cluster",
        alpha=0.6,
        s=2,
        palette="tab20b",
        legend=False,
    )
    plt.show()
