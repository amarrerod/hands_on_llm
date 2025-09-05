#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   topic_modeling.py
@Time    :   2025/09/05 10:44:05
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from bertopic import BERTopic
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from pathlib import Path
from reduce_df import reduce_df
import re
import pandas as pd

DF_ORIGINAL = Path(__file__).with_name("original_topic_label.csv")


def main() -> None:
    df, embds = reduce_df()
    embds_red = df[["x0", "x1"]]
    dataset = load_dataset("maartengr/arxiv_nlp")["train"]
    abstracts, titles = dataset["Abstracts"], dataset["Titles"]
    topic_model = BERTopic(
        embedding_model=SentenceTransformer("thenlper/gte-small"),
        umap_model=UMAP(n_components=2, metric="cosine"),
        hdbscan_model=HDBSCAN(
            min_cluster_size=50, metric="euclidean", cluster_selection_method="eom"
        ),
        verbose=False,
    ).fit(abstracts, embds)
    data = {k: re.sub(r"\d+_", "", v) for k, v in topic_model.topic_labels_.items()}
    df_topics_original = pd.DataFrame.from_dict(
        data, orient="index", columns=["Original"]
    )
    df_topics_original.to_csv(DF_ORIGINAL, index=False)

    print(df_topics_original.head())
    print(embds_red)
    fig = topic_model.visualize_documents(
        titles,
        reduced_embeddings=embds_red,
        width=1200,
        hide_annotations=True,
    )
    fig.update_layout(font=dict(size=16))
    fig.show()
    fig = topic_model.visualize_heatmap(n_clusters=30)
    fig.show()
    fig = topic_model.visualize_hierarchy()
    fig.show()


if __name__ == "__main__":
    main()
