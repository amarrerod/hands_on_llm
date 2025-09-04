#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sentiment.py
@Time    :   2025/09/04 11:25:44
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from datasets import load_dataset
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity


def eval(y_true, y_pred) -> str | dict:
    performance = classification_report(
        y_true, y_pred, target_names=["Negative Review", "Positive Review"]
    )
    print(performance)
    return performance


def predict_representation_model(df) -> str | dict:
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    generator = pipeline(
        model=model_path, tokenizer=model_path, return_all_scores=True, device="cuda"
    )
    predictions = np.zeros(len(df["validation"]))
    for i, output in enumerate(
        tqdm(
            generator(KeyDataset(df["validation"], "text")), total=len(df["validation"])
        )
    ):
        negative, positive = output[0]["score"], output[2]["score"]
        predictions[i] = np.argmax([negative, positive])
    return eval(df["validation"]["label"], predictions)


def predict_embedding_model(df) -> str | dict:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    train_embds = model.encode(df["train"]["text"], show_progress_bar=True)
    val_embds = model.encode(df["validation"]["text"], show_progress_bar=True)
    print(f"Train embeddings shape: {train_embds.shape}")
    clf = RandomForestClassifier(random_state=42)
    clf.fit(train_embds, df["train"]["label"])
    predictions = clf.predict(val_embds)
    return eval(df["validation"]["label"], predictions)


def predict_unlabeled_data(df) -> str | dict:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    label_embds = model.encode(["A negative review", "A postivie review"])
    val_embds = model.encode(df["validation"]["text"])
    sim_matrix = cosine_similarity(val_embds, label_embds)
    predictions = np.argmax(sim_matrix, axis=1)
    return eval(df["validation"]["label"], predictions)


def predict_generative_model(df) -> str | dict:
    generator = pipeline(
        "text2text-generation", model="google/flan-t5-base", device="cuda"
    )
    prompt = "Is the following sentence positive or negative?"
    _df = df.map(lambda x: {"t5": prompt + x["text"]})
    predictions = np.zeros(len(df["validation"]))
    for i, output in enumerate(
        tqdm(
            generator(KeyDataset(_df["validation"], "t5")), total=len(df["validation"])
        )
    ):
        txt = output[0]["generated_text"]
        predictions[i] = 0 if txt == "negative" else 1
    return eval(df["validation"]["label"], predictions)


if __name__ == "__main__":
    df = load_dataset("rotten_tomatoes")
    print(df)
    predictions = predict_representation_model(df)
    performance = predict_embedding_model(df)
    performance = predict_unlabeled_data(df)
    performance = predict_generative_model(df)
