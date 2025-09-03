#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   song_embedding.py
@Time    :   2025/09/03 11:02:41
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import pandas as pd
from urllib import request
from gensim.models import Word2Vec
import numpy as np


def get_n_most_similar(model, song_id: int, songs_df: pd.DataFrame, n: int = 20):
    title, artist = songs_df.iloc[song_id]
    print(f"{title} - {artist} most similar songs are:\n")
    top_most_sim = np.array(model.wv.most_similar(positive=str(song_id), topn=n))[:,0]
    return songs_df.iloc[top_most_sim]


if __name__ == "__main__":
    playlists_raw = (
        request.urlopen(
            "https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt"
        )
        .read()
        .decode("utf-8")
        .split("\n")[2:]
    )
    playlists = [s.rstrip().split() for s in playlists_raw if len(s.split()) > 1]
    songs_raw = (
        request.urlopen(
            "https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt"
        )
        .read()
        .decode("utf-8")
        .split("\n")
    )
    songs_raw = [s.rstrip().split("\t") for s in songs_raw]
    songs_df = pd.DataFrame(data=songs_raw, columns=["id", "title", "artist"])
    songs_df = songs_df.set_index("id")
    print(songs_df.head())
    ids, title, artist = songs_df.iloc[0]
    print(ids, title, artist)
    # Train a model
    model = Word2Vec(
        playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4
    )
    s_id = np.random.default_rng().integers(low=0, high=len(songs_df), size=1)
    most_similar_df = get_n_most_similar(model, song_id=s_id, songs_df=songs_df, n=5)
