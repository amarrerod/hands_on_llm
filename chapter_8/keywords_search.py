#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   keywords_search.py
@Time    :   2025/09/10 10:22:51
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

import numpy as np
from pathlib import Path
from typing import Iterable
import string
from sklearn.feature_extraction import _stop_words
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any


def tokenizer(text: str) -> Iterable:
    return filter(
        lambda token: len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS,
        [token.strip(string.punctuation) for token in text.lower().split()],
    )


def keyword_search(
    bm_25, query: str, top_k: int = 3, num_candidates: int = 15
) -> List[Dict[str, Any]]:
    scores = bm_25.get_scores(list(tokenizer(query)))
    top_n = np.argpartition(scores, -num_candidates)[-num_candidates:]
    hits = [{"corpus_id": idx, "score": scores[idx]} for idx in top_n]
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    return hits[:top_k]


def main() -> None:
    text = Path(__file__).with_name("interstellar.txt").read_text()
    sentences = [s.strip(" \n") for s in text.split(".")]

    tokenized_corpus = [list(tokenizer(passage)) for passage in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    hits = keyword_search(bm25, query="how precise was the science")
    print(hits)
    for h in hits:
        print(f"\t{h['score']:.3f}\t{sentences[h['corpus_id']]}".replace("\n", " "))


if __name__ == "__main__":
    main()
