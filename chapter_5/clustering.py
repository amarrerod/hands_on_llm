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

import matplotlib.pyplot as plt
import seaborn as sns
from .reduce_df import reduce_df

if __name__ == "__main__":
    df, _ = reduce_df()
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
