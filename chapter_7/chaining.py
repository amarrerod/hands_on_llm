#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   chaining.py
@Time    :   2025/09/08 11:11:09
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from langchain import LlamaCpp
from langchain_core.prompts import PromptTemplate
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("Phi-3-mini-4k-instruct-q4.gguf")


def main() -> None:
    template: str = """<|user|>
    {input_prompt}<|end|>
    <|assistant|>"""
    prompt = PromptTemplate(template=template, input_variables=["input_prompt"])
    llm = LlamaCpp(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        max_tokens=500,
        n_gpu_layers=-1,
        seed=42,
        verbose=False,
    )
    chain = prompt | llm
    for i in range(10):
        output = chain.invoke({"input_prompt": "Hi! My name's Ale. What is 2 + 2?"})
        print(f"#{i} -> {output}")


if __name__ == "__main__":
    main()
