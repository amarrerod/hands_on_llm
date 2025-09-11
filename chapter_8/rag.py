#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   rag.py
@Time    :   2025/09/11 11:12:44
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from langchain_community.llms import LlamaCpp
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

MODEL_PATH = (
    Path(__file__).parent.parent / "chapter_7" / "Phi-3-mini-4k-instruct-q4.gguf"
)


def main() -> None:
    text = Path(__file__).with_name("interstellar.txt").read_text()
    sentences = [s.strip(" \n") for s in text.split(".")]

    llm = LlamaCpp(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        max_tokens=500,
        n_gpu_layers=-1,
        seed=42,
        verbose=False,
    )
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    database = FAISS.from_texts(sentences, embedding_model)
    prompt = PromptTemplate(
        template="""<|user|>
    Relevant information:
    {context}
    
    Present a concise answer to the following question using the relevant information
    provided above:
    {input}<|end|>
    <|assistant|>""",
        input_variables=["context", "question"],
    )
    docs_chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(database.as_retriever(), docs_chain)

    outcome = rag.invoke({"input": "Income generated"})
    print(outcome)


if __name__ == "__main__":
    main()
