#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   multiple_prompts.py
@Time    :   2025/09/08 11:40:53
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   Create a story chaining several prompts.
"""

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap

from pathlib import Path

MODEL_PATH = Path(__file__).with_name("Phi-3-mini-4k-instruct-q4.gguf")


def main() -> None:
    llm = LlamaCpp(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        max_tokens=500,
        n_gpu_layers=-1,
        seed=42,
        verbose=False,
    )

    title_prompt = PromptTemplate(
        template="""<|user|>
    Create a title for a story about {summary}. Only return the title.<|end|>
    <|assistant|>""",
        input_variables=["summary"],
    )

    character_prompt = PromptTemplate(
        template="""<|user|>
    Describe the main character of a story about {summary} with the title {title}.
    Use only two senteces.<|end|>
    <|assistant|>""",
        input_variables=["summary", "title"],
    )
    story_prompt = PromptTemplate(
        template="""<|user|>
    Create a story about {summary} with the title {title}. The main chracter is: 
    {character}. Only return the story and it cannot be longer than one paragraph.
    <|end|>
    <|assistant|>""",
        input_variables=["summary", "title", "character"],
    )
    title_chain = title_prompt | llm | StrOutputParser()
    character_chain = character_prompt | llm | StrOutputParser()
    story_chain = story_prompt | llm | StrOutputParser()
    chain = (
        {"title": title_chain, "summary": RunnablePassthrough()}
        | RunnablePassthrough.assign(character=character_chain)
        | RunnablePassthrough.assign(story=story_chain)
    )
    print(chain.invoke("a girl that lost her mother."))


if __name__ == "__main__":
    main()
