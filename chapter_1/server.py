#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   server.py
@Time    :   2025/09/02 13:06:43
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

MODEL_NAME = "BSC-LT/salamandra-2b"


class Question(BaseModel):
    question: str
    max_tokens: int = 50


_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype="auto"
)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_generator = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    return_full_text=True,
    do_sample=True,
    repetition_penalty=1.2,
)
app = FastAPI()


@app.post("/answer/")
def answer(question: Question):
    print(f"Received the following question: {question}")
    output = _generator(question.question, max_new_tokens=question.max_tokens)
    print(f"Here's the output: {output}")
    return {"output": output[0]["generated_text"]}
