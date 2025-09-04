#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   tokenizer.py
@Time    :   2025/09/03 10:16:48
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda", dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = (
        "Write an email apologizing to Sarah for the tragic gardening mishap."
        "Explain how it happened. <|assistant|>"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids=input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))

    # What are the IDs?
    for id in input_ids[0]:
        print(f"ID: {id}, Token: {tokenizer.decode(id)}")
