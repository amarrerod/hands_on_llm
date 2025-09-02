#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   first_text.py
@Time    :   2025/09/02 11:06:18
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


if __name__ == "__main__":
    model_name = "BSC-LT/salamandra-2b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=25,
        do_sample=False,
        repetition_penalty=1.2,
    )

    prompts = [
        "Crea un chiste sobre gallinas.",
        "Todo el mundo sabe que vivir en Barcelona es",
        "¿Pueblo o ciudad? Una ventaja de vivir en la ciudad es que hay muchas oportunidades de ocio y empleo, así como una gran diversidad de comercios para todos los gustos. Sin embargo, las ciudades suelen ser ",
        "Llegir ens proporciona",
        "What I find more fascinating about languages is that",
        "La vie peut être",
        "The future of AI is",
    ]
    set_seed(42)
    outputs = generator(prompts)
    for i, out in enumerate(outputs):
        print(f"Prompt: {prompts[i]} --> Answer: {out[0]['generated_text']}")
