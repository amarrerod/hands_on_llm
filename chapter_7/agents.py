#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   agents.py
@Time    :   2025/09/08 13:48:52
@Author  :   Alejandro Marrero
@Version :   1.0
@Contact :   amarrerd@ull.edu.es
@License :   (C)Copyright 2025, Alejandro Marrero
@Desc    :   None
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from pathlib import Path

# Definitely use a better model
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
    prompt = PromptTemplate(
        template="""Answer the following questions as best as you can. You have
                            access to the following tools:
                            
                            {tools}
                            
                            Use the following format:
                            
                            Question: the input question you must answer
                            Thought: you should always think about what to do
                            Action: the action to take, should be one of [{tool_names}]
                            Action Input: the input to the action
                            Observation: the result of the action
                            ... (this Thought/Action/Action Input/Observation can be repeated N times)
                            Thought: I now know the final answer
                            Final answer: the final answer to the original input question
                            
                            Begin!
                            
                            Question: {input}
                            Thought: {agent_scratchpad}""",
        input_variables=["tools", "tools_names", "input", "agent_scratchpad"],
    )
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="duckduckgo",
        description="A web search engine. Use this as a search engine for general queries.",
        func=search.run,
    )
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(search_tool)

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    print(
        executor.invoke(
            {
                "input": "What is the current price of a Macbook Pro in USD? How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD."
            }
        )
    )


if __name__ == "__main__":
    main()
