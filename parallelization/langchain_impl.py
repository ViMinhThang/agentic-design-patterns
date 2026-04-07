import asyncio
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

llm = None
try:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
except Exception as e:
    print(f"Error initializing language model: {e}")

summarize_chain: Runnable | None = None
questions_chain: Runnable | None = None
terms_chain: Runnable | None = None
map_chain: Runnable | None = None
full_parallel_chain: Runnable | None = None

if llm is not None:
    summarize_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Summarize the following topic concisely:"),
                ("user", "{topic}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    questions_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Identify 5 - 10 key terms from the following topic, seperated by commas:",
                ),
                ("user", "{topic}"),
            ]
        )
        | llm
        | StrOutputParser()
    )
    terms_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Identify 5-10 key terms from the following topic,separated by commas:",
                ),
                ("user", "{topic}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "topic": RunnablePassthrough(),
        }
    )

    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Based on the following information:
            Summary: {summary}
            Related Questions :{questions}
            Key Terms : {key_terms}
            Synthesize a comprehensive answer.
            """,
            ),
            ("user", "Original topic: {topic}"),
        ]
    )

    full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()


async def run_parallel_example(topic: str) -> None:
    if not llm or not full_parallel_chain:
        print("LLM not initialized. Cannot run example.")
        return
    print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
    try:
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")


if __name__ == "__main__":
    test_topic = "The history of space exploration"
    asyncio.run(run_parallel_example(test_topic))
