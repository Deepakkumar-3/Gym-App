r"""
Fitness Q&A Agent — LangGraph, plain Python. RAG over fitness_kb.

Flow:
    retrieve_context --> generate_answer --> END

Uses Astra's NVIDIA vectorize integration for the search itself — we
just pass the question as plain text in the sort clause and Astra
handles embedding + similarity search server-side.
"""

import os
from typing import TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from db import fitness_kb_collection

GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4

ANSWER_SYSTEM_PROMPT = """You are a knowledgeable, friendly gym assistant \
answering general fitness questions. You will be given some reference \
passages and a question. Answer using ONLY the information in the \
passages — if the passages don't cover the question, say you don't have \
enough information on that specific topic rather than guessing. Keep the \
answer conversational and concise, not a bullet-point dump of the raw \
passages. Do not give personalized medical advice."""


class QAState(TypedDict):
    question: str
    context_chunks: list
    response: str


def retrieve_context_node(state: QAState) -> dict:
    results = fitness_kb_collection.find(
        {},
        sort={"$vectorize": state["question"]},
        limit=TOP_K,
        projection={"$vectorize": 1, "topic": 1},
    )
    chunks = []
    for doc in results:
        text = doc.get("$vectorize") or doc.get("text") or ""
        if text:
            chunks.append(text)
    return {"context_chunks": chunks}


def generate_answer_node(state: QAState) -> dict:
    chunks = state["context_chunks"]

    if not chunks:
        return {
            "response": (
                "I don't have any reference material on that topic yet — "
                "try rephrasing, or ask something else fitness-related."
            )
        }

    context_text = "\n\n".join(f"- {c}" for c in chunks)
    user_content = f"Reference passages:\n{context_text}\n\nQuestion: {state['question']}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return {"response": completion.choices[0].message.content}


def build_graph():
    graph = StateGraph(QAState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()


_compiled_graph = build_graph()


def run_fitness_qa_agent(question: str) -> str:
    """Entry point called from the FastAPI endpoint."""
    result = _compiled_graph.invoke({"question": question, "context_chunks": [], "response": ""})
    return result["response"]
