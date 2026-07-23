r"""
Injury / Discomfort Logging Agent — LangGraph, plain Python.

Flow:
    extract_injury --injury mentioned--> store_injury --> generate_response --> END
        \--no injury mentioned--> ask_for_details --> END

The extraction step asks Groq to read the user's free-text message and
pull out: whether an injury/discomfort was mentioned at all, which body
part, how severe, and a short description. That gets written straight
into the injury_notes collection.

This agent DOES NOT diagnose or give medical treatment advice — it logs
what the user reports and gives general, non-specific safety guidance
(e.g. suggesting rest or seeing a professional for anything severe or
persistent).
"""

import json
import os
from datetime import date, datetime, timezone
from typing import Optional, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from db import injury_notes_collection

GROQ_MODEL = "openai/gpt-oss-20b"

EXTRACTION_SYSTEM_PROMPT = """You read a message from someone at the gym and \
determine whether they are describing an injury, pain, or physical \
discomfort. Respond with ONLY a JSON object, no other text, in this \
exact shape:

{
  "has_injury": true or false,
  "body_part": "<short body part name, e.g. 'left knee', or null>",
  "severity": "<one of: mild, moderate, severe, or null>",
  "description": "<a short one-sentence summary in your own words, or null>"
}

Set has_injury to false if the message is a greeting, an unrelated \
question, or doesn't mention any pain/injury/discomfort. Infer severity \
conservatively from the language used (e.g. "sharp pain", "can't put \
weight on it" => severe; "a little sore" => mild)."""


# ---------------------------------------------------------------- state ---
class InjuryState(TypedDict):
    user_id: str
    message: str
    injury_data: Optional[dict]
    stored: bool
    response: str


# ---------------------------------------------------------------- nodes ---
def extract_injury_node(state: InjuryState) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": state["message"]},
        ],
    )

    raw = completion.choices[0].message.content
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {"has_injury": False, "body_part": None, "severity": None, "description": None}

    return {"injury_data": data}


def route_after_extract(state: InjuryState) -> str:
    data = state.get("injury_data") or {}
    return "store_injury" if data.get("has_injury") else "no_injury"


def store_injury_node(state: InjuryState) -> dict:
    data = state["injury_data"]
    doc = {
        "user_id": state["user_id"],
        "date": date.today().isoformat(),
        "body_part": data.get("body_part") or "unspecified",
        "description": data.get("description") or state["message"],
        "severity": data.get("severity") or "mild",
        "status": "active",
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    injury_notes_collection.insert_one(doc)
    return {"stored": True}


def generate_response_node(state: InjuryState) -> dict:
    data = state["injury_data"]
    severity = data.get("severity") or "mild"
    body_part = data.get("body_part") or "the area you mentioned"

    severe_note = (
        "Since this sounds severe, please consider seeing a doctor or "
        "physiotherapist rather than pushing through it. "
        if severity == "severe"
        else ""
    )

    response = (
        f"Got it — I've logged {severity} discomfort in your {body_part}. "
        f"{severe_note}"
        "In general: avoid exercises that directly aggravate this area, "
        "and if it doesn't improve in a few days or gets worse, it's "
        "worth getting checked out by a professional. I'm not able to "
        "diagnose or give medical advice, just keeping a record and "
        "flagging when caution makes sense. Let me know if it changes."
    )
    return {"response": response}


def no_injury_node(state: InjuryState) -> dict:
    return {
        "response": (
            "I didn't catch a specific injury or discomfort in that — "
            "could you tell me what hurts, where, and how it started? "
            "(e.g. \"sharp pain in my right shoulder since yesterday's bench press\")"
        )
    }


# --------------------------------------------------------------- graph ---
def build_graph():
    graph = StateGraph(InjuryState)

    graph.add_node("extract_injury", extract_injury_node)
    graph.add_node("store_injury", store_injury_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("no_injury", no_injury_node)

    graph.add_edge(START, "extract_injury")
    graph.add_conditional_edges(
        "extract_injury",
        route_after_extract,
        {"store_injury": "store_injury", "no_injury": "no_injury"},
    )
    graph.add_edge("store_injury", "generate_response")
    graph.add_edge("generate_response", END)
    graph.add_edge("no_injury", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_injury_agent(user_id: str, message: str) -> str:
    """Entry point called from the FastAPI endpoint."""
    result = _compiled_graph.invoke(
        {"user_id": user_id, "message": message, "injury_data": None, "stored": False, "response": ""}
    )
    return result["response"]
