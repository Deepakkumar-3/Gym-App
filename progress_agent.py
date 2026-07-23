r"""
Progress Tracking Agent — LangGraph, plain Python.

Flow:
    extract_workout --workout mentioned--> store_workout --> fetch_context
        \--nothing logged--> ask_for_details --> END

    fetch_context --> generate_assessment_and_goal --> store_goal --> compose_response --> END

What it does:
  1. Reads the user's free-text workout recap, extracts structured data
     (exercises, duration, notes).
  2. Stores it in workout_logs.
  3. Pulls the user's goal (from their profile) and any ACTIVE injuries
     (from injury_notes) as context.
  4. Asks Groq for a short assessment of today's session + a concrete
     goal for next time, taking injuries into account (won't suggest
     aggravating an active injury).
  5. Stores that goal in daily_goals and returns a friendly summary.
"""

import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from db import daily_goals_collection, injury_notes_collection, users_collection, workout_logs_collection

GROQ_MODEL = "openai/gpt-oss-20b"

EXTRACTION_SYSTEM_PROMPT = """You read a message from someone describing \
their workout (or possibly not describing a workout at all). Respond \
with ONLY a JSON object, no other text, in this exact shape:

{
  "has_workout": true or false,
  "exercises": [{"name": "...", "sets": <int or null>, "reps": <int or null>, "weight_kg": <number or null>}],
  "duration_min": <int or null>,
  "notes": "<short one-sentence summary in your own words, or null>"
}

Set has_workout to false if the message doesn't describe any exercise \
performed (e.g. it's a greeting or an unrelated question). Only include \
numbers the user actually stated — never invent sets/reps/weights that \
weren't mentioned; use null for anything not stated."""

ASSESSMENT_SYSTEM_PROMPT = """You are an encouraging, safety-conscious gym \
coach. You'll be given: the user's stated goal, today's logged workout, \
and any of their currently active injuries. Respond with ONLY a JSON \
object, no other text, in this exact shape:

{
  "assessment": "<1-2 encouraging sentences about today's session>",
  "next_goal": "<one concrete, specific goal for their NEXT session>"
}

If there is an active injury, the next_goal MUST avoid aggravating it \
(e.g. suggest modified movements or lower intensity for that area) and \
should say so briefly. Do not invent specific medical guidance — just \
reasonable training-load caution. Keep both fields short."""


# ---------------------------------------------------------------- state ---
class ProgressState(TypedDict):
    user_id: str
    message: str
    workout_data: Optional[dict]
    user_profile: Optional[dict]
    active_injuries: list
    assessment: str
    next_goal: str
    response: str


# ---------------------------------------------------------------- nodes ---
def extract_workout_node(state: ProgressState) -> dict:
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
        data = {"has_workout": False, "exercises": [], "duration_min": None, "notes": None}
    return {"workout_data": data}


def route_after_extract(state: ProgressState) -> str:
    data = state.get("workout_data") or {}
    return "store_workout" if data.get("has_workout") else "no_workout"


def no_workout_node(state: ProgressState) -> dict:
    return {
        "response": (
            "I didn't catch a workout in that — tell me what you did "
            '(e.g. "3 sets of squats at 60kg, 8 reps each, plus 20 min cardio") '
            "and I'll log it and set your next goal."
        )
    }


def store_workout_node(state: ProgressState) -> dict:
    data = state["workout_data"]
    doc = {
        "user_id": state["user_id"],
        "date": date.today().isoformat(),
        "exercises": data.get("exercises") or [],
        "duration_min": data.get("duration_min"),
        "notes": data.get("notes") or state["message"],
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    workout_logs_collection.insert_one(doc)
    return {}


def fetch_context_node(state: ProgressState) -> dict:
    user = users_collection.find_one({"user_id": state["user_id"]})
    active_injuries = list(
        injury_notes_collection.find({"user_id": state["user_id"], "status": "active"})
    )
    return {"user_profile": user, "active_injuries": active_injuries}


def generate_assessment_and_goal_node(state: ProgressState) -> dict:
    workout = state["workout_data"]
    user = state["user_profile"] or {}
    injuries = state["active_injuries"]

    context_lines = [
        f"User's goal: {user.get('goal', 'unknown').replace('_', ' ')}",
        f"Today's workout: {json.dumps(workout)}",
    ]
    if injuries:
        injury_summary = "; ".join(
            f"{i.get('body_part', 'unspecified')} ({i.get('severity', 'unknown')})" for i in injuries
        )
        context_lines.append(f"Active injuries to work around: {injury_summary}")
    else:
        context_lines.append("No active injuries reported.")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ASSESSMENT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(context_lines)},
        ],
    )
    raw = completion.choices[0].message.content
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {
            "assessment": "Nice work getting today's session in.",
            "next_goal": "Repeat a similar session next time, adjusting based on how you feel.",
        }

    return {"assessment": data.get("assessment", ""), "next_goal": data.get("next_goal", "")}


def store_goal_node(state: ProgressState) -> dict:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    doc = {
        "user_id": state["user_id"],
        "date": tomorrow,
        "goal_text": state["next_goal"],
        "based_on": f"workout_log_{date.today().isoformat()}",
        "completed": False,
    }
    daily_goals_collection.insert_one(doc)
    return {}


def compose_response_node(state: ProgressState) -> dict:
    response = f"{state['assessment']} Your goal for next time: {state['next_goal']}"
    return {"response": response}


# --------------------------------------------------------------- graph ---
def build_graph():
    graph = StateGraph(ProgressState)

    graph.add_node("extract_workout", extract_workout_node)
    graph.add_node("no_workout", no_workout_node)
    graph.add_node("store_workout", store_workout_node)
    graph.add_node("fetch_context", fetch_context_node)
    graph.add_node("generate_assessment_and_goal", generate_assessment_and_goal_node)
    graph.add_node("store_goal", store_goal_node)
    graph.add_node("compose_response", compose_response_node)

    graph.add_edge(START, "extract_workout")
    graph.add_conditional_edges(
        "extract_workout",
        route_after_extract,
        {"store_workout": "store_workout", "no_workout": "no_workout"},
    )
    graph.add_edge("no_workout", END)
    graph.add_edge("store_workout", "fetch_context")
    graph.add_edge("fetch_context", "generate_assessment_and_goal")
    graph.add_edge("generate_assessment_and_goal", "store_goal")
    graph.add_edge("store_goal", "compose_response")
    graph.add_edge("compose_response", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_progress_agent(user_id: str, message: str) -> str:
    """Entry point called from the FastAPI endpoint."""
    result = _compiled_graph.invoke(
        {
            "user_id": user_id,
            "message": message,
            "workout_data": None,
            "user_profile": None,
            "active_injuries": [],
            "assessment": "",
            "next_goal": "",
            "response": "",
        }
    )
    return result["response"]
