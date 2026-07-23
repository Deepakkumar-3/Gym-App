r"""
Nutrition Agent — built with LangGraph, running as plain Python inside
your FastAPI app (main.py calls run_nutrition_agent()).

No visual builder, no wiring — just a small state graph:

    fetch_user --found--> compute_plan --> generate_response --> END
        \--not found--> not_found_response --> END
"""

import os
from typing import Optional, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from db import users_collection

GROQ_MODEL = "openai/gpt-oss-20b"

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extra_active": 1.9,
}

GOAL_MACRO_RULES = {
    "muscle_gain": {"protein_g_per_kg": 1.8, "fat_pct": 0.25},
    "weight_loss": {"protein_g_per_kg": 2.0, "fat_pct": 0.25},
    "stay_active": {"protein_g_per_kg": 1.4, "fat_pct": 0.30},
    "body_builder": {"protein_g_per_kg": 2.2, "fat_pct": 0.25},
}

GOAL_CALORIE_ADJUSTMENT = {
    "muscle_gain": 1.10,
    "weight_loss": 0.80,
    "stay_active": 1.00,
    "body_builder": 1.12,
}


def calculate_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if sex == "male" else base - 161


def calculate_nutrition_plan(user: dict) -> dict:
    bmr = calculate_bmr(user["sex"], user["weight_kg"], user["height_cm"], user["age"])
    tdee = bmr * ACTIVITY_MULTIPLIERS[user["activity_level"]]

    target_calories = tdee * GOAL_CALORIE_ADJUSTMENT[user["goal"]]
    macro_rule = GOAL_MACRO_RULES[user["goal"]]

    protein_g = macro_rule["protein_g_per_kg"] * user["weight_kg"]
    protein_kcal = protein_g * 4

    fat_kcal = target_calories * macro_rule["fat_pct"]
    fat_g = fat_kcal / 9

    carbs_kcal = max(target_calories - protein_kcal - fat_kcal, 0)
    carbs_g = carbs_kcal / 4

    return {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "target_calories": round(target_calories),
        "protein_g": round(protein_g),
        "fat_g": round(fat_g),
        "carbs_g": round(carbs_g),
    }


# ---------------------------------------------------------------- state ---
class NutritionState(TypedDict):
    user_id: str
    question: str
    user: Optional[dict]
    plan: Optional[dict]
    response: str


# ---------------------------------------------------------------- nodes ---
def fetch_user_node(state: NutritionState) -> dict:
    user = users_collection.find_one({"user_id": state["user_id"]})
    return {"user": user}


def route_after_fetch(state: NutritionState) -> str:
    return "compute_plan" if state.get("user") else "not_found"


def compute_plan_node(state: NutritionState) -> dict:
    plan = calculate_nutrition_plan(state["user"])
    return {"plan": plan}


def not_found_node(state: NutritionState) -> dict:
    return {
        "response": (
            f"I couldn't find a profile for user_id '{state['user_id']}'. "
            "Please make sure the profile exists and try again."
        )
    }


def generate_response_node(state: NutritionState) -> dict:
    user = state["user"]
    plan = state["plan"]
    question = state.get("question") or ""

    summary = (
        f"Nutrition data for {user.get('name', 'the user')} "
        f"(goal: {user['goal'].replace('_', ' ')}):\n"
        f"- BMR: {plan['bmr']} kcal/day\n"
        f"- TDEE (maintenance): {plan['tdee']} kcal/day\n"
        f"- Target intake: {plan['target_calories']} kcal/day\n"
        f"- Protein: {plan['protein_g']} g\n"
        f"- Fat: {plan['fat_g']} g\n"
        f"- Carbs: {plan['carbs_g']} g\n"
    )
    if question:
        summary += f"\nUser's question: {question}\n"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly, encouraging gym nutrition assistant. "
                    "Use only the numbers given to you — never invent numbers. "
                    "Explain the target calories and macro split in plain "
                    "language, and answer the user's question if they asked one."
                ),
            },
            {"role": "user", "content": summary},
        ],
    )
    return {"response": completion.choices[0].message.content}


# --------------------------------------------------------------- graph ---
def build_graph():
    graph = StateGraph(NutritionState)

    graph.add_node("fetch_user", fetch_user_node)
    graph.add_node("compute_plan", compute_plan_node)
    graph.add_node("not_found", not_found_node)
    graph.add_node("generate_response", generate_response_node)

    graph.add_edge(START, "fetch_user")
    graph.add_conditional_edges(
        "fetch_user",
        route_after_fetch,
        {"compute_plan": "compute_plan", "not_found": "not_found"},
    )
    graph.add_edge("compute_plan", "generate_response")
    graph.add_edge("generate_response", END)
    graph.add_edge("not_found", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_nutrition_agent(user_id: str, question: str = "") -> str:
    """Entry point called from the FastAPI endpoint."""
    result = _compiled_graph.invoke(
        {"user_id": user_id, "question": question, "user": None, "plan": None, "response": ""}
    )
    return result["response"]
