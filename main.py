"""
Gym App — Auth + Profile API

Endpoints:
  POST /api/signup   -> create account, returns access token + sets refresh cookie
  POST /api/login    -> verify credentials, returns access token + sets refresh cookie
  POST /api/refresh  -> exchange refresh cookie for a new access token
  POST /api/logout   -> clear refresh cookie
  GET  /api/me       -> current user's profile (requires access token)
  PUT  /api/me       -> update current user's profile

Run with:
  uvicorn main:app --reload
"""

import uuid
from datetime import datetime, timezone

import jwt
from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from db import (
    daily_goals_collection,
    injury_notes_collection,
    users_collection,
    workout_logs_collection,
)
from fitness_qa_agent import run_fitness_qa_agent
from injury_agent import run_injury_agent
from nutrition_agent import run_nutrition_agent
from progress_agent import run_progress_agent
from schemas import (
    FitnessQuestion,
    InjuryReport,
    LoginRequest,
    NutritionQuestion,
    ProfileResponse,
    ProfileUpdate,
    SignupRequest,
    TokenResponse,
    WorkoutLog,
)

app = FastAPI(title="Gym App API")

# OAuth2 scheme for Swagger docs
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token", auto_error=False)

REFRESH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        samesite="lax",
        secure=True,  # important for HTTPS in production
        max_age=REFRESH_COOKIE_MAX_AGE,
        path="/api",
    )


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user = users_collection.find_one({"user_id": payload["sub"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# --------------------------- Auth ---------------------------
@app.post("/api/signup", response_model=TokenResponse)
def signup(data: SignupRequest, response: Response):
    if users_collection.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="An account with this email already exists")

    user_id = "u_" + uuid.uuid4().hex[:10]
    doc = {
        "user_id": user_id,
        "name": data.name,
        "email": data.email,
        "password_hash": hash_password(data.password),
        "age": data.age,
        "sex": data.sex,
        "weight_kg": data.weight_kg,
        "height_cm": data.height_cm,
        "activity_level": data.activity_level,
        "goal": data.goal,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    users_collection.insert_one(doc)

    access_token = create_access_token(user_id)
    _set_refresh_cookie(response, create_refresh_token(user_id))
    return TokenResponse(access_token=access_token)


@app.post("/api/login", response_model=TokenResponse)
def login(data: LoginRequest, response: Response):
    user = users_collection.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(user["user_id"])
    _set_refresh_cookie(response, create_refresh_token(user["user_id"]))
    return TokenResponse(access_token=access_token)


@app.post("/api/token", response_model=TokenResponse, include_in_schema=False)
def token_login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(user["user_id"])
    _set_refresh_cookie(response, create_refresh_token(user["user_id"]))
    return TokenResponse(access_token=access_token)


@app.post("/api/refresh", response_model=TokenResponse)
def refresh(refresh_token: str | None = Cookie(default=None)):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    try:
        payload = decode_token(refresh_token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    return TokenResponse(access_token=create_access_token(payload["sub"]))


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("refresh_token", path="/api")
    return {"message": "Logged out"}


# --------------------------- Profile ---------------------------
@app.get("/api/me", response_model=ProfileResponse)
def get_me(user: dict = Depends(get_current_user)):
    return ProfileResponse(**user)


@app.put("/api/me", response_model=ProfileResponse)
def update_me(data: ProfileUpdate, user: dict = Depends(get_current_user)):
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    if updates:
        users_collection.update_one({"user_id": user["user_id"]}, {"$set": updates})
        user.update(updates)
    return ProfileResponse(**user)


# --------------------------- Agents ---------------------------
@app.post("/api/nutrition/ask")
def nutrition_ask(data: NutritionQuestion, user: dict = Depends(get_current_user)):
    response_text = run_nutrition_agent(user["user_id"], data.question)
    return {"response": response_text}


@app.post("/api/injuries/report")
def report_injury(data: InjuryReport, user: dict = Depends(get_current_user)):
    response_text = run_injury_agent(user["user_id"], data.message)
    return {"response": response_text}


@app.get("/api/injuries")
def list_injuries(user: dict = Depends(get_current_user)):
    docs = list(injury_notes_collection.find({"user_id": user["user_id"]}))
    docs.sort(key=lambda d: d.get("logged_at", ""), reverse=True)
    return {"injuries": docs}


@app.post("/api/progress/log")
def log_progress(data: WorkoutLog, user: dict = Depends(get_current_user)):
    response_text = run_progress_agent(user["user_id"], data.message)
    return {"response": response_text}


@app.get("/api/progress/goals")
def list_goals(user: dict = Depends(get_current_user)):
    docs = list(daily_goals_collection.find({"user_id": user["user_id"]}))
    docs.sort(key=lambda d: d.get("date", ""), reverse=True)
    return {"goals": docs}


@app.get("/api/progress/workouts")
def list_workouts(user: dict = Depends(get_current_user)):
    docs = list(workout_logs_collection.find({"user_id": user["user_id"]}))
    docs.sort(key=lambda d: d.get("logged_at", ""), reverse=True)
    return {"workouts": docs}


@app.post("/api/fitness/ask")
def fitness_ask(data: FitnessQuestion, user: dict = Depends(get_current_user)):
    response_text = run_fitness_qa_agent(data.question)
    return {"response": response_text}


# --------------------------- Frontend ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_signup():
    return FileResponse("static/signup.html")

@app.get("/login")
def serve_login():
    return FileResponse("static/login.html")

@app.get("/profile")
def serve_profile():
    return FileResponse("static/profile.html")

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse("static/dashboard.html")
