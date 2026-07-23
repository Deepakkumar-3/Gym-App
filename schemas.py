from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field

ActivityLevel = Literal[
    "sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"
]
Goal = Literal["muscle_gain", "weight_loss", "stay_active", "body_builder"]
Sex = Literal["male", "female"]


class SignupRequest(BaseModel):
    name: str = Field(min_length=1)
    email: EmailStr
    password: str = Field(min_length=8)
    age: int = Field(gt=0, lt=120)
    sex: Sex
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    activity_level: ActivityLevel
    goal: Goal


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(default=None, gt=0, lt=120)
    sex: Optional[Sex] = None
    weight_kg: Optional[float] = Field(default=None, gt=0)
    height_cm: Optional[float] = Field(default=None, gt=0)
    activity_level: Optional[ActivityLevel] = None
    goal: Optional[Goal] = None


class ProfileResponse(BaseModel):
    user_id: str
    name: str
    email: EmailStr
    age: int
    sex: str
    weight_kg: float
    height_cm: float
    activity_level: str
    goal: str


class NutritionQuestion(BaseModel):
    question: str = ""


class InjuryReport(BaseModel):
    message: str = Field(min_length=1)


class WorkoutLog(BaseModel):
    message: str = Field(min_length=1)


class FitnessQuestion(BaseModel):
    question: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
