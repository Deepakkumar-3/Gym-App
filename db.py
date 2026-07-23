"""
Astra DB connection — shared by the whole app.
All agent flows and the auth API pull collections from here.
"""

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]

_client = DataAPIClient(TOKEN)
db = _client.get_database(ENDPOINT)

users_collection = db.get_collection("users")
workout_logs_collection = db.get_collection("workout_logs")
injury_notes_collection = db.get_collection("injury_notes")
daily_goals_collection = db.get_collection("daily_goals")
fitness_kb_collection = db.get_collection("fitness_kb")
