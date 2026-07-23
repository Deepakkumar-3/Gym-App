"""
Populates the fitness_kb collection with fitness/training/nutrition
knowledge for the Fitness Q&A Agent to search over.

Each entry's `text` gets embedded automatically by Astra's NVIDIA
vectorize integration (the $vectorize field) — no manual embedding step.

Run once (safe to re-run — it'll just add duplicates, so only run
again if you've cleared the collection first):
  python ingest_fitness_kb.py
"""

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient

load_dotenv()

ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]

client = DataAPIClient(TOKEN)
db = client.get_database(ENDPOINT)
fitness_kb = db.get_collection("fitness_kb")

ARTICLES = [
    {
        "topic": "progressive_overload",
        "text": "Progressive overload means gradually increasing the demand "
        "placed on your muscles over time — more weight, more reps, more "
        "sets, or less rest between sets. Without it, your body has no "
        "reason to keep adapting, and gains stall. A simple approach: add "
        "small amounts of weight or one extra rep each week on your main "
        "lifts, as long as your form stays solid.",
    },
    {
        "topic": "warm_up",
        "text": "A good warm-up raises muscle temperature and prepares "
        "joints for load, which lowers injury risk and often improves "
        "performance. Five to ten minutes of light cardio followed by a "
        "few lighter warm-up sets of your first exercise (working up "
        "gradually to your working weight) is usually enough — you don't "
        "need a long, separate stretching routine beforehand.",
    },
    {
        "topic": "protein_basics",
        "text": "Protein supplies the amino acids used to repair and build "
        "muscle tissue after training. General ranges people training "
        "regularly often aim for are roughly 1.6 to 2.2 grams of protein "
        "per kilogram of bodyweight per day, spread across meals rather "
        "than eaten all at once, since the body can only use so much "
        "protein for muscle repair in a single sitting.",
    },
    {
        "topic": "sleep_and_recovery",
        "text": "Most muscle repair and hormone regulation (including "
        "growth hormone release) happens during deep sleep, not during "
        "the workout itself. Consistently getting under 6-7 hours of "
        "sleep is associated with slower recovery, reduced strength "
        "performance, and a higher risk of injury, even if training and "
        "nutrition are otherwise on point.",
    },
    {
        "topic": "doms_soreness",
        "text": "Delayed Onset Muscle Soreness (DOMS) is the stiffness and "
        "soreness that shows up 24-72 hours after unfamiliar or "
        "particularly intense exercise. It's a normal adaptation response, "
        "not an indicator of how effective the workout was. Light "
        "movement, walking, and staying hydrated can ease it; sharp, "
        "localized joint pain is different from DOMS and worth paying "
        "attention to.",
    },
    {
        "topic": "rest_days",
        "text": "Rest days let muscles repair and adapt to the training "
        "stress you've placed on them — this is actually when strength "
        "and size gains happen, not during the workout itself. Training "
        "the exact same muscle group with no recovery time tends to "
        "reduce performance and raises injury risk. Most people do well "
        "with at least one to two full rest (or active recovery) days "
        "per week.",
    },
    {
        "topic": "cardio_types",
        "text": "Steady-state cardio (a consistent, moderate pace held for "
        "an extended period) builds aerobic endurance with low recovery "
        "cost. Interval training (alternating high-intensity bursts with "
        "recovery periods) tends to improve both aerobic and anaerobic "
        "fitness in less total time but demands more recovery. Neither is "
        "strictly \"better\" — the right choice depends on your goal and "
        "how it fits around your strength training.",
    },
    {
        "topic": "compound_vs_isolation",
        "text": "Compound exercises (squats, deadlifts, bench press, rows, "
        "overhead press) work multiple joints and muscle groups at once "
        "and are generally the most time-efficient way to build overall "
        "strength. Isolation exercises (bicep curls, leg extensions, "
        "lateral raises) target one muscle or joint and are useful for "
        "addressing specific weak points once a compound-lift base is in "
        "place.",
    },
    {
        "topic": "squat_form_basics",
        "text": "In a basic squat, feet are roughly shoulder-width apart, "
        "the chest stays up, and the knees track in the same direction as "
        "the toes rather than caving inward. Depth (how low you go) "
        "should be limited by maintaining a neutral spine and control, "
        "not pushed past the point where the lower back rounds. Building "
        "up weight gradually with strict form matters more long-term than "
        "the amount lifted early on.",
    },
    {
        "topic": "deload_weeks",
        "text": "A deload week is a planned period of reduced training "
        "volume or intensity, usually every 4-8 weeks of consistent "
        "training, that lets the body fully recover from accumulated "
        "fatigue before it turns into overtraining or injury. It's not a "
        "sign of weakness — many structured programs build deloads in by "
        "design, and skipping them repeatedly often leads to stalled "
        "progress or nagging injuries.",
    },
    {
        "topic": "stretching_timing",
        "text": "Static stretching (holding a stretch for an extended "
        "period) is generally better suited to after a workout or on rest "
        "days, since doing it right before heavy lifting can temporarily "
        "reduce muscle power output. Dynamic stretching (moving through a "
        "range of motion, like leg swings or arm circles) is better "
        "suited to warming up before training.",
    },
    {
        "topic": "muscle_protein_synthesis_window",
        "text": "The idea of a strict 30-minute \"anabolic window\" after "
        "training for eating protein has been largely overstated in "
        "popular fitness culture — total daily protein intake and "
        "consistency matter far more than eating protein within a narrow "
        "post-workout window. Eating a reasonable meal with protein "
        "within a few hours of training is plenty for most people.",
    },
    {
        "topic": "injury_prevention_basics",
        "text": "Most training injuries come from doing too much too "
        "soon (jumping weight or volume up too fast), poor form under "
        "fatigue, or skipping warm-ups — not from strength training being "
        "inherently dangerous. Gradual progression, proper form, and "
        "listening to pain that doesn't resolve after a rest day (as "
        "opposed to normal muscle soreness) are the main ways to avoid "
        "most common injuries.",
    },
    {
        "topic": "rpe_and_rir",
        "text": "RPE (Rate of Perceived Exertion) and RIR (Reps in "
        "Reserve) are ways to gauge training intensity relative to your "
        "current capacity, rather than relying only on fixed percentages "
        "of a one-rep max. Training most sets at an RPE of around 7-9 (or "
        "1-3 reps in reserve) — hard, but with a little left in the tank "
        "— tends to build strength and muscle effectively while managing "
        "fatigue.",
    },
    {
        "topic": "beginner_mistakes",
        "text": "Common beginner mistakes include increasing weight too "
        "quickly at the expense of form, skipping warm-ups, doing too "
        "much volume too soon without adequate recovery, comparing "
        "progress to more experienced lifters, and constantly switching "
        "programs before giving one enough time (typically 8-12 weeks) to "
        "actually show results.",
    },
    {
        "topic": "training_plateaus",
        "text": "A plateau — when progress stalls despite consistent "
        "training — is often caused by insufficient progressive overload, "
        "inadequate recovery (sleep, nutrition, stress), or simply "
        "needing a program change after the body has fully adapted to the "
        "current routine. Tracking workouts over time makes it much "
        "easier to spot exactly where a plateau started and why.",
    },
    {
        "topic": "creatine_basics",
        "text": "Creatine monohydrate is one of the most well-researched "
        "sports supplements, commonly associated with modest improvements "
        "in strength and power output for repeated short bursts of "
        "effort. A typical approach is a steady daily dose (often cited "
        "around 3-5 grams) rather than large front-loading doses. It is "
        "not a substitute for consistent training and adequate protein "
        "intake, just a potential complement to them.",
    },
    {
        "topic": "hydration_basics",
        "text": "Even mild dehydration (as little as 2% of bodyweight in "
        "fluid loss) can measurably reduce strength and endurance "
        "performance. Thirst is a lagging indicator, so it helps to drink "
        "water consistently throughout the day rather than only during a "
        "workout, and to drink more on hot days or during longer, "
        "sweatier training sessions.",
    },
    {
        "topic": "consistency_over_intensity",
        "text": "A moderately effective program followed consistently "
        "over months tends to outperform a theoretically \"optimal\" "
        "program that gets abandoned after a few weeks. Sustainable "
        "habits — showing up regularly, tracking progress, and adjusting "
        "gradually — matter more for long-term results than finding the "
        "single best exercise or program.",
    },
    {
        "topic": "bmi_limitations",
        "text": "BMI (Body Mass Index) is a simple weight-to-height ratio "
        "that doesn't distinguish between muscle and fat mass, so it can "
        "be misleading for people who train regularly and carry more "
        "muscle than average. It can be a rough population-level "
        "screening tool, but body composition (like measured body fat "
        "percentage) gives a more accurate individual picture for someone "
        "actively strength training.",
    },
]


def main():
    print(f"Connected to database: {db.info().name}")
    print(f"Inserting {len(ARTICLES)} articles into 'fitness_kb'...\n")

    for article in ARTICLES:
        doc = {"topic": article["topic"], "$vectorize": article["text"]}
        fitness_kb.insert_one(doc)
        print(f"  inserted: {article['topic']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
