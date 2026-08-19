"""Generate 6 unique, cleaned synthetic CSV datasets for training/practice."""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def save(df: pd.DataFrame, name: str):
    out = DATA_DIR / name
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df)} rows, {len(df.columns)} cols)")


def coffee_shop_sales(n=1500):
    drinks = ["Espresso", "Latte", "Cappuccino", "Americano", "Mocha", "Cold Brew", "Flat White", "Macchiato"]
    sizes = ["Small", "Medium", "Large"]
    branches = ["Downtown", "Uptown", "Riverside", "Airport", "Mall"]
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365, n)]
    base_price = {"Espresso": 2.5, "Latte": 4.0, "Cappuccino": 3.8, "Americano": 3.0,
                  "Mocha": 4.5, "Cold Brew": 3.5, "Flat White": 4.2, "Macchiato": 3.9}
    size_mult = {"Small": 0.85, "Medium": 1.0, "Large": 1.25}
    drink = rng.choice(drinks, n)
    size = rng.choice(sizes, n, p=[0.3, 0.45, 0.25])
    qty = rng.integers(1, 4, n)
    unit_price = np.round([base_price[d] * size_mult[s] for d, s in zip(drink, size)], 2)
    total = np.round(unit_price * qty, 2)
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "branch": rng.choice(branches, n),
        "drink": drink,
        "size": size,
        "quantity": qty,
        "unit_price": unit_price,
        "total_price": total,
        "payment_method": rng.choice(["Card", "Cash", "Mobile Wallet"], n, p=[0.5, 0.2, 0.3]),
        "loyalty_member": rng.choice([True, False], n, p=[0.4, 0.6]),
    })
    save(df, "coffee_shop_sales_dataset.csv")


def airline_flight_delays(n=2000):
    airlines = ["SkyJet", "AeroLink", "NorthWind Air", "BlueHawk", "Continental Express"]
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "SEA", "MIA", "DEN"]
    start = datetime(2025, 1, 1)
    origin = rng.choice(airports, n)
    dest = np.array([rng.choice([a for a in airports if a != o]) for o in origin])
    distance = rng.integers(200, 2800, n)
    sched_dep = [start + timedelta(days=int(d), hours=int(h)) for d, h in
                 zip(rng.integers(0, 365, n), rng.integers(5, 23, n))]
    delay_min = np.clip(rng.normal(12, 35, n), -15, None).round().astype(int)
    weather_delay = rng.choice([0, 1], n, p=[0.85, 0.15])
    delay_min = np.where(weather_delay == 1, delay_min + rng.integers(20, 90, n), delay_min)
    status = np.where(delay_min <= 0, "On Time", np.where(delay_min < 15, "Minor Delay", "Delayed"))
    cancelled = rng.choice([0, 1], n, p=[0.97, 0.03])
    status = np.where(cancelled == 1, "Cancelled", status)
    df = pd.DataFrame({
        "flight_id": [f"FL{1000+i}" for i in range(n)],
        "airline": rng.choice(airlines, n),
        "origin": origin,
        "destination": dest,
        "distance_miles": distance,
        "scheduled_departure": [d.strftime("%Y-%m-%d %H:%M") for d in sched_dep],
        "delay_minutes": np.where(cancelled == 1, 0, np.maximum(delay_min, 0)),
        "weather_delay": weather_delay.astype(bool),
        "status": status,
        "passengers": rng.integers(50, 300, n),
    })
    save(df, "airline_flight_delay_dataset.csv")


def video_game_sales(n=1200):
    genres = ["Action", "RPG", "Sports", "Shooter", "Strategy", "Puzzle", "Adventure", "Racing"]
    platforms = ["PC", "PS5", "Xbox Series X", "Switch", "Mobile"]
    publishers = ["NovaGames", "PixelForge", "Starlight Studios", "RedCircuit", "BlueOrbit Interactive"]
    release_year = rng.integers(2015, 2026, n)
    critic_score = np.clip(rng.normal(75, 12, n), 20, 99).round(1)
    base_sales = np.clip(rng.exponential(1.2, n), 0.01, 25)
    sales_mult = 1 + (critic_score - 75) / 100
    units_million = np.round(base_sales * sales_mult, 2)
    df = pd.DataFrame({
        "game_id": np.arange(1, n + 1),
        "title": [f"Game Title {i}" for i in range(1, n + 1)],
        "genre": rng.choice(genres, n),
        "platform": rng.choice(platforms, n),
        "publisher": rng.choice(publishers, n),
        "release_year": release_year,
        "critic_score": critic_score,
        "units_sold_million": units_million,
        "price_usd": rng.choice([19.99, 29.99, 39.99, 49.99, 59.99, 69.99], n),
        "multiplayer": rng.choice([True, False], n, p=[0.45, 0.55]),
    })
    save(df, "video_game_sales_dataset.csv")


def online_course_engagement(n=1800):
    courses = ["Python Basics", "Data Science 101", "Web Dev Bootcamp", "Machine Learning",
               "Digital Marketing", "UX Design", "Cloud Computing", "Cybersecurity Fundamentals"]
    devices = ["Desktop", "Mobile", "Tablet"]
    enroll_day = rng.integers(0, 300, n)
    start = datetime(2025, 1, 1)
    enroll_date = [start + timedelta(days=int(d)) for d in enroll_day]
    videos_total = rng.integers(20, 80, n)
    videos_watched = np.minimum(videos_total, np.round(videos_total * np.clip(rng.beta(2, 1.5, n), 0, 1)).astype(int))
    completion_pct = np.round(videos_watched / videos_total * 100, 1)
    quiz_avg = np.clip(rng.normal(72, 15, n) + (completion_pct - 50) * 0.1, 0, 100).round(1)
    certificate = (completion_pct >= 80) & (quiz_avg >= 60)
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "course": rng.choice(courses, n),
        "enrollment_date": [d.strftime("%Y-%m-%d") for d in enroll_date],
        "device": rng.choice(devices, n, p=[0.5, 0.4, 0.1]),
        "videos_total": videos_total,
        "videos_watched": videos_watched,
        "completion_pct": completion_pct,
        "avg_quiz_score": quiz_avg,
        "forum_posts": rng.poisson(2.5, n),
        "certificate_earned": certificate,
    })
    save(df, "online_course_engagement_dataset.csv")


def solar_power_generation(n=2200):
    plants = ["Site Alpha", "Site Beta", "Site Gamma", "Site Delta", "Site Epsilon"]
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365, n)]
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal = 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + 5
    sunlight_hours = np.clip(seasonal + rng.normal(0, 1.2, n), 1, 14).round(2)
    cloud_cover_pct = np.clip(rng.normal(40, 25, n), 0, 100).round(1)
    panel_capacity_kw = rng.choice([250, 500, 750, 1000], n)
    efficiency = np.clip(1 - cloud_cover_pct / 150, 0.3, 1)
    energy_kwh = np.round(panel_capacity_kw * sunlight_hours * efficiency * rng.uniform(0.9, 1.0, n), 1)
    temp_c = np.round(seasonal * 1.5 + 15 + rng.normal(0, 3, n), 1)
    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "plant": rng.choice(plants, n),
        "panel_capacity_kw": panel_capacity_kw,
        "sunlight_hours": sunlight_hours,
        "cloud_cover_pct": cloud_cover_pct,
        "ambient_temp_c": temp_c,
        "energy_generated_kwh": energy_kwh,
        "maintenance_flag": rng.choice([True, False], n, p=[0.05, 0.95]),
    })
    save(df, "solar_power_generation_dataset.csv")


def pet_adoption(n=1400):
    species = ["Dog", "Cat", "Rabbit", "Bird", "Guinea Pig"]
    breeds_by_species = {
        "Dog": ["Labrador Mix", "Terrier Mix", "Shepherd Mix", "Poodle Mix", "Beagle Mix"],
        "Cat": ["Domestic Shorthair", "Siamese Mix", "Maine Coon Mix", "Tabby"],
        "Rabbit": ["Dutch", "Lop", "Mixed Breed"],
        "Bird": ["Parakeet", "Cockatiel", "Finch"],
        "Guinea Pig": ["American", "Abyssinian"],
    }
    sp = rng.choice(species, n, p=[0.45, 0.4, 0.07, 0.05, 0.03])
    breed = [rng.choice(breeds_by_species[s]) for s in sp]
    age_months = rng.integers(1, 180, n)
    size = np.where(sp == "Dog", rng.choice(["Small", "Medium", "Large"], n),
                    np.where(sp == "Cat", "Medium", "Small"))
    start = datetime(2024, 6, 1)
    intake_date = [start + timedelta(days=int(d)) for d in rng.integers(0, 400, n)]
    days_in_shelter = rng.integers(1, 200, n)
    vaccinated = rng.choice([True, False], n, p=[0.85, 0.15])
    sterilized = rng.choice([True, False], n, p=[0.7, 0.3])
    adoption_fee = np.round(rng.uniform(20, 250, n), 2)
    adopted = rng.choice([True, False], n, p=[0.68, 0.32])
    df = pd.DataFrame({
        "pet_id": np.arange(1, n + 1),
        "species": sp,
        "breed": breed,
        "age_months": age_months,
        "size": size,
        "intake_date": [d.strftime("%Y-%m-%d") for d in intake_date],
        "days_in_shelter": days_in_shelter,
        "vaccinated": vaccinated,
        "sterilized": sterilized,
        "adoption_fee_usd": adoption_fee,
        "adopted": adopted,
    })
    save(df, "pet_adoption_dataset.csv")


if __name__ == "__main__":
    coffee_shop_sales()
    airline_flight_delays()
    video_game_sales()
    online_course_engagement()
    solar_power_generation()
    pet_adoption()
