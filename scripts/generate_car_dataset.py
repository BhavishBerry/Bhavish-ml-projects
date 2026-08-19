"""Used car pricing dataset. Simple, relatable features like the salary
one, but a different domain. One regression target (resale price) and one
classification target (good_deal) from the same features, for
Linear/Logistic Regression, Decision Tree, Random Forest, and SVM practice.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(99)
n = 3000

fuel_type = rng.choice(["petrol", "diesel", "electric"], n, p=[0.55, 0.3, 0.15])
fuel_bonus = pd.Series(fuel_type).map({"petrol": 0, "diesel": 3000, "electric": 9000}).to_numpy()

df = pd.DataFrame({
    "car_age_years": rng.integers(0, 15, n),
    "mileage_km": rng.integers(0, 200000, n),
    "engine_size_l": rng.uniform(1.0, 4.0, n).round(1),
    "num_previous_owners": rng.integers(1, 5, n),
    "fuel_type": fuel_type,
    "had_accident": rng.integers(0, 2, n),
    "service_history_score": rng.integers(1, 11, n),
})

noise = rng.normal(0, 1500, n)
df["resale_price"] = (
    30000
    - df["car_age_years"] * 1400
    - df["mileage_km"] * 0.05
    + df["engine_size_l"] * 1200
    - df["num_previous_owners"] * 800
    - df["had_accident"] * 4000
    + df["service_history_score"] * 600
    + fuel_bonus
    + noise
).clip(500, None).round(2)

class_score = (
    -0.3 * df["car_age_years"]
    - 0.00002 * df["mileage_km"]
    + 0.5 * df["engine_size_l"]
    - 0.4 * df["num_previous_owners"]
    - 1.2 * df["had_accident"]
    + 0.3 * df["service_history_score"]
    + fuel_bonus * 0.0001
    + rng.normal(0, 1.0, n)
)
df["good_deal"] = (class_score > np.median(class_score)).astype(int)

df.to_csv("data/used_car_dataset.csv", index=False)
print(df.head())
print(df.shape)
print("good_deal balance:\n", df["good_deal"].value_counts())
