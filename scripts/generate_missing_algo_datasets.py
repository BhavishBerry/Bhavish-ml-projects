"""Synthetic datasets for the algorithm notebooks that were missing a
standalone version (decision tree regression, random forest
classification/regression, knn classification/regression, svm regression).
Same style as generate_car_dataset.py: a handful of numeric features, one
categorical feature, and a target built from a linear combination plus noise.
"""
import numpy as np
import pandas as pd

OUT_DIR = "data"
n = 2500


def smartphone_price():
    rng = np.random.default_rng(211)
    brand = rng.choice(["samsung", "xiaomi", "apple", "oneplus"], n, p=[0.3, 0.3, 0.2, 0.2])
    brand_bonus = pd.Series(brand).map(
        {"samsung": 2000, "xiaomi": 0, "apple": 30000, "oneplus": 3000}
    ).to_numpy()

    df = pd.DataFrame({
        "ram_gb": rng.choice([4, 6, 8, 12, 16], n),
        "storage_gb": rng.choice([64, 128, 256, 512], n),
        "battery_mah": rng.integers(3000, 6000, n),
        "brand": brand,
        "camera_mp": rng.choice([12, 48, 64, 108, 200], n),
    })

    noise = rng.normal(0, 1500, n)
    df["price"] = (
        5000
        + df["ram_gb"] * 800
        + df["storage_gb"] * 20
        + df["battery_mah"] * 1.5
        + df["camera_mp"] * 50
        + brand_bonus
        + noise
    ).clip(4000, None).round(2)

    df.to_csv(f"{OUT_DIR}/smartphone_price_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def customer_churn():
    rng = np.random.default_rng(212)
    contract_type = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.5, 0.3, 0.2])
    contract_bonus = pd.Series(contract_type).map(
        {"month-to-month": -1.5, "one-year": 0.3, "two-year": 1.2}
    ).to_numpy()

    df = pd.DataFrame({
        "tenure_months": rng.integers(0, 72, n),
        "monthly_charges": rng.uniform(20, 150, n).round(2),
        "contract_type": contract_type,
        "support_calls": rng.integers(0, 10, n),
        "has_addon": rng.integers(0, 2, n),
    })

    class_score = (
        0.05 * df["tenure_months"]
        - 0.01 * df["monthly_charges"]
        + contract_bonus
        - 0.4 * df["support_calls"]
        + 0.3 * df["has_addon"]
        + rng.normal(0, 1.0, n)
    )
    df["churned"] = (class_score < np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/customer_churn_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("churned balance:\n", df["churned"].value_counts())


def concrete_strength():
    rng = np.random.default_rng(213)
    mix_type = rng.choice(["standard", "highperf", "lightweight"], n, p=[0.5, 0.3, 0.2])
    mix_bonus = pd.Series(mix_type).map(
        {"standard": 0, "highperf": 8, "lightweight": -5}
    ).to_numpy()

    df = pd.DataFrame({
        "cement_kg": rng.uniform(100, 500, n).round(1),
        "water_kg": rng.uniform(100, 250, n).round(1),
        "age_days": rng.integers(1, 365, n),
        "mix_type": mix_type,
        "aggregate_kg": rng.uniform(700, 1200, n).round(1),
    })

    noise = rng.normal(0, 3, n)
    df["compressive_strength"] = (
        10
        + df["cement_kg"] * 0.08
        - df["water_kg"] * 0.05
        + np.log1p(df["age_days"]) * 3
        + mix_bonus
        + df["aggregate_kg"] * 0.01
        + noise
    ).clip(2, None).round(2)

    df.to_csv(f"{OUT_DIR}/concrete_strength_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def wine_quality():
    rng = np.random.default_rng(214)
    wine_type = rng.choice(["red", "white", "rose"], n, p=[0.45, 0.45, 0.1])
    type_bonus = pd.Series(wine_type).map(
        {"red": 0.2, "white": 0.0, "rose": 0.1}
    ).to_numpy()

    df = pd.DataFrame({
        "acidity": rng.uniform(3.0, 9.0, n).round(2),
        "sugar_content": rng.uniform(0.5, 15.0, n).round(2),
        "alcohol_pct": rng.uniform(8.0, 15.0, n).round(1),
        "wine_type": wine_type,
        "ph_level": rng.uniform(2.8, 4.0, n).round(2),
    })

    class_score = (
        -0.1 * df["acidity"]
        + 0.05 * df["sugar_content"]
        + 0.3 * df["alcohol_pct"]
        + type_bonus
        - 0.2 * df["ph_level"]
        + rng.normal(0, 1.0, n)
    )
    df["is_good"] = (class_score > np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/wine_quality_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("is_good balance:\n", df["is_good"].value_counts())


def used_bike_price():
    rng = np.random.default_rng(215)
    bike_type = rng.choice(["sport", "cruiser", "commuter", "scooter"], n, p=[0.25, 0.25, 0.3, 0.2])
    type_bonus = pd.Series(bike_type).map(
        {"sport": 25000, "cruiser": 15000, "commuter": 0, "scooter": -5000}
    ).to_numpy()

    df = pd.DataFrame({
        "age_years": rng.integers(0, 15, n),
        "km_driven": rng.integers(500, 80000, n),
        "engine_cc": rng.integers(100, 1000, n),
        "bike_type": bike_type,
        "num_owners": rng.integers(1, 4, n),
    })

    noise = rng.normal(0, 3000, n)
    df["price"] = (
        90000
        - df["age_years"] * 4500
        - df["km_driven"] * 0.3
        + df["engine_cc"] * 60
        + type_bonus
        - df["num_owners"] * 2000
        + noise
    ).clip(5000, None).round(2)

    df.to_csv(f"{OUT_DIR}/used_bike_price_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def electricity_bill():
    rng = np.random.default_rng(216)
    area_type = rng.choice(["urban", "rural", "semiurban"], n, p=[0.45, 0.25, 0.3])
    area_bonus = pd.Series(area_type).map(
        {"urban": 150, "rural": -80, "semiurban": 20}
    ).to_numpy()

    df = pd.DataFrame({
        "units_consumed": rng.uniform(50, 1200, n).round(1),
        "household_size": rng.integers(1, 8, n),
        "ac_units": rng.integers(0, 5, n),
        "area_type": area_type,
        "season_factor": rng.uniform(0.8, 1.5, n).round(2),
    })

    noise = rng.normal(0, 80, n)
    df["bill_amount"] = (
        100
        + df["units_consumed"] * 7.5
        + df["household_size"] * 50
        + df["ac_units"] * 300
        + area_bonus
        * df["season_factor"]
        + noise
    ).clip(50, None).round(2)

    df.to_csv(f"{OUT_DIR}/electricity_bill_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    smartphone_price()
    customer_churn()
    concrete_strength()
    wine_quality()
    used_bike_price()
    electricity_bill()
