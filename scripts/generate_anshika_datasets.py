"""Synthetic datasets for the anshika_project notebooks, one per algorithm
notebook (linear regression, logistic regression, decision tree
classification/regression, random forest classification/regression, knn
classification/regression, svm classification/regression). Same style as
generate_car_dataset.py: a handful of numeric features, one categorical
feature, and a target built from a linear combination plus noise.
"""
import numpy as np
import pandas as pd

OUT_DIR = "data/dataset_anshika"
n = 2500


def laptop_price():
    rng = np.random.default_rng(11)
    brand = rng.choice(["dell", "hp", "lenovo", "apple", "asus"], n, p=[0.25, 0.25, 0.2, 0.1, 0.2])
    brand_bonus = pd.Series(brand).map(
        {"dell": 0, "hp": 0, "lenovo": 1000, "apple": 25000, "asus": 2000}
    ).to_numpy()

    df = pd.DataFrame({
        "ram_gb": rng.choice([4, 8, 16, 32, 64], n),
        "storage_gb": rng.choice([128, 256, 512, 1024], n),
        "screen_size_in": rng.uniform(11.0, 17.0, n).round(1),
        "cpu_cores": rng.integers(2, 17, n),
        "brand": brand,
        "has_ssd": rng.integers(0, 2, n),
    })

    noise = rng.normal(0, 2500, n)
    df["price"] = (
        15000
        + df["ram_gb"] * 900
        + df["storage_gb"] * 25
        + df["screen_size_in"] * 600
        + df["cpu_cores"] * 1500
        + df["has_ssd"] * 4000
        + brand_bonus
        + noise
    ).clip(8000, None).round(2)

    df.to_csv(f"{OUT_DIR}/laptop_price_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def loan_approval():
    rng = np.random.default_rng(22)
    employment_type = rng.choice(["salaried", "self_employed", "business"], n, p=[0.55, 0.25, 0.2])
    employment_bonus = pd.Series(employment_type).map(
        {"salaried": 1.0, "self_employed": 0.3, "business": 0.6}
    ).to_numpy()

    df = pd.DataFrame({
        "applicant_income": rng.integers(15000, 200000, n),
        "loan_amount": rng.integers(50000, 2000000, n),
        "credit_score": rng.integers(300, 900, n),
        "employment_type": employment_type,
        "years_employed": rng.integers(0, 30, n),
        "has_default": rng.integers(0, 2, n),
    })

    class_score = (
        0.00002 * df["applicant_income"]
        - 0.0000015 * df["loan_amount"]
        + 0.004 * df["credit_score"]
        + 0.05 * df["years_employed"]
        - 1.5 * df["has_default"]
        + employment_bonus
        + rng.normal(0, 1.0, n)
    )
    df["approved"] = (class_score > np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/loan_approval_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("approved balance:\n", df["approved"].value_counts())


def student_performance():
    rng = np.random.default_rng(33)
    parental_education = rng.choice(
        ["highschool", "bachelors", "masters", "phd"], n, p=[0.35, 0.35, 0.2, 0.1]
    )
    education_bonus = pd.Series(parental_education).map(
        {"highschool": 0, "bachelors": 1, "masters": 2, "phd": 3}
    ).to_numpy()

    df = pd.DataFrame({
        "study_hours": rng.uniform(0, 10, n).round(1),
        "attendance_pct": rng.uniform(40, 100, n).round(1),
        "parental_education": parental_education,
        "extra_classes": rng.integers(0, 2, n),
        "sleep_hours": rng.uniform(4, 10, n).round(1),
    })

    class_score = (
        0.5 * df["study_hours"]
        + 0.04 * df["attendance_pct"]
        + 0.3 * education_bonus
        + 0.4 * df["extra_classes"]
        + 0.1 * df["sleep_hours"]
        + rng.normal(0, 1.0, n)
    )
    df["passed"] = (class_score > np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/student_performance_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("passed balance:\n", df["passed"].value_counts())


def house_price():
    rng = np.random.default_rng(44)
    location = rng.choice(["urban", "suburban", "rural"], n, p=[0.4, 0.4, 0.2])
    location_bonus = pd.Series(location).map(
        {"urban": 3000000, "suburban": 1200000, "rural": 0}
    ).to_numpy()

    df = pd.DataFrame({
        "area_sqft": rng.integers(400, 5000, n),
        "bedrooms": rng.integers(1, 6, n),
        "age_years": rng.integers(0, 40, n),
        "location": location,
        "has_garage": rng.integers(0, 2, n),
    })

    noise = rng.normal(0, 150000, n)
    df["price"] = (
        500000
        + df["area_sqft"] * 1800
        + df["bedrooms"] * 250000
        - df["age_years"] * 8000
        + df["has_garage"] * 200000
        + location_bonus
        + noise
    ).clip(300000, None).round(2)

    df.to_csv(f"{OUT_DIR}/house_price_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def employee_attrition():
    rng = np.random.default_rng(55)
    department = rng.choice(["sales", "it", "hr", "finance"], n, p=[0.3, 0.3, 0.2, 0.2])
    department_bonus = pd.Series(department).map(
        {"sales": -0.2, "it": 0.1, "hr": 0.0, "finance": 0.2}
    ).to_numpy()

    df = pd.DataFrame({
        "age": rng.integers(21, 60, n),
        "years_at_company": rng.integers(0, 25, n),
        "monthly_salary": rng.integers(20000, 150000, n),
        "department": department,
        "satisfaction_score": rng.integers(1, 11, n),
        "overtime": rng.integers(0, 2, n),
    })

    class_score = (
        -0.03 * df["age"]
        - 0.08 * df["years_at_company"]
        - 0.00002 * df["monthly_salary"]
        - 0.4 * df["satisfaction_score"]
        + 0.8 * df["overtime"]
        + department_bonus
        + rng.normal(0, 1.0, n)
    )
    df["left"] = (class_score > np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/employee_attrition_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("left balance:\n", df["left"].value_counts())


def crop_yield():
    rng = np.random.default_rng(66)
    soil_type = rng.choice(["loamy", "clay", "sandy", "silty"], n, p=[0.35, 0.25, 0.25, 0.15])
    soil_bonus = pd.Series(soil_type).map(
        {"loamy": 1.5, "clay": 0.5, "sandy": -0.5, "silty": 0.8}
    ).to_numpy()

    df = pd.DataFrame({
        "rainfall_mm": rng.uniform(200, 2000, n).round(1),
        "temperature_c": rng.uniform(10, 40, n).round(1),
        "soil_type": soil_type,
        "fertilizer_kg": rng.uniform(0, 200, n).round(1),
        "area_hectare": rng.uniform(0.5, 20, n).round(2),
    })

    noise = rng.normal(0, 1.5, n)
    df["yield_tons"] = (
        2.0
        + df["rainfall_mm"] * 0.004
        - (df["temperature_c"] - 25).abs() * 0.1
        + soil_bonus
        + df["fertilizer_kg"] * 0.02
        + df["area_hectare"] * 0.3
        + noise
    ).clip(0.1, None).round(2)

    df.to_csv(f"{OUT_DIR}/crop_yield_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def fruit_quality():
    rng = np.random.default_rng(77)
    color = rng.choice(["red", "green", "yellow", "orange"], n, p=[0.3, 0.3, 0.2, 0.2])
    color_bonus = pd.Series(color).map(
        {"red": 0.2, "green": 0.0, "yellow": 0.1, "orange": 0.15}
    ).to_numpy()

    df = pd.DataFrame({
        "weight_g": rng.uniform(50, 400, n).round(1),
        "diameter_cm": rng.uniform(3, 12, n).round(1),
        "sugar_content_brix": rng.uniform(5, 20, n).round(1),
        "color": color,
        "days_since_harvest": rng.integers(0, 20, n),
    })

    class_score = (
        0.01 * df["sugar_content_brix"]
        + 0.05 * df["diameter_cm"]
        + color_bonus
        - 0.15 * df["days_since_harvest"]
        + rng.normal(0, 1.0, n)
    )
    df["is_fresh"] = (class_score > np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/fruit_quality_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("is_fresh balance:\n", df["is_fresh"].value_counts())


def fitness_calories():
    rng = np.random.default_rng(88)
    workout_type = rng.choice(["running", "cycling", "swimming", "yoga"], n, p=[0.35, 0.3, 0.2, 0.15])
    workout_bonus = pd.Series(workout_type).map(
        {"running": 6.0, "cycling": 5.0, "swimming": 7.0, "yoga": 2.0}
    ).to_numpy()

    df = pd.DataFrame({
        "duration_min": rng.uniform(10, 90, n).round(1),
        "heart_rate_avg": rng.integers(80, 180, n),
        "weight_kg": rng.uniform(45, 110, n).round(1),
        "workout_type": workout_type,
        "age": rng.integers(16, 65, n),
    })

    noise = rng.normal(0, 30, n)
    df["calories_burned"] = (
        df["duration_min"] * workout_bonus
        + df["heart_rate_avg"] * 1.5
        + df["weight_kg"] * 0.8
        - df["age"] * 0.5
        + noise
    ).clip(20, None).round(2)

    df.to_csv(f"{OUT_DIR}/fitness_calories_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


def email_spam():
    rng = np.random.default_rng(99)
    sender_domain = rng.choice(["gmail", "yahoo", "outlook", "unknown"], n, p=[0.4, 0.25, 0.25, 0.1])
    domain_bonus = pd.Series(sender_domain).map(
        {"gmail": -0.3, "yahoo": -0.1, "outlook": -0.2, "unknown": 1.5}
    ).to_numpy()

    df = pd.DataFrame({
        "num_links": rng.integers(0, 20, n),
        "num_words": rng.integers(10, 500, n),
        "has_attachment": rng.integers(0, 2, n),
        "sender_domain": sender_domain,
        "exclamation_count": rng.integers(0, 15, n),
    })

    class_score = (
        0.2 * df["num_links"]
        - 0.002 * df["num_words"]
        + 0.3 * df["has_attachment"]
        + domain_bonus
        + 0.25 * df["exclamation_count"]
        + rng.normal(0, 1.0, n)
    )
    df["is_spam"] = (class_score > np.median(class_score)).astype(int)

    df.to_csv(f"{OUT_DIR}/email_spam_dataset.csv", index=False)
    print(df.head())
    print(df.shape)
    print("is_spam balance:\n", df["is_spam"].value_counts())


def delivery_time():
    rng = np.random.default_rng(111)
    vehicle_type = rng.choice(["bike", "car", "truck"], n, p=[0.4, 0.4, 0.2])
    vehicle_bonus = pd.Series(vehicle_type).map(
        {"bike": -5.0, "car": 0.0, "truck": 10.0}
    ).to_numpy()

    df = pd.DataFrame({
        "distance_km": rng.uniform(1, 50, n).round(1),
        "num_packages": rng.integers(1, 20, n),
        "traffic_level": rng.integers(1, 6, n),
        "vehicle_type": vehicle_type,
        "driver_experience_years": rng.integers(0, 20, n),
    })

    noise = rng.normal(0, 5, n)
    df["delivery_time_min"] = (
        10
        + df["distance_km"] * 2.5
        + df["num_packages"] * 1.2
        + df["traffic_level"] * 4
        + vehicle_bonus
        - df["driver_experience_years"] * 0.3
        + noise
    ).clip(5, None).round(2)

    df.to_csv(f"{OUT_DIR}/delivery_time_dataset.csv", index=False)
    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    laptop_price()
    loan_approval()
    student_performance()
    house_price()
    employee_attrition()
    crop_yield()
    fruit_quality()
    fitness_calories()
    email_spam()
    delivery_time()
