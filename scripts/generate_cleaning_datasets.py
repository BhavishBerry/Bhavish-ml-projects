"""Three messy synthetic datasets for the data cleaning project notebooks.
Unlike the other generate_*.py scripts, these intentionally inject missing
values, duplicate rows, and outliers so there is something to clean.
"""
import numpy as np
import pandas as pd

n = 1500


def retail_sales():
    rng = np.random.default_rng(401)
    category = rng.choice(["electronics", "clothing", "grocery", "furniture"], n, p=[0.3, 0.3, 0.25, 0.15])

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "product_category": category,
        "quantity": rng.integers(1, 10, n),
        "unit_price": rng.uniform(5, 500, n).round(2),
        "discount_pct": rng.uniform(0, 30, n).round(1),
        "customer_age": rng.integers(18, 70, n),
    })
    df["total_amount"] = (df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100)).round(2)

    # inject missing values
    missing_idx = rng.choice(n, 60, replace=False)
    df.loc[missing_idx, "discount_pct"] = np.nan
    missing_idx2 = rng.choice(n, 40, replace=False)
    df.loc[missing_idx2, "customer_age"] = np.nan

    # inject outliers in unit_price
    outlier_idx = rng.choice(n, 12, replace=False)
    df.loc[outlier_idx, "unit_price"] = rng.uniform(4000, 9000, 12).round(2)

    # inject duplicate rows
    dup_rows = df.sample(25, random_state=401)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv("data/retail_sales_dataset.csv", index=False)
    print(df.shape)


def hospital_patient():
    rng = np.random.default_rng(402)
    department = rng.choice(["cardiology", "orthopedics", "neurology", "general"], n, p=[0.25, 0.25, 0.2, 0.3])
    gender = rng.choice(["male", "female"], n)

    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": rng.integers(1, 95, n),
        "gender": gender,
        "blood_pressure": rng.integers(90, 160, n),
        "cholesterol": rng.uniform(120, 300, n).round(1),
        "bmi": rng.uniform(15, 40, n).round(1),
        "department": department,
    })

    missing_idx = rng.choice(n, 70, replace=False)
    df.loc[missing_idx, "cholesterol"] = np.nan
    missing_idx2 = rng.choice(n, 50, replace=False)
    df.loc[missing_idx2, "bmi"] = np.nan

    outlier_idx = rng.choice(n, 10, replace=False)
    df.loc[outlier_idx, "blood_pressure"] = rng.integers(220, 280, 10)

    dup_rows = df.sample(20, random_state=402)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv("data/hospital_patient_dataset.csv", index=False)
    print(df.shape)


def gym_membership():
    rng = np.random.default_rng(403)
    membership_type = rng.choice(["basic", "standard", "premium"], n, p=[0.4, 0.4, 0.2])

    df = pd.DataFrame({
        "member_id": np.arange(1, n + 1),
        "age": rng.integers(16, 65, n),
        "membership_type": membership_type,
        "monthly_fee": pd.Series(membership_type).map({"basic": 20, "standard": 40, "premium": 70}).to_numpy(),
        "visits_per_month": rng.integers(0, 30, n),
        "weight_kg": rng.uniform(45, 120, n).round(1),
    })

    missing_idx = rng.choice(n, 55, replace=False)
    df.loc[missing_idx, "weight_kg"] = np.nan

    outlier_idx = rng.choice(n, 8, replace=False)
    df.loc[outlier_idx, "visits_per_month"] = rng.integers(80, 120, 8)

    dup_rows = df.sample(15, random_state=403)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv("data/gym_membership_dataset.csv", index=False)
    print(df.shape)


if __name__ == "__main__":
    retail_sales()
    hospital_patient()
    gym_membership()
