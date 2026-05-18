"""
Generate demo datasets for testing DataSage AI.
Includes an industrial predictive maintenance dataset — ABB's core domain.
"""
import pandas as pd
import numpy as np
from pathlib import Path

out = Path("demo_data")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

n = 400

# 1. Regression — House Prices
df_reg = pd.DataFrame({
    "area_sqft": rng.integers(500, 5000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 4, n),
    "age_years": rng.integers(0, 50, n),
    "garage": rng.choice([0, 1], n),
    "distance_to_center_km": rng.uniform(1, 30, n).round(2),
    "price_usd": None,
})
df_reg["price_usd"] = (
    df_reg["area_sqft"] * 120 + df_reg["bedrooms"] * 8000
    + df_reg["bathrooms"] * 5000 - df_reg["age_years"] * 500
    + df_reg["garage"] * 15000 - df_reg["distance_to_center_km"] * 1200
    + rng.normal(0, 20000, n)
).round(0).astype(int)
df_reg.to_csv(out / "house_prices.csv", index=False)
print("[OK] house_prices.csv")

# 2. Classification — Customer Churn
df_clf = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_charges": rng.uniform(20, 120, n).round(2),
    "num_products": rng.integers(1, 5, n),
    "has_support_plan": rng.choice([0, 1], n),
    "complaints_last_year": rng.integers(0, 5, n),
    "satisfaction_score": rng.integers(1, 10, n),
})
churn_prob = 1 / (1 + np.exp(-(
    -df_clf["tenure_months"] * 0.05 + df_clf["monthly_charges"] * 0.03
    + df_clf["complaints_last_year"] * 0.4 - df_clf["satisfaction_score"] * 0.2
)))
df_clf["churn"] = (rng.uniform(0, 1, n) < churn_prob).astype(int).map({0: "No", 1: "Yes"})
df_clf.to_csv(out / "customer_churn.csv", index=False)
print("[OK] customer_churn.csv")

# 3. Clustering — Customer Segments
df_clust = pd.DataFrame({
    "annual_income_k": rng.uniform(20, 200, n).round(1),
    "spending_score": rng.integers(1, 100, n),
    "age": rng.integers(18, 70, n),
    "purchase_frequency": rng.integers(1, 52, n),
    "avg_order_value": rng.uniform(15, 500, n).round(2),
})
df_clust.to_csv(out / "customer_segments.csv", index=False)
print("[OK] customer_segments.csv")

# 4. Time Series — Monthly Sales
months = pd.date_range("2020-01-01", periods=60, freq="ME")
trend = np.linspace(1000, 3000, 60)
seasonal = 400 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = rng.normal(0, 150, 60)
df_ts = pd.DataFrame({
    "date": months.strftime("%Y-%m-%d"),
    "sales": (trend + seasonal + noise).round(0).astype(int),
    "marketing_spend": rng.uniform(500, 2000, 60).round(0).astype(int),
    "region": rng.choice(["North", "South", "East", "West"], 60),
})
df_ts.to_csv(out / "monthly_sales.csv", index=False)
print("[OK] monthly_sales.csv")

# 5. INDUSTRIAL — ABB Predictive Maintenance (the KEY differentiator)
# Simulates motor/pump sensor data typical of ABB industrial assets
n_ind = 500
op_hours = rng.uniform(100, 10000, n_ind)
vibration_x = rng.normal(0.5, 0.15, n_ind) + op_hours * 0.00005
vibration_y = rng.normal(0.4, 0.12, n_ind) + op_hours * 0.00004
temp_bearing = rng.normal(65, 8, n_ind) + op_hours * 0.003
temp_stator = rng.normal(80, 10, n_ind) + op_hours * 0.002
current_phase_a = rng.normal(48, 5, n_ind)
current_phase_b = rng.normal(48, 5, n_ind)
current_phase_c = rng.normal(48, 5, n_ind)
voltage_imbalance = rng.uniform(0, 3, n_ind)
rpm = rng.normal(1480, 20, n_ind)
oil_viscosity = 100 - op_hours * 0.005 + rng.normal(0, 3, n_ind)
load_factor = rng.uniform(0.5, 1.0, n_ind)

# Remaining Useful Life (RUL) — regression target
rul = (
    8000 - op_hours
    - vibration_x * 200
    - (temp_bearing - 65) * 15
    - (voltage_imbalance * 30)
    + rng.normal(0, 100, n_ind)
).clip(0, 8000).round(0).astype(int)

# Fault classification target
fault_score = (
    vibration_x * 0.3 + (temp_bearing - 65) / 30
    + voltage_imbalance * 0.2 + op_hours / 10000 * 0.5
)
fault_probs = 1 / (1 + np.exp(-(fault_score - 0.7) * 4))
fault_class = np.where(
    fault_probs > 0.7, "Imminent Failure",
    np.where(fault_probs > 0.4, "Degraded", "Healthy")
)

df_industrial = pd.DataFrame({
    "asset_id": [f"MOTOR-{i:04d}" for i in range(n_ind)],
    "operating_hours": op_hours.round(1),
    "vibration_x_rms": vibration_x.round(4),
    "vibration_y_rms": vibration_y.round(4),
    "bearing_temp_c": temp_bearing.round(2),
    "stator_temp_c": temp_stator.round(2),
    "phase_a_current_a": current_phase_a.round(2),
    "phase_b_current_a": current_phase_b.round(2),
    "phase_c_current_a": current_phase_c.round(2),
    "voltage_imbalance_pct": voltage_imbalance.round(3),
    "shaft_rpm": rpm.round(1),
    "oil_viscosity_cst": oil_viscosity.round(2),
    "load_factor": load_factor.round(3),
    "remaining_useful_life_h": rul,
    "fault_status": fault_class,
})
df_industrial.to_csv(out / "abb_motor_predictive_maintenance.csv", index=False)
print("[OK] abb_motor_predictive_maintenance.csv  [INDUSTRIAL DEMO]")

print("\nAll demo datasets created in ./demo_data/")
print("Upload any of these to DataSage AI at http://localhost:8000")
print("\nKey dataset for ABB judges: abb_motor_predictive_maintenance.csv")
print("  Regression target: remaining_useful_life_h")
print("  Classification target: fault_status")
