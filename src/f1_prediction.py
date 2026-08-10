import fastf1 as f1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

CACHE_DIR = "cache"

YEAR = 2023
EVENT = "bahrain"
SESSION_TYPE = "r"

RANDOM_STATE = 42

FEATURES = [
    "StintLap",
    "LapNumber",
    "RollingAvg",
    "Compound",
    "Driver",
]
f1.Cache.enable_cache(CACHE_DIR)
session = f1.get_session(
    YEAR,
    EVENT,
    SESSION_TYPE
)

session.load()
laps = session.laps
clean_laps = laps.copy()
clean_laps = clean_laps[
    ~clean_laps["PitInTime"].notna()
]
clean_laps = clean_laps[
    ~clean_laps["PitOutTime"].notna()
]
clean_laps = clean_laps[
    clean_laps["TrackStatus"] == "1"
]
clean_laps = clean_laps.dropna(
    subset=["LapTime"]
)
columns_to_drop = [
    "FastF1Generated",
    "IsAccurate",
    "Deleted",
    "DeletedReason",
]
clean_laps = clean_laps.drop(
    columns=columns_to_drop,
    errors="ignore"
)
clean_laps["LapTimeSeconds"] = (
    clean_laps["LapTime"].dt.total_seconds()
)
clean_laps["LapNumber"] = (
    clean_laps["LapNumber"].astype(int)
)
clean_laps["StintLap"] = (
    clean_laps
    .groupby(["Driver", "Stint"])
    .cumcount()
    + 1
)
clean_laps["RollingAvg"] = (
    clean_laps
    .groupby("Driver")["LapTimeSeconds"]
    .rolling(
        window=4,
        min_periods=1
    )
    .mean()
    .reset_index(
        level=0,
        drop=True
    )
)
clean_laps["DeltaFromRolling"] = (
    clean_laps["LapTimeSeconds"]
    - clean_laps["RollingAvg"]
)

print("\nTyre compounds:")
print(clean_laps["Compound"].value_counts())

x = clean_laps[FEATURES]
y = clean_laps["LapTimeSeconds"]

x = pd.get_dummies(
    x,
    columns=["Compound", "Driver"],
    drop_first=True
)

print("\nFeature information:")
x.info()
print("\n" + "=" * 50)
print("RANDOM SPLIT BASELINE")
print("=" * 50)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

linear_model = LinearRegression()
linear_model.fit(
    x_train,
    y_train
)
y_pred_linear = linear_model.predict(
    x_test
)
linear_mae = mean_absolute_error(
    y_test,
    y_pred_linear
)
linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_linear
    )
)

print(f"\nLinear Regression MAE: {linear_mae:.4f}")
print(f"Linear Regression RMSE: {linear_rmse:.4f}")
plt.figure(figsize=(7, 5))
plt.scatter(
    y_test,
    y_pred_linear,
    s=10
)
plt.xlabel("Actual Lap Time")
plt.ylabel("Predicted Lap Time")
plt.title(
    "Baseline Model:Actual vs Predicted"
)
plt.tight_layout()
plt.show()

coefficient_df = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": linear_model.coef_
}).sort_values(
    by="Coefficient"
)

print("\nLinear Regression Coefficients:")
print(coefficient_df)

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_forest.fit(
    x_train,
    y_train
)

y_pred_rf = random_forest.predict(
    x_test
)

rf_mae = mean_absolute_error(
    y_test,
    y_pred_rf
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_rf
    )
)

print(f"\nRandom Forest MAE: {rf_mae:.4f}")
print(f"Random Forest RMSE: {rf_rmse:.4f}")

random_results = x_test.copy()

random_results["LinearResidual"] = (
    y_test - y_pred_linear
)

random_results["RFResidual"] = (
    y_test - y_pred_rf
)

print("\nRandom Split Residual Summary:")
print(
    random_results[
        ["LinearResidual", "RFResidual"]
    ].describe()
)

plt.figure(figsize=(10, 5))

plt.scatter(
    random_results["StintLap"],
    random_results["LinearResidual"],
    s=10,
    label="Linear Regression"
)

plt.scatter(
    random_results["StintLap"],
    random_results["RFResidual"],
    s=10,
    label="Random Forest"
)

plt.axhline(0)

plt.xlabel("Stint Lap")
plt.ylabel("Residual")
plt.title(
    "Random Split Residuals: Linear Regression vs Random Forest"
)

plt.legend()
plt.tight_layout()
plt.show()


feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": random_forest.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

print("\n" + "=" * 50)
print("CHRONOLOGICAL EVALUATION")
print("=" * 50)

clean_laps = clean_laps.sort_values(
    by=["Driver", "LapNumber"]
)

split_lap = int(
    clean_laps["LapNumber"].max() * 0.8
)

train_df = clean_laps[
    clean_laps["LapNumber"] <= split_lap
]

test_df = clean_laps[
    clean_laps["LapNumber"] > split_lap
]


x_train = train_df[FEATURES]
y_train = train_df["LapTimeSeconds"]

x_test = test_df[FEATURES]
y_test = test_df["LapTimeSeconds"]

x_train = pd.get_dummies(
    x_train,
    columns=["Compound", "Driver"],
    drop_first=True
)

x_test = pd.get_dummies(
    x_test,
    columns=["Compound", "Driver"],
    drop_first=True
)

x_test = x_test.reindex(
    columns=x_train.columns,
    fill_value=0
)


linear_model.fit(
    x_train,
    y_train
)

y_pred_linear = linear_model.predict(
    x_test
)


random_forest.fit(
    x_train,
    y_train
)

y_pred_rf = random_forest.predict(
    x_test
)

chronological_linear_mae = mean_absolute_error(
    y_test,
    y_pred_linear
)

chronological_linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_linear
    )
)

chronological_rf_mae = mean_absolute_error(
    y_test,
    y_pred_rf
)

chronological_rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_rf
    )
)

print(
    f"\nLinear Regression MAE: "
    f"{chronological_linear_mae:.4f}"
)

print(
    f"Linear Regression RMSE: "
    f"{chronological_linear_rmse:.4f}"
)

print(
    f"Random Forest MAE: "
    f"{chronological_rf_mae:.4f}"
)

print(
    f"Random Forest RMSE: "
    f"{chronological_rf_rmse:.4f}"
)

chronological_results = test_df[
    ["LapNumber", "StintLap", "Driver"]
].copy()

chronological_results["Actual"] = (
    y_test.values
)

chronological_results["LinearPrediction"] = (
    y_pred_linear
)

chronological_results["RFPrediction"] = (
    y_pred_rf
)

chronological_results["LinearResidual"] = (
    chronological_results["Actual"]
    - chronological_results["LinearPrediction"]
)

chronological_results["RFResidual"] = (
    chronological_results["Actual"]
    - chronological_results["RFPrediction"]
)


plt.figure(figsize=(10, 5))

plt.scatter(
    chronological_results["LapNumber"],
    chronological_results["LinearResidual"],
    s=10,
    label="Linear Regression"
)

plt.scatter(
    chronological_results["LapNumber"],
    chronological_results["RFResidual"],
    s=10,
    label="Random Forest"
)

plt.axhline(0)
plt.xlabel("Lap")
plt.ylabel("Residual")
plt.title(
    "Chronological Residuals: Linear Regression vs Random Forest"
)

plt.legend()
plt.tight_layout()
plt.show()

driver_consistency = (
    chronological_results
    .groupby("Driver")
    .agg(
        linear_mean=(
            "LinearResidual",
            "mean"
        ),
        linear_std=(
            "LinearResidual",
            "std"
        ),
        rf_mean=(
            "RFResidual",
            "mean"
        ),
        rf_std=(
            "RFResidual",
            "std"
        ),
        laps=(
            "LinearResidual",
            "count"
        )
    )
    .sort_values(
        "linear_std"
    )
)

print("\nDriver Prediction Consistency:")
print(driver_consistency)

driver_consistency[
    ["linear_std", "rf_std"]
].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("Residual Standard Deviation")
plt.title(
    "Driver Consistency: Linear Regression vs Random Forest"
)

plt.tight_layout()
plt.show()

late_stint_threshold = 15

late_stint = chronological_results[
    chronological_results["StintLap"]
    >= late_stint_threshold
].copy()

print("\nLate Stint Sample:")
print(late_stint.head())

late_risk_linear = (
    late_stint
    .groupby("Driver")
    .agg(
        mean_residual=(
            "LinearResidual",
            "mean"
        ),
        residual_std=(
            "LinearResidual",
            "std"
        ),
        spike_rate=(
            "LinearResidual",
            lambda x: (
                x.abs() > 0.8
            ).mean()
        ),
        laps=(
            "LinearResidual",
            "count"
        )
    )
    .sort_values(
        "residual_std"
    )
)

print("\nLate-Stint Linear Regression Risk:")
print(late_risk_linear)

late_risk_linear[
    ["residual_std", "spike_rate"]
].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("Risk Metric")
plt.title(
    "Late-Stint Risk: Linear Regression"
)

plt.tight_layout()
plt.show()

late_risk_rf = (
    late_stint
    .groupby("Driver")
    .agg(
        mean_residual=(
            "RFResidual",
            "mean"
        ),
        residual_std=(
            "RFResidual",
            "std"
        ),
        spike_rate=(
            "RFResidual",
            lambda x: (
                x.abs() > 0.8
            ).mean()
        ),
        laps=(
            "RFResidual",
            "count"
        )
    )
    .sort_values(
        "residual_std"
    )
)

print("\nLate-Stint Random Forest Risk:")
print(late_risk_rf)

late_risk_rf[
    ["residual_std", "spike_rate"]
].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("Risk Metric")
plt.title(
    "Late-Stint Risk: Random Forest"
)

plt.tight_layout()
plt.show()
print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)