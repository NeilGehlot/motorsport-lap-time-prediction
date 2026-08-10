import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
TOTAL_LAPS = 300
CARS = ["7", "8"]
DRIVERS = {
    "7": ["PRO_A", "PRO_B", "PRO_C"],
    "8": ["PRO_D", "PRO_E", "PRO_F"],
}
DRIVER_TYPE = {
    "PRO_A": "Pro",
    "PRO_B": "Pro",
    "PRO_C": "Pro",
    "PRO_D": "Pro",
    "PRO_E": "Pro",
    "PRO_F": "Pro",
}
FEATURES = [
    "StintLap",
    "Lap",
    "Night",
    "Driver",
    "RollingAvg",
]

np.random.seed(RANDOM_STATE)
rows = []
for car in CARS:
    lap = 1
    stint = 1
    while lap <= TOTAL_LAPS:
        driver = DRIVERS[car][
            (stint - 1) % 3
        ]
        stint_length = np.random.randint(
            25,
            35
        )
        for stint_lap in range(
            1,
            stint_length + 1
        ):
            if lap > TOTAL_LAPS:
                break

            base_pace = 210
            degradation = (
                0.06 * stint_lap
            )
            driver_effect = np.random.normal(
                0,
                0.15
            )
            lap_time = (
                base_pace
                + degradation
                + driver_effect
            )
            rows.append({
                "Car": car,
                "Driver": driver,
                "Lap": lap,
                "LapTimeSeconds": lap_time,
                "Stint": stint,
                "StintLap": stint_lap,
                "Night": int(lap > 150),
                "DriverType": DRIVER_TYPE[driver],
                "Class": "Hypercar",
            })

            lap += 1

        stint += 1

wec_df = pd.DataFrame(rows)
print("\nSynthetic WEC dataset:")
print(wec_df)

clean_laps = wec_df.copy()
clean_laps["RollingAvg"] = (
    clean_laps
    .groupby("Driver")["LapTimeSeconds"]
    .rolling(
        window=3,
        min_periods=1
    )
    .mean()
    .reset_index(
        level=0,
        drop=True
    )
)

print("\n" + "=" * 50)
print("RANDOM SPLIT BASELINE")
print("=" * 50)
x = clean_laps[FEATURES]
y = clean_laps["LapTimeSeconds"]
x = pd.get_dummies(
    x,
    columns=["Driver"],
    drop_first=True
)
x_train, x_test, y_train, y_test = (
    train_test_split(
        x,
        y,
        train_size=0.3,
        random_state=RANDOM_STATE
    )
)

linear_model = LinearRegression()
linear_model.fit(
    x_train,
    y_train
)
y_pred_linear = (
    linear_model.predict(x_test)
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
print(
    f"\nLinear Regression MAE: "
    f"{linear_mae:.4f}"
)
print(
    f"Linear Regression RMSE: "
    f"{linear_rmse:.4f}"
)

wec_coefficients = pd.DataFrame({
    "Feature": x.columns,
    "Coefficient": linear_model.coef_,
}).sort_values(
    by="Coefficient"
)
print("\nLinear Regression Coefficients:")
print(wec_coefficients)

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
y_pred_rf = (
    random_forest.predict(x_test)
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
print(
    f"\nRandom Forest MAE: "
    f"{rf_mae:.4f}"
)
print(
    f"Random Forest RMSE: "
    f"{rf_rmse:.4f}"
)
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
    "Random Split Residuals"
)
plt.legend()
plt.tight_layout()
plt.show()

wec_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": (
        random_forest.feature_importances_
    ),
}).sort_values(
    by="Importance",
    ascending=False
)
print("\nRandom Forest Feature Importance:")
print(wec_importance)
print("\n" + "=" * 50)
print("CHRONOLOGICAL EVALUATION")
print("=" * 50)

clean_laps = clean_laps.sort_values(
    by=["Lap", "Driver"]
).reset_index(drop=True)
split_lap = int(
    clean_laps["Lap"].max() * 0.8
)
train_df = clean_laps[
    clean_laps["Lap"] <= split_lap
]
test_df = clean_laps[
    clean_laps["Lap"] > split_lap
]
x_train = train_df[FEATURES]
y_train = train_df[
    "LapTimeSeconds"
]
x_test = test_df[FEATURES]
y_test = test_df[
    "LapTimeSeconds"
]
x_train = pd.get_dummies(
    x_train,
    columns=["Driver"],
    drop_first=True
)
x_test = pd.get_dummies(
    x_test,
    columns=["Driver"],
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
y_pred_linear = (
    linear_model.predict(x_test)
)
random_forest.fit(
    x_train,
    y_train
)
y_pred_rf = (
    random_forest.predict(x_test)
)
chronological_linear_mae = (
    mean_absolute_error(
        y_test,
        y_pred_linear
    )
)
chronological_linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_linear
    )
)
chronological_rf_mae = (
    mean_absolute_error(
        y_test,
        y_pred_rf
    )
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
    ["Car", "Driver", "Lap", "StintLap"]
].copy()
chronological_results["Actual"] = (
    y_test.values
)
chronological_results[
    "LinearPrediction"
] = y_pred_linear
chronological_results[
    "RFPrediction"
] = y_pred_rf
chronological_results[
    "LinearResidual"
] = (
    chronological_results["Actual"]
    - chronological_results["LinearPrediction"]
)
chronological_results[
    "RFResidual"
] = (
    chronological_results["Actual"]
    - chronological_results["RFPrediction"]
)
plt.figure(figsize=(10, 5))
plt.scatter(
    chronological_results["Lap"],
    chronological_results["LinearResidual"],
    s=10,
    label="Linear Regression"
)
plt.scatter(
    chronological_results["Lap"],
    chronological_results["RFResidual"],
    s=10,
    label="Random Forest"
)
plt.axhline(0)
plt.xlabel("Lap")
plt.ylabel("Residual")
plt.title(
    "Chronological Residuals"
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
        rf_mean=(
            "RFResidual",
            "mean"
        ),
        linear_std=(
            "LinearResidual",
            "std"
        ),
        rf_std=(
            "RFResidual",
            "std"
        ),
        laps=(
            "LinearResidual",
            "count"
        ),
    )
    .sort_values(
        by="linear_std"
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
plt.ylabel(
    "Residual Standard Deviation"
)
plt.title(
    "Driver Consistency: Linear Regression vs Random Forest"
)
plt.tight_layout()
plt.show()

late_stint_threshold = 30
late_stint = chronological_results[
    chronological_results["StintLap"]
    >= late_stint_threshold
].copy()
stint_risk_linear = (
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
        ),
    )
    .sort_values(
        by="residual_std"
    )
)
print(
    "\nLate-Stint Linear Regression Risk:"
)
print(stint_risk_linear)
stint_risk_linear[
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

stint_risk_rf = (
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
        ),
    )
    .sort_values(
        by="residual_std"
    )
)
print(
    "\nLate-Stint Random Forest Risk:"
)
print(stint_risk_rf)
stint_risk_rf[
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
print("WEC ANALYSIS COMPLETE")
print("=" * 50)