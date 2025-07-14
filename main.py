import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

## Feature engineering.

proximity_map = {
    "NEAR BAY": 0,
    "<1H OCEAN": 1,
    "INLAND": 2,
    "NEAR OCEAN": 3,
    "ISLAND": 4,
}

df["ocean_proximity"] = df["ocean_proximity"].map(proximity_map)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].mean())

y = df["median_house_value"]
X = df.drop(columns=["median_house_value"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")