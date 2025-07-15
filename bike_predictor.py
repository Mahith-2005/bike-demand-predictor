# bike_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("day.csv")
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

# Define features and target
features = ['season', 'yr', 'mnth', 'holiday', 'weekday',
            'workingday', 'temp', 'hum', 'windspeed']
X = df[features]
y = df['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.plot(y_test.values[:50], label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Bike Count")
plt.xlabel("Sample")
plt.ylabel("Bike Count")
plt.show()
