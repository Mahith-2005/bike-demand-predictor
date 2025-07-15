import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
df = pd.read_csv('day.csv')  # Dataset: UCI Bike Sharing (day.csv)

# Drop unnecessary columns
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

# Define input features and target
features = ['season', 'yr', 'mnth', 'holiday', 'weekday',
            'workingday', 'temp', 'hum', 'windspeed']
X = df[features]
y = df['cnt']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save model
with open('bike_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# (Optional) Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", color='blue')
plt.plot(y_pred, label="Predicted", color='red')
plt.title("Actual vs Predicted Bike Counts")
plt.xlabel("Sample")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("result_chart.png")
plt.show()