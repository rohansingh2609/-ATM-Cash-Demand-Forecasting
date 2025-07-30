import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("atm_cash_data.csv")
X = df[["day_of_week", "is_holiday", "avg_withdrawal", "location_type"]]
y = df["cash_to_load"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train better model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "atm_model.pkl")
print(" Random Forest model trained and saved.")
