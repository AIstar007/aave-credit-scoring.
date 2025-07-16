import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.feature_engineering import extract_features

# Load JSON data
with open("data/user_transactions.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Extract features
features_df = extract_features(df)
X = features_df.drop(columns=["userWallet"])

# Simulate scores (for training only)
y = X["num_deposits"] * 100 + X["num_repays"] * 50 - X["num_liquidations"] * 150
y = MinMaxScaler((0, 1000)).fit_transform(y.values.reshape(-1, 1)).ravel()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")