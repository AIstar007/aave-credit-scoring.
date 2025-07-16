import json
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import extract_features
from src.model import load_model, predict_score

def main():
    with open("data/user_transactions.json") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    features_df = extract_features(df)

    model = load_model("model.pkl")
    scores = predict_score(model, features_df)

    result = pd.DataFrame({
        "userWallet": features_df["userWallet"],
        "credit_score": scores
    })

    result.to_csv("wallet_scores.csv", index=False)
    print("✅ Scoring complete! Results saved to wallet_scores.csv.")

if __name__ == "__main__":
    main()