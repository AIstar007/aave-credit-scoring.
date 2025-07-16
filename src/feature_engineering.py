import pandas as pd

def extract_features(df):
    grouped = df.groupby("userWallet")

    feature_data = []

    for wallet, group in grouped:
        feature_data.append({
            "userWallet": wallet,
            "num_deposits": (group["action"] == "deposit").sum(),
            "num_borrows": (group["action"] == "borrow").sum(),
            "num_repays": (group["action"] == "repay").sum(),
            "num_liquidations": (group["action"] == "liquidationcall").sum(),
            "total_txns": len(group),
        })

    return pd.DataFrame(feature_data)
