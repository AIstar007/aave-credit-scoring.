# 💳 Aave V2 Wallet Credit Scoring

This project builds a credit scoring model for wallets using transaction-level data from the Aave V2 protocol.

## 🧠 Objective

Assign a **credit score between 0 and 1000** to each wallet based on historical actions like deposit, borrow, repay, etc.

## ⚙️ Method

- Feature Engineering: Transaction counts per action type
- Model: RandomForestRegressor
- Target: Synthetic score = +100 * deposits + 50 * repays - 150 * liquidations
- Scores scaled between 0–1000 using MinMaxScaler

## 🏁 How to Run

1. Train model:
```bash
python train_model.py
```

2. Score wallets:
```bash
python src/score_wallets.py
```

3. Plot distribution:
```bash
python plot_distribution.py
```

## 📊 Analysis

See [analysis.md](./analysis.md) for insights and charts.
