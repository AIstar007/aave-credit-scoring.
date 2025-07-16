# 💳 Aave V2 Wallet Credit Scoring

This project builds a machine learning-based **credit scoring system** for wallets interacting with the Aave V2 DeFi protocol. The score ranges from 0 to 1000, representing the reliability of user behavior.

---

## 🧠 Objective

Given raw transaction data from the Aave V2 protocol, develop a model that assigns credit scores to wallets. The higher the score, the more trustworthy the wallet is based on past behaviors (like deposit, borrow, repay, and liquidation).

---

## 🏗️ Architecture & Processing Flow

```
user_transactions.json
│
▼ [Feature Engineering] (src/feature_engineering.py)
│
▼ Aggregated Features
│
▼ [Model Training] (train_model.py)
│
▼ Trained Model (model.pkl)
│
▼ [Wallet Scoring] (src/score_wallets.py)
│
▼ wallet_scores.csv
│
▼ [Visualization] (plot_distribution.py → score_distribution.png)
```

---

## ⚙️ Methodology

### 🧩 Feature Engineering

From raw transaction records, we computed wallet-level features:

- `num_deposits`
- `num_borrows`
- `num_repays`
- `num_liquidations`
- `total_txns`

These are extracted by grouping all transactions per wallet using the `userWallet` field.

---

### 🤖 Model

- **Model Used**: `RandomForestRegressor`
- **Synthetic Target**:
    ```python
    score = (100 * deposits) + (50 * repays) - (150 * liquidations)
    ```
- **Normalization**: Scaled scores to range `0–1000` using `MinMaxScaler`

---

## 🚀 How to Run

1. **Train the model**
    ```bash
    python train_model.py
    ```

2. **Score wallets**
    ```bash
    python -m src.score_wallets
    ```

3. **Plot score distribution**
    ```bash
    python plot_distribution.py
    ```

---

## 📁 Project Structure

```
aave-credit-scoring/
├── data/user_transactions.json
├── model.pkl
├── wallet_scores.csv
├── score_distribution.png
├── train_model.py
├── plot_distribution.py
├── src/
│   ├── feature_engineering.py
│   ├── model.py
│   └── score_wallets.py
├── README.md
├── analysis.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Analysis

See [analysis.md](./analysis.md) for detailed breakdown of score distribution and wallet behavior across score ranges.