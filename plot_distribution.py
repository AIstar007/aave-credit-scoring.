import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wallet_scores.csv")

plt.hist(df["credit_score"], bins=10, range=(0, 1000), edgecolor='black')
plt.title("Credit Score Distribution")
plt.xlabel("Score Range")
plt.ylabel("Number of Wallets")
plt.grid(True)
plt.savefig("score_distribution.png")
plt.show()