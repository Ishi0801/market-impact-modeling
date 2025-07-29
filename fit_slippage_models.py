import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load the snapshot data
df = pd.read_csv("all_minute_order_book_snapshots.csv")

# Ensure necessary columns exist
required_cols = [f'ask_px_0{i}' for i in range(10)] + [f'ask_sz_0{i}' for i in range(10)]
df = df.dropna(subset=required_cols).reset_index(drop=True)

# Sample one snapshot per stock
samples = df.groupby("stock").sample(n=1, random_state=42)

# Volumes to test
volumes = np.arange(50, 1050, 50).reshape(-1, 1)

# Function to compute slippage
def compute_slippage(x, ask_prices, ask_sizes, mid_price):
    cost, remaining = 0, x
    for price, size in zip(ask_prices, ask_sizes):
        take = min(size, remaining)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        return np.nan
    return (cost / x) - mid_price

# Loop over samples
for _, row in samples.iterrows():
    stock = row['stock']
    minute = row['minute']
    ask_prices = [row[f'ask_px_0{i}'] for i in range(10)]
    ask_sizes = [row[f'ask_sz_0{i}'] for i in range(10)]
    bid = row['bid_px_00'] if pd.notna(row['bid_px_00']) else ask_prices[0] - 0.05
    ask = ask_prices[0]
    mid_price = (bid + ask) / 2

    # Compute slippage values
    y = np.array([compute_slippage(x[0], ask_prices, ask_sizes, mid_price) for x in volumes])
    mask = ~np.isnan(y)
    X = volumes[mask]
    y = y[mask]

    # Fit linear model
    lin_model = LinearRegression().fit(X, y)
    y_lin_pred = lin_model.predict(X)
    r2_lin = r2_score(y, y_lin_pred)

    # Fit quadratic model
    poly = PolynomialFeatures(degree=2)
    X_quad = poly.fit_transform(X)
    quad_model = LinearRegression().fit(X_quad, y)
    y_quad_pred = quad_model.predict(X_quad)
    r2_quad = r2_score(y, y_quad_pred)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='black', label="Observed Slippage")
    plt.plot(X, y_lin_pred, 'r--', label=f"Linear Fit (R² = {r2_lin:.3f})")
    plt.plot(X, y_quad_pred, 'b-', label=f"Quadratic Fit (R² = {r2_quad:.3f})")
    plt.title(f"{stock} @ {minute} — Slippage vs Volume")
    plt.xlabel("Volume (shares)")
    plt.ylabel("Slippage ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fit_slippage_{stock}.png")
    plt.show()

    print(f"Stock: {stock} | Time: {minute}")
    print(f"R² (Linear): {r2_lin:.4f}")
    print(f"R² (Quadratic): {r2_quad:.4f}")
