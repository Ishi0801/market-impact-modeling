# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load your snapshot file
# df = pd.read_csv("all_minute_order_book_snapshots.csv")

# # Drop rows with missing top-10 ask data
# required_columns = [f'ask_px_0{i}' for i in range(10)] + [f'ask_sz_0{i}' for i in range(10)]
# df = df.dropna(subset=required_columns).reset_index(drop=True)

# # Sample N random snapshots per stock
# samples = df.groupby("stock").sample(n=3, random_state=42)

# # Volume levels to simulate
# volumes = np.arange(50, 1050, 50)

# # Function to compute slippage
# def compute_slippage(x, ask_prices, ask_sizes, mid_price):
#     cost, remaining = 0, x
#     for price, size in zip(ask_prices, ask_sizes):
#         take = min(size, remaining)
#         cost += take * price
#         remaining -= take
#         if remaining <= 0:
#             break
#     if remaining > 0:
#         return np.nan
#     return (cost / x) - mid_price

# # Plot
# fig, axs = plt.subplots(len(samples), 1, figsize=(8, 4 * len(samples)), constrained_layout=True)

# for i, (_, row) in enumerate(samples.iterrows()):
#     ask_prices = [row[f'ask_px_0{j}'] for j in range(10)]
#     ask_sizes = [row[f'ask_sz_0{j}'] for j in range(10)]
#     bid = row.get('bid_px_00', ask_prices[0] - 0.05)
#     ask = ask_prices[0]
#     mid_price = (bid + ask) / 2

#     slippages = [compute_slippage(x, ask_prices, ask_sizes, mid_price) for x in volumes]

#     axs[i].plot(volumes, slippages, marker='o')
#     axs[i].set_title(f"{row['stock']} @ {row['minute']} — Slippage vs Volume")
#     axs[i].set_xlabel("Order Volume (shares)")
#     axs[i].set_ylabel("Slippage ($)")
#     axs[i].grid(True)

# plt.suptitle("Temporary Impact Function gₜ(x): Slippage vs Volume", fontsize=16)
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load snapshot file
df = pd.read_csv("all_minute_order_book_snapshots.csv")

# Filter rows with valid ask prices/sizes
required_columns = [f'ask_px_0{i}' for i in range(10)] + [f'ask_sz_0{i}' for i in range(10)]
df = df.dropna(subset=required_columns).reset_index(drop=True)

# Sample N rows per stock
samples = df.groupby("stock").sample(n=3, random_state=42)

# Volume test range
volumes = np.arange(50, 1050, 50)

# Output directory
output_dir = "slippage_plots"
os.makedirs(output_dir, exist_ok=True)

# Slippage function
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

# Loop through samples and save individual plots
for i, (_, row) in enumerate(samples.iterrows()):
    ask_prices = [row[f'ask_px_0{j}'] for j in range(10)]
    ask_sizes = [row[f'ask_sz_0{j}'] for j in range(10)]
    bid = row.get('bid_px_00', ask_prices[0] - 0.05)
    ask = ask_prices[0]
    mid_price = (bid + ask) / 2

    slippages = [compute_slippage(x, ask_prices, ask_sizes, mid_price) for x in volumes]

    plt.figure(figsize=(8, 5))
    plt.plot(volumes, slippages, marker='o')
    plt.title(f"{row['stock']} @ {row['minute']} — Slippage vs Volume")
    plt.xlabel("Order Volume (shares)")
    plt.ylabel("Slippage ($)")
    plt.grid(True)

    filename = f"{row['stock']}_{str(row['minute']).replace(':','-')}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
