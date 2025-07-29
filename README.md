# Market Impact Modeling

This repository contains code and analysis for modeling the **temporary impact function** \( g_t(x) \), which represents slippage when executing market orders of size \( x \) at time \( t \).

We evaluate whether a **linear** model \( g_t(x) = \beta x \) is sufficient, or if a **quadratic** model \( g_t(x) = ax^2 + bx \) better fits real market conditions.

---

## 📂 Project Structure

- `scripts/`
  - `reconstruct_order_book_snapshots.py`: Processes raw tick-level data into minute-wise snapshots
  - `compute_slippage_strategy.py`: Simulates a greedy execution strategy
  - `fit_slippage_models.py`: Fits linear and quadratic models to slippage data
- `notebooks/`
  - `Slippage_Modeling_Analysis.ipynb`: End-to-end exploration and visualization
- `plots/`: Slippage vs. volume plots with fitted curves
- `requirements.txt`: Python dependencies

---

## 🔍 Method Summary

- Extracts top 10 ask prices and sizes from each snapshot
- Computes slippage at varying buy sizes
- Fits both linear and quadratic models
- Evaluates model quality using R² score

---

## 📈 Key Findings

- **FROG** and **CRWV** exhibit strongly convex impact curves → quadratic model preferred
- **SOUN** often shows flat curves due to high liquidity at top price level
- Quadratic model achieves significantly higher R² in most cases

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/ishi0801/market-impact-modeling.git
cd market-impact-modeling

# Install dependencies
pip install -r requirements.txt

# Run analysis
python scripts/fit_slippage_models.py
