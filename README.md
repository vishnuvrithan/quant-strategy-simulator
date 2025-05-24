# 📈 Quant Strategy Simulator

This is a lightweight Python-based backtester for a moving average crossover strategy.

---

## 💡 Features

- Fast backtesting using Pandas and NumPy
- Long-only strategy: Buy when SMA50 > SMA200
- Sharpe Ratio calculation
- Optimized from 45s → ~3s for large datasets
- Plot of market vs strategy performance

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2.Run the script:
python main.py


Strategy Logic
If 50-day SMA > 200-day SMA → Go long

Else → Exit to cash

No leverage, no fees assumed


