import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load or Simulate Data ---
def generate_fake_data(size=1000, seed=42):
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0005, scale=0.01, size=size)
    prices = 100 * (1 + returns).cumprod()
    return pd.Series(prices, name='Close')

data = generate_fake_data()
df = pd.DataFrame(data)

# --- Parameters ---
short_window = 50
long_window = 200
initial_cash = 100000

# --- Strategy Logic ---
df['SMA50'] = df['Close'].rolling(window=short_window).mean()
df['SMA200'] = df['Close'].rolling(window=long_window).mean()

df['Signal'] = 0
df['Signal'][short_window:] = np.where(
    df['SMA50'][short_window:] > df['SMA200'][short_window:], 1, 0
)
df['Position'] = df['Signal'].diff()

# --- Backtesting ---
df['Daily Return'] = df['Close'].pct_change()
df['Strategy Return'] = df['Signal'].shift(1) * df['Daily Return']

df['Cumulative Market'] = (1 + df['Daily Return']).cumprod()
df['Cumulative Strategy'] = (1 + df['Strategy Return']).cumprod()

# --- Performance Metrics ---
def calculate_sharpe(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

sharpe_ratio = calculate_sharpe(df['Strategy Return'].dropna())
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(df['Cumulative Market'], label='Market (Buy & Hold)', alpha=0.7)
plt.plot(df['Cumulative Strategy'], label='Moving Average Strategy', alpha=0.9)
plt.title('Moving Average Crossover Backtest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
