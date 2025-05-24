"""
Moving Average Crossover Trading Strategy Backtest

This script implements and backtests a simple moving average crossover strategy:
- Buy when the short-term MA crosses above the long-term MA
- Sell (go to cash) when the short-term MA crosses below the long-term MA

Features:
- Modular design for easy extension
- Comprehensive performance metrics
- Visualization of results
- Parameter optimization capability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Implements a moving average crossover trading strategy with backtesting capabilities
    """
    
    def __init__(self, short_window: int = 50, long_window: int = 200):
        """
        Initialize the strategy with MA windows
        
        Args:
            short_window: Days for short moving average
            long_window: Days for long moving average
        """
        self.short_window = short_window
        self.long_window = long_window
        self.data = None
        
    def generate_test_data(self, size: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic price data for testing
        
        Args:
            size: Number of data points to generate
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic price data
        """
        np.random.seed(seed)
        returns = np.random.normal(loc=0.0005, scale=0.01, size=size)
        prices = 100 * (1 + returns).cumprod()
        return pd.DataFrame(prices, columns=['Close'])
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving averages and generate signals
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added indicator columns
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
            
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
            
        data = data.copy()
        data['SMA50'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA200'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals (1 = long, 0 = cash)
        data['Signal'] = 0
        data.loc[self.long_window:, 'Signal'] = np.where(
            data['SMA50'][self.long_window:] > data['SMA200'][self.long_window:], 1, 0
        )
        
        # Position changes (1 = buy, -1 = sell, 0 = hold)
        data['Position'] = data['Signal'].diff()
        
        return data
    
    def backtest_strategy(self, data: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
        """
        Run backtest on the strategy
        
        Args:
            data: DataFrame with price data and signals
            initial_capital: Starting portfolio value
            
        Returns:
            DataFrame with backtest results
        """
        if 'Signal' not in data.columns:
            raise ValueError("Data must contain signals. Run calculate_moving_averages() first.")
            
        data = data.copy()
        
        # Calculate returns
        data['Daily Return'] = data['Close'].pct_change()
        data['Strategy Return'] = data['Signal'].shift(1) * data['Daily Return']
        
        # Calculate cumulative returns
        data['Cumulative Market'] = (1 + data['Daily Return']).cumprod()
        data['Cumulative Strategy'] = (1 + data['Strategy Return']).cumprod()
        
        # Calculate portfolio values
        data['Portfolio Value'] = initial_capital * data['Cumulative Strategy']
        
        return data
    
    def calculate_performance_metrics(self, returns: pd.Series, risk_free_rate: float = 0.0) -> Dict:
        """
        Calculate key performance metrics
        
        Args:
            returns: Series of strategy returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        if returns.isnull().all():
            raise ValueError("Return series contains no valid data")
            
        returns = returns.dropna()
        excess_returns = returns - risk_free_rate / 252
        
        # Basic metrics
        total_return = returns.add(1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).mean()
        }
    
    def plot_results(self, data: pd.DataFrame) -> None:
        """
        Visualize backtest results
        
        Args:
            data: DataFrame containing backtest results
        """
        if 'Cumulative Strategy' not in data.columns:
            raise ValueError("Data must contain backtest results. Run backtest_strategy() first.")
            
        plt.figure(figsize=(12, 8))
        
        # Price and moving averages
        plt.subplot(2, 1, 1)
        plt.plot(data['Close'], label='Price', alpha=0.7)
        plt.plot(data['SMA50'], label=f'SMA {self.short_window}', alpha=0.7)
        plt.plot(data['SMA200'], label=f'SMA {self.long_window}', alpha=0.7)
        
        # Plot buy/sell signals
        buy_signals = data[data['Position'] == 1]
        sell_signals = data[data['Position'] == -1]
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color='g', label='Buy', alpha=1)
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color='r', label='Sell', alpha=1)
        
        plt.title('Price and Moving Averages')
        plt.legend()
        plt.grid(True)
        
        # Cumulative returns
        plt.subplot(2, 1, 2)
        plt.plot(data['Cumulative Market'], label='Buy & Hold', alpha=0.7)
        plt.plot(data['Cumulative Strategy'], label='Strategy', alpha=0.9)
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    try:
        # Initialize strategy
        strategy = TradingStrategy(short_window=50, long_window=200)
        
        # Generate or load data
        data = strategy.generate_test_data(size=1000)
        logger.info("Generated synthetic price data")
        
        # Calculate indicators and signals
        data = strategy.calculate_moving_averages(data)
        logger.info("Calculated moving averages and signals")
        
        # Run backtest
        data = strategy.backtest_strategy(data)
        logger.info("Completed strategy backtest")
        
        # Calculate performance metrics
        metrics = strategy.calculate_performance_metrics(data['Strategy Return'].dropna())
        logger.info("Calculated performance metrics")
        
        # Display results
        print("\nPerformance Metrics:")
        for k, v in metrics.items():
            print(f"{k.replace('_', ' ').title():<20}: {v:.4f}")
        
        # Plot results
        strategy.plot_results(data)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
