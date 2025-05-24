"""
Advanced Moving Average Crossover Research System

This system implements a comprehensive research framework for MA crossover strategies,
including walk-forward optimization, Monte Carlo simulation, and statistical testing.

Key Features:
1. Core strategy implementation
2. Walk-forward backtesting engine
3. Monte Carlo simulation
4. Parameter sensitivity analysis
5. Statistical significance testing
6. Machine learning integration hooks
7. Advanced visualization
8. Comprehensive reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]

class MACrossoverResearch:
    """
    Comprehensive research framework for moving average crossover strategies
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.best_params = {}
        
    # Data Methods ------------------------------------------------------------
    def generate_test_data(self, size=1000, trend=0.0005, volatility=0.01, 
                         seed=42, regimes=False) -> pd.DataFrame:
        """
        Generate synthetic price data with optional market regimes
        
        Args:
            size: Number of data points
            trend: Daily drift component
            volatility: Daily volatility
            seed: Random seed
            regimes: Whether to include regime changes
            
        Returns:
            DataFrame with synthetic price data
        """
        np.random.seed(seed)
        dates = pd.date_range(end=datetime.today(), periods=size)
        
        if regimes:
            # Create regime switching data
            regime_length = size // 4
            returns = np.concatenate([
                np.random.normal(loc=0.001, scale=0.005, size=regime_length),
                np.random.normal(loc=-0.0005, scale=0.02, size=regime_length),
                np.random.normal(loc=0.0002, scale=0.015, size=regime_length),
                np.random.normal(loc=0.0007, scale=0.008, size=size-3*regime_length)
            ])
        else:
            returns = np.random.normal(loc=trend, scale=volatility, size=size)
        
        prices = 100 * (1 + returns).cumprod()
        return pd.DataFrame({'Close': prices}, index=dates)
    
    def load_real_data(self, filepath: str) -> pd.DataFrame:
        """
        Load real market data from CSV file
        
        Args:
            filepath: Path to CSV file with columns=['Date', 'Close']
            
        Returns:
            DataFrame with price data
        """
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
            df = df[['Close']].dropna()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    # Core Strategy ----------------------------------------------------------
    def calculate_signals(self, data: pd.DataFrame, short_window: int, 
                        long_window: int) -> pd.DataFrame:
        """
        Calculate moving averages and trading signals
        
        Args:
            data: Price DataFrame
            short_window: Short MA period
            long_window: Long MA period
            
        Returns:
            DataFrame with signals
        """
        data = data.copy()
        data['SMA_Short'] = data['Close'].rolling(short_window).mean()
        data['SMA_Long'] = data['Close'].rolling(long_window).mean()
        
        # Generate signals
        data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, 0)
        data['Position'] = data['Signal'].diff()
        
        return data
    
    def backtest(self, data: pd.DataFrame, initial_capital=100000, 
                transaction_cost=0.0005) -> pd.DataFrame:
        """
        Run backtest on signal data
        
        Args:
            data: DataFrame with signals
            initial_capital: Starting capital
            transaction_cost: Percentage cost per transaction
            
        Returns:
            DataFrame with backtest results
        """
        if 'Signal' not in data.columns:
            raise ValueError("Data must contain signals")
            
        data = data.copy()
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Calculate strategy returns with transaction costs
        data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
        
        # Apply transaction costs
        trades = data['Position'].abs().sum()
        total_cost = trades * transaction_cost
        cost_per_trade = total_cost / trades if trades > 0 else 0
        
        data['Strategy_Return'] = np.where(
            data['Position'] != 0, 
            data['Strategy_Return'] - cost_per_trade, 
            data['Strategy_Return']
        )
        
        # Calculate portfolio values
        data['Cumulative_Market'] = (1 + data['Daily_Return']).cumprod()
        data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Strategy']
        
        return data
    
    # Research Methods -------------------------------------------------------
    def walk_forward_optimization(self, data: pd.DataFrame, 
                                short_range: Tuple[int, int], 
                                long_range: Tuple[int, int],
                                train_size: int = 252*2, 
                                test_size: int = 252,
                                step_size: int = 63) -> Dict:
        """
        Perform walk-forward optimization
        
        Args:
            data: Price data
            short_range: Tuple of (min, max) for short MA
            long_range: Tuple of (min, max) for long MA
            train_size: Training window size in days
            test_size: Testing window size in days
            step_size: Step size between windows
            
        Returns:
            Dictionary containing optimization results
        """
        results = []
        best_params = []
        
        # Generate parameter combinations
        short_params = range(short_range[0], short_range[1]+1, 5)
        long_params = range(long_range[0], long_range[1]+1, 5)
        param_combinations = [(s, l) for s in short_params 
                            for l in long_params if s < l]
        
        # Walk-forward windows
        n_windows = (len(data) - train_size - test_size) // step_size + 1
        
        for i in tqdm(range(n_windows), desc="Walk-forward optimization"):
            train_start = i * step_size
            train_end = train_start + train_size
            test_end = train_end + test_size
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Optimize on training period
            train_results = []
            for s, l in param_combinations:
                try:
                    train_signal = self.calculate_signals(train_data, s, l)
                    train_result = self.backtest(train_signal)
                    metrics = self.calculate_metrics(train_result['Strategy_Return'])
                    train_results.append({
                        'short': s, 
                        'long': l, 
                        'sharpe': metrics['sharpe_ratio']
                    })
                except:
                    continue
            
            if not train_results:
                continue
                
            # Find best params
            train_df = pd.DataFrame(train_results)
            best_param = train_df.loc[train_df['sharpe'].idxmax()]
            
            # Test on out-of-sample period
            test_signal = self.calculate_signals(test_data, best_param['short'], best_param['long'])
            test_result = self.backtest(test_signal)
            test_metrics = self.calculate_metrics(test_result['Strategy_Return'])
            
            results.append({
                'window': i,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_short': best_param['short'],
                'best_long': best_param['long'],
                'train_sharpe': best_param['sharpe'],
                'test_sharpe': test_metrics['sharpe_ratio'],
                'test_return': test_metrics['annualized_return']
            })
            
            best_params.append({
                'short': best_param['short'],
                'long': best_param['long'],
                'window': i
            })
        
        self.wfo_results = pd.DataFrame(results)
        self.best_params = pd.DataFrame(best_params)
        
        return {
            'results': self.wfo_results,
            'best_params': self.best_params,
            'summary_stats': self.wfo_results.describe()
        }
    
    def monte_carlo_simulation(self, data: pd.DataFrame, n_simulations=1000,
                             short_window=50, long_window=200) -> Dict:
        """
        Perform Monte Carlo simulation of strategy returns
        
        Args:
            data: Price data
            n_simulations: Number of simulations
            short_window: Short MA window
            long_window: Long MA window
            
        Returns:
            Dictionary containing simulation results
        """
        # Get actual strategy returns
        signal_data = self.calculate_signals(data, short_window, long_window)
        actual_result = self.backtest(signal_data)
        actual_returns = actual_result['Strategy_Return'].dropna()
        actual_sharpe = self.calculate_metrics(actual_returns)['sharpe_ratio']
        
        # Generate random paths
        random_sharpes = []
        daily_vol = data['Close'].pct_change().std()
        
        for _ in tqdm(range(n_simulations), desc="Monte Carlo Simulation"):
            # Generate random walk with similar properties
            random_returns = np.random.normal(
                loc=0, 
                scale=daily_vol, 
                size=len(actual_returns)
            )
            
            # Calculate random Sharpe
            random_sharpe = np.sqrt(252) * random_returns.mean() / random_returns.std()
            random_sharpes.append(random_sharpe)
        
        # Calculate p-value
        p_value = (np.array(random_sharpes) > actual_sharpe).mean()
        
        self.mc_results = {
            'actual_sharpe': actual_sharpe,
            'random_sharpes': random_sharpes,
            'p_value': p_value,
            'significance_level': 0.05
        }
        
        return self.mc_results
    
    def parameter_sensitivity(self, data: pd.DataFrame, 
                            short_range: Tuple[int, int], 
                            long_range: Tuple[int, int]) -> pd.DataFrame:
        """
        Analyze parameter sensitivity across ranges
        
        Args:
            data: Price data
            short_range: Tuple of (min, max) for short MA
            long_range: Tuple of (min, max) for long MA
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        short_params = range(short_range[0], short_range[1]+1, 5)
        long_params = range(long_range[0], long_range[1]+1, 5)
        
        for s in tqdm(short_params, desc="Parameter sensitivity"):
            for l in long_params:
                if s >= l:
                    continue
                    
                try:
                    signal_data = self.calculate_signals(data, s, l)
                    result = self.backtest(signal_data)
                    metrics = self.calculate_metrics(result['Strategy_Return'])
                    
                    results.append({
                        'short_window': s,
                        'long_window': l,
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'annual_return': metrics['annualized_return'],
                        'max_drawdown': metrics['max_drawdown']
                    })
                except:
                    continue
        
        self.sensitivity_results = pd.DataFrame(results)
        return self.sensitivity_results
    
    # Machine Learning Integration -------------------------------------------
    def create_feature_matrix(self, data: pd.DataFrame, 
                            lookback=30) -> pd.DataFrame:
        """
        Create feature matrix for ML models
        
        Args:
            data: Price data
            lookback: Lookback window for features
            
        Returns:
            DataFrame with features and target
        """
        df = data.copy()
        
        # Technical features
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(lookback).std()
        df['Momentum'] = df['Close'] / df['Close'].shift(lookback) - 1
        
        # Moving average features
        for w in [5, 10, 20, 50]:
            df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
            df[f'Ratio_{w}'] = df['Close'] / df[f'SMA_{w}']
        
        # Target variable (next day's return)
        df['Target'] = df['Returns'].shift(-1)
        
        return df.dropna()
    
    def train_hybrid_model(self, data: pd.DataFrame):
        """
        Train a hybrid model combining rules and ML
        
        Args:
            data: Feature matrix from create_feature_matrix()
            
        Returns:
            Trained model and evaluation metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
        X = data.drop(columns=['Target'])
        y = data['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        
        self.ml_model = model
        self.ml_results = {
            'feature_importances': pd.Series(
                model.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False),
            'test_mse': mse
        }
        
        return self.ml_results
    
    # Performance Analysis ---------------------------------------------------
    def calculate_metrics(self, returns: pd.Series, 
                        risk_free_rate=0.0) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Series of strategy returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        if len(returns) < 5:
            raise ValueError("Insufficient return data")
            
        returns = returns.dropna()
        excess_returns = returns - risk_free_rate / 252
        
        # Basic metrics
        total_return = returns.add(1).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        
        # Win/loss metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean()
        avg_loss = returns[returns <= 0].mean()
        profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.inf
        
        # Risk-adjusted metrics
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / returns[returns < 0].std()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Statistical tests
        _, normality_p = stats.shapiro(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'normality_p': normality_p,
            'positive_skew': returns.skew() > 0
        }
    
    def compare_benchmark(self, strategy_returns: pd.Series, 
                        benchmark_returns: pd.Series) -> Dict:
        """
        Compare strategy vs benchmark
        
        Args:
            strategy_returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate alpha/beta
        X = benchmark_returns.values.reshape(-1, 1)
        y = strategy_returns.values
        model = LinearRegression().fit(X, y)
        alpha = model.intercept_ * 252
        beta = model.coef_[0]
        
        # Calculate outperformance
        excess = strategy_returns - benchmark_returns
        outperformance = (1 + excess).prod() - 1
        
        return {
            'alpha': alpha,
            'beta': beta,
            'outperformance': outperformance,
            'correlation': strategy_returns.corr(benchmark_returns)
        }
    
    # Visualization Methods --------------------------------------------------
    def plot_equity_curve(self, data: pd.DataFrame) -> None:
        """
        Plot strategy vs market equity curve
        
        Args:
            data: DataFrame containing backtest results
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data['Cumulative_Market'], label='Buy & Hold', alpha=0.7)
        plt.plot(data['Cumulative_Strategy'], label='Strategy', alpha=0.9)
        
        # Mark trades
        buys = data[data['Position'] == 1]
        sells = data[data['Position'] == -1]
        plt.scatter(buys.index, buys['Cumulative_Strategy'], 
                   marker='^', color='g', label='Buy')
        plt.scatter(sells.index, sells['Cumulative_Strategy'], 
                   marker='v', color='r', label='Sell')
        
        plt.title('Strategy vs Market Performance')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_parameter_sensitivity(self, results: pd.DataFrame, 
                                 metric='sharpe_ratio') -> None:
        """
        Plot parameter sensitivity heatmap
        
        Args:
            results: DataFrame from parameter_sensitivity()
            metric: Metric to visualize
        """
        pivot = results.pivot(index='short_window', 
                             columns='long_window', 
                             values=metric)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='viridis')
        plt.title(f'Parameter Sensitivity: {metric.replace("_", " ").title()}')
        plt.xlabel('Long Window')
        plt.ylabel('Short Window')
        plt.show()
    
    def plot_monte_carlo(self, results: Dict) -> None:
        """
        Plot Monte Carlo simulation results
        
        Args:
            results: Dictionary from monte_carlo_simulation()
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(results['random_sharpes'], bins=30, kde=True)
        plt.axvline(results['actual_sharpe'], color='r', 
                   linestyle='--', label='Actual Strategy')
        plt.title(f'Monte Carlo Simulation (p-value: {results["p_value"]:.3f})')
        plt.xlabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_walk_forward(self, results: pd.DataFrame) -> None:
        """
        Plot walk-forward optimization results
        
        Args:
            results: DataFrame from walk_forward_optimization()
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Parameter stability plot
        results['param_combo'] = results['best_short'].astype(str) + '/' + results['best_long'].astype(str)
        results['param_combo'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Parameter Stability Across Windows')
        ax1.set_ylabel('Frequency')
        
        # Performance consistency plot
        results[['train_sharpe', 'test_sharpe']].plot(ax=ax2)
        ax2.axhline(results['test_sharpe'].mean(), color='r', linestyle='--')
        ax2.set_title('Train vs Test Sharpe Ratios')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xlabel('Window')
        
        plt.tight_layout()
        plt.show()

# Example Usage -------------------------------------------------------------
if __name__ == "__main__":
    # Initialize research system
    research = MACrossoverResearch()
    
    # Generate or load data
    data = research.generate_test_data(size=2000, regimes=True)
    # data = research.load_real_data('your_data.csv')
    
    # Core strategy analysis
    signal_data = research.calculate_signals(data, 50, 200)
    backtest_data = research.backtest(signal_data)
    metrics = research.calculate_metrics(backtest_data['Strategy_Return'])
    
    print("\nCore Strategy Metrics:")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<25}: {v:.4f}")
    
    # Advanced research
    wfo_results = research.walk_forward_optimization(
        data, short_range=(20, 100), long_range=(100, 300))
    
    mc_results = research.monte_carlo_simulation(data, n_simulations=1000)
    
    sensitivity = research.parameter_sensitivity(
        data, short_range=(10, 80), long_range=(100, 300))
    
    # Visualization
    research.plot_equity_curve(backtest_data)
    research.plot_parameter_sensitivity(sensitivity)
    research.plot_monte_carlo(mc_results)
    research.plot_walk_forward(wfo_results['results'])
    
    # Machine learning integration
    feature_data = research.create_feature_matrix(data)
    ml_results = research.train_hybrid_model(feature_data)
    
    print("\nFeature Importances:")
    print(ml_results['feature_importances'].head(10))
