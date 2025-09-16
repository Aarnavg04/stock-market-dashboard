# Stock Market Analytics Dashboard


## Overview

A sophisticated financial analytics dashboard that leverages **Monte Carlo simulations** with **GARCH(1,1) volatility modeling** to forecast stock price distributions, particularly around earnings announcements. The platform processes 5 years of S&P 500 historical data and provides comprehensive risk analytics with interactive visualizations.

### Key Features

- ** 10,000 Monte Carlo Simulations** - Geometric Brownian Motion with stochastic volatility
- ** GARCH(1,1) Volatility Modeling** - Captures volatility clustering and mean reversion
- ** Event-Driven Analysis** - Special handling for earnings announcements with 73% accuracy
- ** 100k+ Historical Data Points** - 5 years of S&P 500 data with SQLite storage
- ** Advanced Risk Metrics** - VaR, CVaR, Sharpe Ratio, probability distributions
- ** Interactive Visualizations** - Real-time Plotly Dash interface with dark theme


##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum (4GB recommended for 10,000 simulations)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-market-analytics-dashboard.git
cd stock-market-analytics-dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize database**
```bash
python initialize_data.py
```

5. **Run the dashboard**
```bash
python main.py
```

6. **Open browser**
Navigate to `http://localhost:8050`

## 📊 Technical Architecture

### Monte Carlo Simulation Engine
```python
# Geometric Brownian Motion with GARCH volatility
S(t+1) = S(t) × exp(μ*dt + σ(t)*√dt*Z)

Where:
- μ = Drift (historical mean return)
- σ(t) = Time-varying volatility from GARCH
- Z ~ N(0,1) = Random shock
```

### GARCH(1,1) Model
```python
# Variance equation
σ²(t) = ω + α×ε²(t-1) + β×σ²(t-1)

Parameters:
- α = Impact of recent shocks (avg: 0.07)
- β = Volatility persistence (avg: 0.92)
- ω = Long-term variance baseline
```

### Risk Metrics

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **VaR (95%)** | Maximum loss in 95% of scenarios | 5th percentile of return distribution |
| **CVaR** | Expected loss beyond VaR | Mean of returns below VaR |
| **Sharpe Ratio** | Risk-adjusted return | (μ - rf) / σ |
| **Probability of Profit** | Chance of positive return | P(return > 0) |

## 🏗️ Project Structure

```
stock-market-analytics-dashboard/
│
├── src/                      # Source code
│   ├── dashboard.py         # Main Dash application
│   ├── monte_carlo.py       # Monte Carlo simulation engine
│   ├── garch_model.py       # GARCH volatility modeling
│   ├── risk_metrics.py      # Risk calculations (VaR, CVaR)
│   ├── data_fetcher.py      # Yahoo Finance data pipeline
│   └── backtest.py          # Backtesting framework
│
├── data/                    # Data storage
│   └── stock_data.db       # SQLite database (100k+ records)
│
├── tests/                   # Unit tests
│   └── test_simulation.py
│
├── docs/                    # Documentation
│   └── technical_spec.md
│
├── requirements.txt         # Python dependencies
├── main.py                 # Entry point
└── README.md              # This file
```

## 🔬 Performance Metrics

### Simulation Performance
- **10,000 paths**: ~3-5 seconds
- **Memory usage**: ~500MB peak
- **Database queries**: <100ms

### Model Accuracy
- **Post-earnings volatility**: 73% accuracy
- **Backtested on**: 100+ earnings events
- **Validation period**: 2019-2024

## 🛠️ Configuration

### Adjust Simulation Parameters

```python
# In src/monte_carlo.py
simulator = MonteCarloSimulator(
    n_simulations=10000,  # Number of paths
    random_seed=42        # For reproducibility
)
```

### Modify GARCH Settings

```python
# In src/garch_model.py
model = GARCHVolatilityModel(
    p=1,  # GARCH lag order
    q=1   # ARCH lag order
)
```

## 📈 Usage Examples

### Basic Simulation
```python
from src.monte_carlo import MonteCarloSimulator
from src.data_fetcher import StockDataFetcher

# Fetch data
fetcher = StockDataFetcher()
prices = fetcher.fetch_historical_data("AAPL", period="2y")

# Run simulation
simulator = MonteCarloSimulator(n_simulations=10000)
paths = simulator.simulate_gbm_garch(prices['Close'], days_ahead=30)
```

### Calculate Risk Metrics
```python
from src.risk_metrics import RiskAnalyzer

analyzer = RiskAnalyzer()
metrics = analyzer.calculate_metrics(paths, initial_price=150.0)

print(f"VaR (95%): {metrics['var_95']:.2f}%")
print(f"Expected Return: {metrics['expected_return']:.2f}%")
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Performance benchmark:
```bash
python tests/benchmark.py
```

## 📝 API Documentation

### Dashboard Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard interface |
| `/api/simulate` | POST | Run simulation (JSON) |
| `/api/metrics` | GET | Get current metrics |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📊 Future Enhancements

- [ ] Options pricing with Black-Scholes
- [ ] Portfolio optimization
- [ ] Real-time data streaming
- [ ] Machine learning predictions
- [ ] Multi-asset correlation analysis
- [ ] Backtesting framework expansion

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Yahoo Finance** for historical data
- **Plotly Dash** for visualization framework
- **ARCH** library for GARCH modeling
- **NumPy/Pandas** for numerical computing

##  Contact

Aarnav Gandhi - aarnavgandhi04@gmail.com
