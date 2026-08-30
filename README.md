# 🤖 IBKR AI Trading Bot & Penny Stock Scanner

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Interactive Brokers](https://img.shields.io/badge/broker-Interactive%20Brokers-red.svg)](https://www.interactivebrokers.com/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-orange.svg)](https://xgboost.readthedocs.io/)
[![NLP Sentiment](https://img.shields.io/badge/NLP-FinBERT%20%28HuggingFace%29-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, machine-learning-driven quantitative trading bot and multi-factor **Penny Stock Scanner** designed specifically for **Interactive Brokers (TWS / IB Gateway)**.

The system combines historical price data, **50+ engineered technical indicators**, and **FinBERT NLP news sentiment analysis** with **XGBoost / LightGBM** models to forecast directional price movements. It features walk-forward backtesting with realistic slippage and tiered commission modeling, dynamic ATR risk controls, and automated order execution in paper and live environments via `ib_insync`.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Interactive Brokers (TWS / Gateway) Configuration](#-interactive-brokers-tws--gateway-configuration)
- [Configuration Reference (`config.yaml`)](#-configuration-reference-configyaml)
- [CLI Usage Guide](#-cli-usage-guide)
  - [1. View Bot Information & Settings](#1-view-bot-information--settings)
  - [2. Model Training & Hyperparameter Tuning](#2-model-training--hyperparameter-tuning)
  - [3. Walk-Forward Backtesting](#3-walk-forward-backtesting)
  - [4. Scanning Penny Stocks](#4-scanning-penny-stocks)
  - [5. Automated Paper / Live Trading](#5-automated-paper--live-trading)
- [Feature Engineering & Indicators](#-feature-engineering--indicators)
- [News Sentiment (FinBERT)](#-news-sentiment-finbert)
- [Multi-Factor Penny Stock Scanner](#-multi-factor-penny-stock-scanner)
- [Risk Management & Order Controls](#-risk-management--order-controls)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [Disclaimer](#-disclaimer)

---

## 🏛 System Architecture

```text
                     ┌──────────────────────────────────────┐
                     │          Market Data Sources         │
                     │   (yfinance / IBKR Market Data)      │
                     └──────────────────┬───────────────────┘
                                        │ OHLCV
                                        ▼
┌───────────────────────┐    ┌───────────────────────┐
│ NewsAPI / Web Scraper │    │     Feature Engine    │
└──────────┬────────────┘    │  - 50+ Indicators     │
           │ Headlines       │  - Moving Averages    │
           ▼                 │  - Volatility / ATR   │
┌───────────────────────┐    │  - Volume / Momentum  │
│   FinBERT Sentiment   │───▶│  - Support / Resist   │
│   (Positive/Neg/Neu)  │    └──────────┬────────────┘
└───────────────────────┘               │
                                        │ Feature Matrix
                                        ▼
                             ┌───────────────────────┐
                             │    ML Model Engine    │
                             │  (XGBoost / LightGBM) │
                             │  - Confidence Scoring │
                             │  - Optuna Tuning      │
                             └──────────┬────────────┘
                                        │
                                        │ Buy / Sell / Hold Signal
                                        ▼
                             ┌───────────────────────┐
                             │     Risk Manager      │
                             │  - Position Sizing    │
                             │  - ATR Stop Loss / TP │
                             │  - Trailing Stop      │
                             │  - Daily Drawdown Max │
                             └──────────┬────────────┘
                                        │
                                        │ Sized & Validated Orders
                                        ▼
                 ┌─────────────────────────────────────────────┐
                 │                Execution Layer              │
                 │  ┌──────────────────┐ ┌───────────────────┐ │
                 │  │ Backtest Engine  │ │  IBKR API Client  │ │
                 │  │ (Tiered Comm /   │ │  (ib_insync TWS / │ │
                 │  │  Slippage / Logs)│ │   Gateway Engine) │ │
                 │  └──────────────────┘ └───────────────────┘ │
                 └─────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Machine Learning Forecasting**: Predicts forward returns using XGBoost or LightGBM with customizable confidence thresholds (e.g. $\ge 0.60$) and walk-forward cross-validation.
- **50+ Engineered Technical Indicators**: Computes Trend (EMA, SMA, MACD, ADX), Momentum (RSI, Stochastic, Williams %R), Volatility (Bollinger Bands, ATR), Volume (OBV, VWAP, CMF, Volume Surge), and Price-Action features.
- **FinBERT NLP Sentiment Analysis**: Integrates HuggingFace's `ProsusAI/finbert` to extract sentiment scores from financial headlines and incorporate real-time market sentiment into ML features.
- **Multi-Factor Penny Stock Scanner**: Automated scanner for low-float, high-momentum small caps ($0.50–$10.00), scored across volume surges, momentum, technical posture, sentiment, and relative strength vs. SPY.
- **Institutional Risk Controls**:
  - Position sizing based on portfolio equity risk percent.
  - ATR-based dynamic Stop-Loss and Take-Profit brackets.
  - Trailing stop protection with high-watermark tracking.
  - Circuit breaker / daily portfolio loss kill-switch.
  - Maximum open position caps to prevent overexposure.
- **Realistic Backtesting Engine**: Models IBKR tiered commissions ($0.005/share) and slippage basis points, logging full trade histories, Sharpe Ratio, Profit Factor, Win Rate, and Max Drawdown.
- **TWS / IB Gateway Integration**: Seamless connection via `ib_insync` for live and paper trading.

---

## 📁 Project Structure

```text
ibkr_trading_bot/
├── config.yaml          # Central configuration for strategy, risk, IBKR & scanner
├── .env.example         # Template for environment variables and API keys
├── requirements.txt     # Complete Python dependencies
├── main.py              # CLI entry point for all workflows
│
├── data_loader.py       # Market price data fetcher, caching & news collector
├── sentiment.py         # FinBERT NLP transformer sentiment analyzer
├── feature_engine.py    # Computes 50+ technical and statistical indicators
├── ml_model.py          # XGBoost / LightGBM model trainer, validator & predictor
├── penny_scanner.py     # Multi-factor penny stock scanner and scoring engine
├── backtester.py        # Walk-forward backtester with commission & slippage modeling
├── risk_manager.py      # Capital protection, position sizing & kill-switch logic
├── ibkr_client.py       # Interactive Brokers API wrapper (ib_insync)
├── bot.py               # Main asynchronous trading loop & execution orchestrator
├── debug_features.py    # Utility script to inspect engineered features
│
├── models/              # Serialized ML models (.joblib) and metadata (.json)
├── data/                # Cached historical price & news data
├── logs/                # System execution logs (bot.log, trades.csv)
└── reports/             # Generated backtest summaries and penny scanner CSVs
```

---

## 📋 Prerequisites

1. **Python**: Python 3.10 or higher.
2. **Interactive Brokers Account**: Active IBKR account (paper trading accounts are free and supported).
3. **Trader Workstation (TWS)** or **IB Gateway**: Installed and configured on your host machine.
4. **NewsAPI Key (Optional)**: Free tier API key from [newsapi.org](https://newsapi.org) for financial news sentiment.

---

## 🚀 Installation & Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/your-username/ibkr_trading_bot.git
cd ibkr_trading_bot
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
# NewsAPI Key (Get a free key from https://newsapi.org)
NEWS_API_KEY=your_news_api_key_here

# Interactive Brokers Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Optional: Telegram Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

---

## 🔌 Interactive Brokers (TWS / Gateway) Configuration

To allow the bot to communicate with Interactive Brokers, ensure the following API settings are enabled in TWS or IB Gateway:

1. Launch **Trader Workstation (TWS)** or **IB Gateway**.
2. Navigate to **Edit** $\rightarrow$ **Global Configuration** $\rightarrow$ **API** $\rightarrow$ **Settings**.
3. Configure the following checkboxes:
   - ✅ **Enable ActiveX and Socket Clients**
   - ❌ **Uncheck "Read-Only API"** (Required for the bot to place orders)
   - ✅ Set **Socket Port**:
     - `7497` for **TWS Paper Trading**
     - `7496` for **TWS Live Trading**
     - `4002` for **IB Gateway Paper Trading**
     - `4001` for **IB Gateway Live Trading**
   - ✅ Add `127.0.0.1` under **Trusted IP Addresses** (prevents repetitive confirmation popups)
4. Click **Apply** and **OK**.

> [!TIP]
> Always verify that TWS or IB Gateway is running and logged in before starting `python main.py paper-trade`.

---

## ⚙️ Configuration Reference (`config.yaml`)

All parameters can be tuned in `config.yaml` without changing application code:

```yaml
# --- IBKR Connection ---
ibkr:
  host: "127.0.0.1"
  port: 7497                  # 7497 = TWS Paper, 7496 = TWS Live
  client_id: 1
  account: ""                 # Leave blank to auto-detect

# --- Core Watchlist ---
watchlist:
  symbols:
    - SPY
    - AAPL
    - MSFT
    - NVDA
    - TSLA

# --- Penny Stock Scanner ---
penny_scanner:
  enabled: true
  min_price: 0.50             # Minimum stock price ($)
  max_price: 10.00            # Maximum stock price ($)
  min_avg_volume: 500000      # 20-day average volume threshold
  min_market_cap: 10000000    # $10M min market cap
  max_market_cap: 500000000   # $500M max market cap
  exchanges:
    - NYSE
    - NASDAQ
    - AMEX
  top_n_picks: 10
  weights:                    # Multi-factor scoring weights (sum to 1.0)
    volume_surge: 0.25
    momentum: 0.20
    technical: 0.20
    sentiment: 0.20
    relative_strength: 0.15

# --- ML Strategy ---
strategy:
  timeframe: "1d"              # Bar interval: 1d, 1h, 15m, 5m
  lookback_years: 5            # Historical training data window
  label_horizon_days: 5        # Forward return window for labels
  label_threshold_pct: 1.5     # Return threshold for BUY (1) / SELL (-1) / HOLD (0)
  confidence_threshold: 0.60   # Minimum ML probability to trigger an entry
  model_type: "xgboost"        # "xgboost" or "lightgbm"
  retrain_interval_days: 30

# --- Risk Management ---
risk:
  max_risk_per_trade_pct: 1.5  # Max account equity % risked per trade
  max_position_pct: 5.0        # Max portfolio weight for a single stock
  max_daily_loss_pct: 3.0      # Daily drawdown kill-switch (% account loss)
  max_open_positions: 10       # Maximum concurrent positions
  stop_loss_atr_mult: 2.0      # ATR multiple for stop-loss distance
  take_profit_atr_mult: 3.0    # ATR multiple for take-profit target
  trailing_stop_pct: 2.0       # Trailing stop trigger (% below high watermark)

# --- NLP Sentiment ---
sentiment:
  provider: "newsapi"          # "newsapi" or web scraper
  model: "ProsusAI/finbert"    # HuggingFace NLP model
  lookback_days: 7             # News window to score
  min_articles: 2

# --- Backtesting ---
backtest:
  initial_capital: 100000.0
  commission_per_share: 0.005  # IBKR Tiered commission per share
  slippage_pct: 0.05           # 5 bps slippage model
```

---

## 💻 CLI Usage Guide

The bot provides an intuitive Click-based CLI via `main.py`.

### 1. View Bot Information & Settings
Check current settings, IBKR connection parameters, risk limits, and model training status:
```bash
python main.py info
```

---

### 2. Model Training & Hyperparameter Tuning

Train the ML model on historical price action and sentiment:

```bash
# Train on default watchlist defined in config.yaml
python main.py train

# Train on custom symbols with 3 years of history
python main.py train --symbols AAPL,MSFT,NVDA,TSLA --years 3

# Train with Optuna hyperparameter optimization (e.g. 50 trials)
python main.py train --tune --tune-trials 50
```

> **Outputs**: Saves the trained pipeline to `models/trading_model.joblib` and feature metadata to `models/trading_model_meta.json`.

---

### 3. Walk-Forward Backtesting

Simulate trading strategies on historical data with commission and slippage modeling:

```bash
# Backtest on SPY for 1 year (daily timeframe)
python main.py backtest --symbol SPY --period 1y

# Backtest on AAPL with custom initial capital ($50,000)
python main.py backtest --symbol AAPL --period 2y --capital 50000

# Backtest on recent intraday data (e.g. last 7 days on 15-minute bars)
python main.py backtest --symbol NVDA --days 7 --interval 15m
```

#### Backtest Metrics Generated:
- **Total Trades & Win Rate (%)**
- **Profit Factor & Expectancy**
- **Sharpe & Sortino Ratios**
- **Maximum Drawdown (%) & Peak Equity**
- **Detailed trade log in `reports/`**

---

### 4. Scanning Penny Stocks

Run the multi-factor scanner to discover high-conviction breakout penny stocks:

```bash
# Scan today's top 10 penny stock picks
python main.py scan-pennies

# Scan top 25 picks
python main.py scan-pennies --top 25
```

> **Outputs**: Generates a ranked terminal report and exports results to `reports/penny_picks_YYYYMMDD.csv`.

---

### 5. Automated Paper / Live Trading

Start the continuous trading loop (requires TWS or IB Gateway active):

```bash
# Start paper trading on port 7497
python main.py paper-trade
```

The trading loop will:
1. Connect to IBKR and sync account equity and positions.
2. Monitor real-time prices and generate features.
3. Query the FinBERT sentiment engine for relevant news.
4. Generate ML signals and filter by confidence threshold.
5. Check risk manager constraints (daily loss limits, position caps).
6. Dispatch bracket orders (Entry + Stop Loss + Take Profit) to IBKR.
7. Continuously monitor trailing stops and open positions.

---

## 📈 Feature Engineering & Indicators

The `FeatureEngine` transforms raw OHLCV price series into 50+ features:

| Category | Indicators & Features |
|:---|:---|
| **Trend** | EMA (9, 21, 50, 200), SMA (20, 50), MACD & Signal line, MACD Histogram, ADX (+DI, -DI) |
| **Momentum** | RSI (14), Stochastic Oscillator (%K, %D), Williams %R (14), Rate of Change (ROC) |
| **Volatility** | Average True Range (ATR 14, ATR %), Bollinger Bands (Upper, Lower, Width, %B) |
| **Volume** | On-Balance Volume (OBV), Chaikin Money Flow (CMF), Volume SMA ratio, Volume Surge Multiplier |
| **Price Action** | High/Low ranges, Log returns, Multi-day momentum (1d, 3d, 5d, 10d), Support/Resistance distance |
| **Sentiment** | Mean sentiment score, Positive/Negative headline ratios, Sentiment momentum |

---

## 📰 News Sentiment (FinBERT)

The `SentimentAnalyzer` uses HuggingFace’s pre-trained `ProsusAI/finbert` model (a specialized BERT language model fine-tuned on financial text):

- Fetches headlines and descriptions via NewsAPI or fallback scrapers.
- Computes softmax probabilities for `positive`, `negative`, and `neutral`.
- Converts scores into a scalar $(-1.0 \text{ to } +1.0)$ composite sentiment index:
  $$\text{Score} = P(\text{positive}) - P(\text{negative})$$
- Aggregates article volume, polarity dispersion, and sentiment shifts over 7-day lookback windows.

---

## 🎯 Multi-Factor Penny Stock Scanner

The `PennyScanner` scans thousands of active US equities to identify high-probability breakout penny stocks using weighted factors:

1. **Volume Surge ($25\%$)**: Ratio of current volume to 20-day average volume.
2. **Price Momentum ($20\%$)**: 5-day and 10-day price momentum.
3. **Technical Setup ($20\%$)**: RSI momentum, MACD bullish crossovers, and moving average alignment.
4. **News Sentiment ($20\%$)**: FinBERT sentiment score from recent financial headlines.
5. **Relative Strength ($15\%$)**: Stock return relative to the SPY benchmark.

*Optional Overlay*: If a trained ML model exists, it computes an ML confidence score for each candidate.

---

## 🛡️ Risk Management & Order Controls

The `RiskManager` strictly regulates all order generation:

- **Position Sizing Formula**:
  $$\text{Shares} = \min\left(\frac{\text{Equity} \times \text{RiskPerTrade}\%}{\text{ATR} \times \text{StopLossMult}}, \frac{\text{Equity} \times \text{MaxPosition}\%}{\text{EntryPrice}}\right)$$
- **ATR Dynamic Stops**: Stop-loss and take-profit levels adapt dynamically to current market volatility.
- **Trailing Stop Loss**: Tracks peak price since entry and adjusts stops upwards to protect profits.
- **Daily Drawdown Circuit Breaker**: If daily losses exceed `max_daily_loss_pct` (e.g. 3%), trading is automatically halted for the day.
- **Portfolio Heat Limiter**: Enforces `max_open_positions` to prevent overconcentration.

---

## 🔧 Troubleshooting & FAQ

### 1. `ConnectionRefusedError` / `ib_insync.wrapper: Connection refused`
- Make sure TWS or IB Gateway is running and logged in.
- In TWS, verify that **"Enable ActiveX and Socket Clients"** is checked.
- Ensure the port in `config.yaml` matches your TWS setting (`7497` for TWS Paper, `7496` for TWS Live).
- Verify that `127.0.0.1` is added to Trusted IP addresses.

### 2. "Order placement rejected / Read-Only API"
- In TWS API Settings, ensure **"Read-Only API"** is **unchecked**.

### 3. PyTorch / Transformers / FinBERT slow or memory intensive
- FinBERT will automatically run on CUDA GPU if available; otherwise, it falls back to CPU.
- If running on low-resource machines, ensure batch sizes and news lookback days are kept modest.

### 4. Rate limits with NewsAPI
- NewsAPI free tier permits 100 requests per day. The bot automatically caches fetched news in `data/` to conserve API credits.

---

## ⚠️ Disclaimer

> [!CAUTION]
> **Financial & Trading Risk Warning:**
> - Trading stocks, penny stocks, and derivatives involves substantial risk of loss and is not suitable for every investor.
> - Past performance simulated in backtests does not guarantee future results.
> - Penny stocks are inherently volatile, illiquid, and subject to rapid price manipulation.
> - **Always start with paper trading** to thoroughly evaluate bot behavior before committing real capital.
> - This software is provided for educational and research purposes only. The authors and contributors assume no responsibility for financial losses incurred through the use of this software.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
