"""
main.py — CLI Entry Point
===========================
Command-line interface for the IBKR AI Trading Bot.

Usage:
    python main.py train          — Train ML model on historical data
    python main.py backtest       — Backtest the trained model
    python main.py scan-pennies   — Scan for top penny stock picks
    python main.py paper-trade    — Start paper trading (requires TWS/Gateway)
    python main.py monitor        — Show model health report
    python main.py info           — Show configuration summary
"""

import logging
import sys
from pathlib import Path

import click
import yaml

# Ensure directories exist BEFORE logging setup
for d in ["data", "models", "logs", "reports"]:
    Path(d).mkdir(exist_ok=True)

# Setup logging (UTF-8 safe for Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """🤖 IBKR AI Trading Bot — ML-powered stock trading with penny stock scanner."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.option("--symbols", default=None, help="Comma-separated symbols (overrides config)")
@click.option("--years", default=None, type=int, help="Years of historical data")
@click.option("--tune", is_flag=True, help="Run hyperparameter tuning with Optuna")
@click.option("--tune-trials", default=50, type=int, help="Number of Optuna trials")
@click.pass_context
def train(ctx, symbols, years, tune, tune_trials):
    """Train the ML model on historical data + news sentiment."""
    import numpy as np
    import pandas as pd
    from data_loader import DataLoader
    from feature_engine import FeatureEngine
    from sentiment import SentimentAnalyzer
    from ml_model import MLModel

    config_path = ctx.obj["config"]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve symbols
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = config.get("watchlist", {}).get("symbols", ["SPY", "AAPL"])

    if years:
        config["strategy"]["lookback_years"] = years

    # Determine training timeframe and period
    train_timeframe = config["strategy"].get("timeframe", "1d")
    train_years = config["strategy"].get("lookback_years", 1)
    # yfinance caps 15m/5m data at 60 days
    if train_timeframe in ("15m", "5m"):
        train_period = "60d"
    elif train_timeframe in ("1h", "1H"):
        train_period = "2y"  # yfinance allows up to 2y for 1h
    else:
        train_period = f"{train_years}y"

    # Read ensemble/model config
    model_cfg = config.get("model", {})
    ensemble_method = model_cfg.get("ensemble_method", "single")

    # Read label generation config
    label_mode = config["strategy"].get("label_mode", "fixed")
    label_atr_mult = config["strategy"].get("label_atr_multiplier", 1.5)

    logger.info("=" * 60)
    logger.info("🧠 TRAINING ML MODEL")
    logger.info(f"   Symbols: {', '.join(symbol_list)}")
    logger.info(f"   Timeframe: {train_timeframe} | Period: {train_period}")
    logger.info(f"   Model: {config['strategy'].get('model_type', 'xgboost')}")
    logger.info(f"   Ensemble: {ensemble_method}")
    logger.info(f"   Label mode: {label_mode} (ATR mult: {label_atr_mult})")
    logger.info(f"   Feature selection: {model_cfg.get('feature_selection', True)}")
    logger.info("=" * 60)

    data_loader = DataLoader(config_path)
    feature_engine = FeatureEngine()
    sentiment_analyzer = SentimentAnalyzer()
    ml_model = MLModel(config_path)

    # Collect data and features for all symbols
    all_features = []
    all_labels = []

    for symbol in symbol_list:
        logger.info(f"\n📦 Processing {symbol}...")

        # Fetch price data (use correct timeframe and capped period)
        df = data_loader.fetch_price_data(symbol, period=train_period, interval=train_timeframe)
        if df.empty or len(df) < 100:
            logger.warning(f"  Insufficient data for {symbol}, skipping")
            continue

        # Fetch news sentiment
        sentiment_features = {}
        try:
            articles = data_loader.fetch_news(symbol, days_back=7)
            if articles:
                scored = sentiment_analyzer.score_articles(articles)
                sentiment_features = sentiment_analyzer.compute_aggregate_sentiment(scored)
                logger.info(f"  📰 Sentiment: {len(articles)} articles scored")
        except Exception as e:
            logger.warning(f"  Sentiment failed: {e}")

        # Engineer features (including new microstructure, regime, statistical)
        features = feature_engine.compute_features(df, sentiment_features)
        if features.empty:
            continue

        # Create labels (convert horizon days to bars based on timeframe)
        horizon_days = config["strategy"].get("label_horizon_days", 2)
        timeframe = config["strategy"].get("timeframe", "1d")
        if timeframe in ("1h", "1H"):
            horizon_bars = max(int(horizon_days * 7), 5)  # 7 hourly bars per trading day
        elif timeframe == "15m":
            horizon_bars = max(int(horizon_days * 26), 10)
        else:
            horizon_bars = max(int(horizon_days), 1)

        threshold = config["strategy"].get("label_threshold_pct", 2.0)
        labels = feature_engine.create_labels(
            features,
            horizon=horizon_bars,
            threshold_pct=threshold,
            mode=label_mode,
            atr_multiplier=label_atr_mult,
        )

        # Align and drop NaN labels
        combined = features.copy()
        combined["label"] = labels
        combined.dropna(inplace=True)

        feature_cols = [c for c in combined.columns if c != "label"]
        all_features.append(combined[feature_cols])
        all_labels.append(combined["label"])

        logger.info(f"  ✅ {len(combined)} samples, {len(feature_cols)} features "
                     f"(horizon={horizon_bars} bars, mode={label_mode})")

    if not all_features:
        logger.error("❌ No training data collected. Check data sources.")
        return

    # Combine all symbols
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)

    # Remove 'Close' from features if present (it would leak future info)
    if "Close" in X.columns:
        X = X.drop(columns=["Close"])

    logger.info(f"\n📊 Combined dataset: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"   Label distribution: {dict(y.value_counts().sort_index().to_dict())}")

    # Train
    if tune:
        logger.info(f"\n🔧 Hyperparameter tuning ({tune_trials} trials)...")
        result = ml_model.tune_hyperparameters(X, y, n_trials=tune_trials)
        logger.info(f"   Best F1: {result['best_f1']:.4f}")
    else:
        # Walk-forward validation first
        logger.info("\n📈 Walk-forward validation (purged)...")
        wf_results = ml_model.walk_forward_validate(X, y, n_splits=5)

        if "error" not in wf_results:
            logger.info(f"   Walk-forward accuracy: {wf_results['overall_accuracy']:.4f}")
            logger.info(f"   Walk-forward F1: {wf_results['overall_f1']:.4f}")
            logger.info(f"   Buy signal win rate: {wf_results['buy_signal_win_rate']:.4f}")

        # Then train on full data with ensemble
        logger.info(f"\n🏋️ Training {ensemble_method} model on full dataset...")
        ml_model.train(X, y)

    # Save model
    ml_model.save("trading_model")
    logger.info("✅ Model saved to models/trading_model.joblib")

    # Feature importance
    importance = ml_model.get_feature_importance(top_n=15)
    if not importance.empty:
        logger.info("\n🏆 Top 15 Features:")
        for _, row in importance.iterrows():
            bar = "█" * int(row["importance"] * 100)
            logger.info(f"   {row['feature']:>30} | {bar} {row['importance']:.4f}")

    # Model health baseline
    logger.info("\n📊 Model health baseline set from training metrics")


@cli.command()
@click.option("--symbol", default="SPY", help="Symbol to backtest")
@click.option("--period", default="1y", help="Historical period to download (default: 1y)")
@click.option("--days", default=None, type=int, help="Limit backtest to the most recent N days (e.g. 7 for 1 week)")
@click.option("--interval", default="1h", help="Bar timeframe: 1d, 1h, 15m, 5m (default: 1h)")
@click.option("--capital", default=None, type=float, help="Initial capital in dollars (e.g. 30)")
@click.option("--confidence", default=None, type=float, help="Confidence threshold (e.g. 0.35)")
@click.pass_context
def backtest(ctx, symbol, period, days, interval, capital, confidence):
    """Backtest the trained model on historical data (supports 1 week via --days 7)."""
    import pandas as pd
    from data_loader import DataLoader
    from feature_engine import FeatureEngine
    from sentiment import SentimentAnalyzer
    from ml_model import MLModel
    from backtester import Backtester

    config_path = ctx.obj["config"]

    time_desc = f"Last {days} days" if days else f"{period} ({interval})"
    logger.info("=" * 60)
    logger.info(f"BACKTESTING — {symbol} | {time_desc} | Capital: ${capital or 100000:,.2f}")
    logger.info("=" * 60)

    data_loader = DataLoader(config_path)
    feature_engine = FeatureEngine()
    ml_model = MLModel(config_path)
    backtester = Backtester(config_path)
    if capital is not None:
        backtester.initial_capital = float(capital)

    # Load model
    try:
        ml_model.load("trading_model")
    except FileNotFoundError:
        logger.error("❌ No trained model found. Run 'python main.py train' first.")
        return

    # Map period strings (7d, 14d, 1mo, 3mo, 6mo) to exact days filter
    if days is None and period:
        if period.endswith("d") and period[:-1].isdigit():
            days = int(period[:-1])
        elif period == "1mo":
            days = 30
        elif period == "3mo":
            days = 90
        elif period == "6mo":
            days = 180

    # Fetch data (download enough history for indicator warm-up: ADX, 200 EMA)
    if interval == "1d":
        fetch_period = "2y"
    elif interval in ("15m", "5m"):
        fetch_period = "60d"     # yfinance caps 15m at 60 days
    elif interval in ("1h", "1H"):
        fetch_period = "3mo"
    else:
        fetch_period = period

    df = data_loader.fetch_price_data(symbol, period=fetch_period, interval=interval)
    if df.empty:
        logger.error(f"❌ No data for {symbol}")
        return

    # Compute features
    features = feature_engine.compute_features(df)
    if features.empty:
        logger.error("❌ Feature computation failed")
        return

    # Remove Close from features for prediction
    predict_features = features.drop(columns=["Close"], errors="ignore")

    # Get ML signals
    with open(config_path) as f:
        config = yaml.safe_load(f)
    threshold = confidence if confidence is not None else config["strategy"].get("confidence_threshold", 0.35)

    signals = ml_model.predict(predict_features, confidence_threshold=threshold)

    # ── Debug: show what the model actually predicted ──
    logger.info(f"\n📊 SIGNAL DIAGNOSTICS (confidence threshold = {threshold}):")
    raw_counts = signals["raw_signal"].value_counts().to_dict()
    final_counts = signals["signal"].value_counts().to_dict()
    logger.info(f"   Raw predictions (before threshold): BUY={raw_counts.get(1,0)}, SELL={raw_counts.get(-1,0)}, HOLD={raw_counts.get(0,0)}")
    logger.info(f"   After threshold filter:             BUY={final_counts.get(1,0)}, SELL={final_counts.get(-1,0)}, HOLD={final_counts.get(0,0)}")
    if "confidence" in signals.columns:
        logger.info(f"   Confidence stats: min={signals['confidence'].min():.3f}, "
                     f"mean={signals['confidence'].mean():.3f}, "
                     f"max={signals['confidence'].max():.3f}")
    if "prob_buy" in signals.columns:
        logger.info(f"   Buy probability:  min={signals['prob_buy'].min():.3f}, "
                     f"mean={signals['prob_buy'].mean():.3f}, "
                     f"max={signals['prob_buy'].max():.3f}")

    # Align prices with signals
    prices = df.loc[signals.index]

    # If --days is specified, slice to the most recent N days
    if days is not None and days > 0:
        cutoff_date = prices.index.max() - pd.Timedelta(days=days)
        prices = prices[prices.index >= cutoff_date]
        signals = signals.loc[prices.index]
        if prices.empty:
            logger.error(f"❌ No data found in the last {days} days")
            return
        logger.info(f"Filtered to the most recent {len(prices)} bars (last {days} days)")

        # Show filtered signal counts too
        filt_counts = signals["signal"].value_counts().to_dict()
        logger.info(f"   Signals in window: BUY={filt_counts.get(1,0)}, SELL={filt_counts.get(-1,0)}, HOLD={filt_counts.get(0,0)}")

    # Run backtest
    results = backtester.run(prices, signals)

    logger.info(f"\n✅ Backtest complete — {results['metrics']['total_trades']} trades "
                 f"(Long: {results['metrics'].get('long_trades', '?')}, "
                 f"Short: {results['metrics'].get('short_trades', '?')})")


@cli.command("scan-pennies")
@click.option("--top", default=10, type=int, help="Number of top picks")
@click.pass_context
def scan_pennies(ctx, top):
    """Scan for top penny stock picks today."""
    from penny_scanner import PennyScanner
    from ml_model import MLModel

    config_path = ctx.obj["config"]

    logger.info("=" * 60)
    logger.info("🔍 PENNY STOCK SCANNER")
    logger.info("=" * 60)

    scanner = PennyScanner(config_path)
    scanner.top_n = top

    # Try to load ML model for confidence overlay
    ml_model = None
    try:
        ml_model = MLModel(config_path)
        ml_model.load("trading_model")
        logger.info("✅ ML model loaded for confidence scoring")
    except FileNotFoundError:
        logger.info("ℹ️ No ML model found — scanning without ML confidence")

    picks = scanner.scan(ml_model=ml_model)

    if picks.empty:
        logger.warning("No penny stock picks found. Try adjusting filters in config.yaml")
    else:
        logger.info(f"\n✅ {len(picks)} penny stock picks generated!")
        logger.info(f"   Report saved to reports/")


@cli.command("paper-trade")
@click.pass_context
def paper_trade(ctx):
    """Start paper trading (requires TWS/Gateway running)."""
    from bot import TradingBot

    config_path = ctx.obj["config"]
    bot = TradingBot(config_path)
    bot.start(paper=True)


@cli.command()
@click.pass_context
def monitor(ctx):
    """Show model health report and prediction quality analysis."""
    from model_monitor import ModelMonitor
    from ml_model import MLModel

    config_path = ctx.obj["config"]

    logger.info("=" * 60)
    logger.info("🧠 MODEL HEALTH MONITOR")
    logger.info("=" * 60)

    monitor = ModelMonitor(config_path)

    # Load training accuracy baseline
    try:
        ml_model = MLModel(config_path)
        ml_model.load("trading_model")
        training_acc = ml_model.training_metrics.get("train_accuracy", 0)
        if training_acc:
            monitor.set_training_baseline(accuracy=training_acc)
            logger.info(f"Training baseline: {training_acc:.4f}")
    except FileNotFoundError:
        logger.warning("No trained model found — some metrics unavailable")

    # Print health report
    monitor.print_health_report()

    # Print regime performance
    regime_perf = monitor.get_regime_performance()
    if regime_perf.get("status") != "insufficient_data":
        logger.info("\n📊 Performance by Regime:")
        for regime, stats in regime_perf.items():
            if isinstance(stats, dict):
                logger.info(f"   {regime:>10}: accuracy={stats['accuracy']:.2%}, "
                             f"avg_pnl=${stats['avg_pnl']:+.2f}, "
                             f"total_pnl=${stats['total_pnl']:+.2f} "
                             f"({stats['count']} trades)")


@cli.command()
@click.pass_context
def info(ctx):
    """Show current configuration summary."""
    config_path = ctx.obj["config"]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Connection info
        ibkr = config.get("ibkr", {})
        is_paper = ibkr.get("port", 7497) in (7497, 4002)
        console.print(Panel(
            f"[bold]IBKR Connection[/bold]\n"
            f"  Host: {ibkr.get('host')}:{ibkr.get('port')}\n"
            f"  Mode: {'📝 Paper' if is_paper else '💰 Live'}\n"
            f"  Client ID: {ibkr.get('client_id')}",
            title="🔌 Connection",
            border_style="blue",
        ))

        # Watchlist
        symbols = config.get("watchlist", {}).get("symbols", [])
        console.print(Panel(
            f"  {', '.join(symbols)}",
            title="📋 Watchlist",
            border_style="cyan",
        ))

        # Strategy
        strat = config.get("strategy", {})
        model_cfg = config.get("model", {})
        console.print(Panel(
            f"  Timeframe: {strat.get('timeframe')}\n"
            f"  Model: {strat.get('model_type')}\n"
            f"  Ensemble: {model_cfg.get('ensemble_method', 'single')}\n"
            f"  Base models: {', '.join(model_cfg.get('base_models', []))}\n"
            f"  Feature selection: {model_cfg.get('feature_selection', True)}\n"
            f"  Max features: {model_cfg.get('max_features', 30)}\n"
            f"  Lookback: {strat.get('lookback_years')}y\n"
            f"  Confidence: {strat.get('confidence_threshold')}\n"
            f"  Label mode: {strat.get('label_mode', 'fixed')}\n"
            f"  Label horizon: {strat.get('label_horizon_days')}d\n"
            f"  Label threshold: {strat.get('label_threshold_pct')}%",
            title="🧠 Strategy",
            border_style="green",
        ))

        # Risk
        risk = config.get("risk", {})
        targets = config.get("targets", {})
        console.print(Panel(
            f"  Max risk/trade: {risk.get('max_risk_per_trade_pct')}%\n"
            f"  Max position: {risk.get('max_position_pct')}%\n"
            f"  Max daily loss: {risk.get('max_daily_loss_pct')}%\n"
            f"  Max positions: {risk.get('max_open_positions')}\n"
            f"  SL ATR mult: {risk.get('stop_loss_atr_mult')}x\n"
            f"  TP ATR mult: {risk.get('take_profit_atr_mult')}x\n"
            f"  Trailing stop: {risk.get('trailing_stop_pct')}%\n"
            f"  Time stop: {risk.get('time_stop_bars', 'N/A')} bars\n"
            f"  Daily profit target: ${targets.get('daily_profit_target', 5.0)}\n"
            f"  Scale down at: {targets.get('scale_down_at_pct', 60)}% of target\n"
            f"  Stop trading at: {targets.get('stop_trading_at_pct', 120)}% of target",
            title="🛡️ Risk Management",
            border_style="yellow",
        ))

        # Regime
        regime_cfg = config.get("regime", {})
        if regime_cfg.get("enabled"):
            console.print(Panel(
                f"  Enabled: {regime_cfg.get('enabled')}\n"
                f"  Volatility lookback: {regime_cfg.get('volatility_lookback')} bars\n"
                f"  Trend ADX threshold: {regime_cfg.get('trend_strength_min_adx')}\n"
                f"  High vol threshold: {regime_cfg.get('high_vol_threshold')}%ile\n"
                f"  Low vol threshold: {regime_cfg.get('low_vol_threshold')}%ile",
                title="📊 Regime Detection",
                border_style="bright_blue",
            ))

        # Penny Scanner
        penny = config.get("penny_scanner", {})
        console.print(Panel(
            f"  Price range: ${penny.get('min_price')} – ${penny.get('max_price')}\n"
            f"  Min volume: {penny.get('min_avg_volume'):,}\n"
            f"  Top picks: {penny.get('top_n_picks')}\n"
            f"  Exchanges: {', '.join(penny.get('exchanges', []))}",
            title="🔍 Penny Scanner",
            border_style="magenta",
        ))

        # Check model status
        model_path = Path("models/trading_model.joblib")
        if model_path.exists():
            import json
            meta_path = Path("models/trading_model_meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                console.print(Panel(
                    f"  Status: ✅ Trained\n"
                    f"  Type: {meta.get('model_type')}\n"
                    f"  Ensemble: {meta.get('ensemble_method', 'single')}\n"
                    f"  Trained at: {meta.get('saved_at', 'Unknown')}\n"
                    f"  Features (original): {len(meta.get('feature_names', []))}\n"
                    f"  Features (selected): {len(meta.get('selected_features', meta.get('feature_names', [])))}",
                    title="🤖 Model Status",
                    border_style="green",
                ))
            else:
                console.print("[green]✅ Model file found[/green]")
        else:
            console.print(Panel(
                "  Status: ❌ Not trained\n"
                "  Run: python main.py train",
                title="🤖 Model Status",
                border_style="red",
            ))

    except ImportError:
        # Fallback without rich
        print("\n=== Configuration ===")
        print(yaml.dump(config, default_flow_style=False))


if __name__ == "__main__":
    cli()
