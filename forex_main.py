"""
forex_main.py — CLI Entry Point for Forex Trading Bot
=======================================================
Command-line interface for the Forex AI Trading Bot.

Usage:
    python forex_main.py train          — Train ML model on forex historical data
    python forex_main.py backtest       — Backtest the trained model on forex pairs
    python forex_main.py scan-pairs     — Rank forex pairs by signal strength
    python forex_main.py paper-trade    — Start paper trading (requires TWS/Gateway)
    python forex_main.py monitor        — Show model health report
    python forex_main.py info           — Show configuration summary

    MT5/XM Commands:
    python forex_main.py mt5-trade      — Start AI trading on MT5/XM
    python forex_main.py mt5-info       — Show MT5 account, positions, prices
    python forex_main.py mt5-close-all  — Emergency close all MT5 positions
"""

import logging
import os
import sys
from pathlib import Path

import click
import yaml

# Ensure directories exist
for d in ["data/forex", "models/forex", "logs", "reports/forex"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Setup logging
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/forex_bot.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", default="forex_config.yaml", help="Path to forex config file")
@click.pass_context
def cli(ctx, config):
    """🤖 Forex AI Trading Bot — ML-powered forex trading with IBKR & MT5/XM."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@cli.command()
@click.option("--pairs", default=None, help="Comma-separated pairs (overrides config)")
@click.option("--years", default=None, type=int, help="Years of historical data")
@click.option("--period", default=None, help="Historical period (e.g. 60d, 1y, 2y)")
@click.option("--timeframe", default=None, help="Bar timeframe (e.g. 1h, 4h, 1d)")
@click.option("--tune", is_flag=True, help="Run hyperparameter tuning with Optuna")
@click.option("--tune-trials", default=50, type=int, help="Number of Optuna trials")
@click.pass_context
def train(ctx, pairs, years, period, timeframe, tune, tune_trials):
    """Train the ML model on historical forex data."""
    import numpy as np
    import pandas as pd
    from forex_data_loader import ForexDataLoader
    from forex_feature_engine import ForexFeatureEngine
    from forex_ml_model import ForexMLModel

    config_path = ctx.obj["config"]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve pairs
    if pairs:
        pair_list = [p.strip().upper() for p in pairs.split(",")]
    else:
        pair_list = config.get("watchlist", {}).get("active_pairs", ["EURUSD"])

    if years:
        config["strategy"]["lookback_years"] = years
    if timeframe:
        config["strategy"]["timeframe"] = timeframe

    train_timeframe = config["strategy"].get("timeframe", "1h")

    # Determine period
    if period:
        train_period = period
    elif train_timeframe in ("15m", "5m"):
        train_period = "60d"
    elif train_timeframe in ("1h", "1H"):
        train_period = "2y"
    else:
        train_period = f"{config['strategy'].get('lookback_years', 2)}y"

    label_mode = config["strategy"].get("label_mode", "atr_relative")
    label_atr_mult = config["strategy"].get("label_atr_multiplier", 1.0)
    ensemble_method = config.get("model", {}).get("ensemble_method", "single")

    logger.info("=" * 60)
    logger.info("🧠 TRAINING FOREX ML MODEL")
    logger.info(f"   Pairs: {', '.join(pair_list)}")
    logger.info(f"   Timeframe: {train_timeframe} | Period: {train_period}")
    logger.info(f"   Model: {config['strategy'].get('model_type', 'xgboost')}")
    logger.info(f"   Ensemble: {ensemble_method}")
    logger.info(f"   Label mode: {label_mode} (ATR mult: {label_atr_mult})")
    logger.info("=" * 60)

    data_loader = ForexDataLoader(config_path)
    feature_engine = ForexFeatureEngine()
    ml_model = ForexMLModel(config_path)

    # Fetch macro context
    macro_df = None
    try:
        macro_df = data_loader.fetch_macro_context(period=train_period, interval=train_timeframe)
        if macro_df is not None and not macro_df.empty:
            logger.info(f"🌐 Macro context loaded: {len(macro_df)} bars (DXY, US10Y, Gold, Oil, VIX)")
    except Exception as e:
        logger.warning(f"Could not load macro context: {e}")

    # Compute currency strength
    currency_strength = {}
    try:
        currency_strength = data_loader.compute_currency_strength(
            period="60d", interval="1d", lookback_bars=10
        )
    except Exception as e:
        logger.warning(f"Currency strength calc failed: {e}")

    # Collect data and features for all pairs
    all_features = []
    all_labels = []

    for pair in pair_list:
        logger.info(f"\n📦 Processing {pair}...")

        df = data_loader.fetch_forex_data(pair, period=train_period, interval=train_timeframe)
        if df.empty or len(df) < 200:
            logger.warning(f"  Insufficient data for {pair} ({len(df)} bars), skipping")
            continue

        # Compute features
        features = feature_engine.compute_features(
            df, pair=pair,
            macro_df=macro_df,
            currency_strength=currency_strength,
        )
        if features.empty:
            continue

        # Create labels
        horizon_bars = config["strategy"].get("label_horizon_bars", 24)
        labels = feature_engine.create_labels(
            features, pair=pair, horizon=horizon_bars,
            mode=label_mode, atr_multiplier=label_atr_mult,
        )

        # Align and drop NaN labels
        combined = features.copy()
        combined["label"] = labels
        combined.dropna(inplace=True)

        feature_cols = [c for c in combined.columns if c != "label"]
        all_features.append(combined[feature_cols])
        all_labels.append(combined["label"])

        logger.info(f"  ✅ {len(combined)} samples, {len(feature_cols)} features "
                     f"(horizon={horizon_bars} bars)")

    if not all_features:
        logger.error("❌ No training data collected. Check data sources.")
        return

    # Combine all pairs
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)

    # Remove Close from features
    if "Close" in X.columns:
        X = X.drop(columns=["Close"])

    logger.info(f"\n📊 Combined dataset: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"   Label distribution: {dict(y.value_counts().sort_index().to_dict())}")

    # Purge gap
    wf_purge_gap = max(horizon_bars * 2, 48)

    # Train
    if tune:
        logger.info(f"\n🔧 Hyperparameter tuning ({tune_trials} trials)...")
        result = ml_model.tune_hyperparameters(X, y, n_trials=tune_trials)
        logger.info(f"   Best F1: {result['best_f1']:.4f}")
        logger.info("\n📈 Walk-forward validation with best params...")
        wf_results = ml_model.walk_forward_validate(X, y, n_splits=5,
                                                     params=result.get("best_params"),
                                                     purge_gap=wf_purge_gap)
        ml_model.train(X, y, params=result.get("best_params"))
    else:
        logger.info(f"\n📈 Walk-forward validation (purge: {wf_purge_gap} bars)...")
        wf_results = ml_model.walk_forward_validate(X, y, n_splits=5, purge_gap=wf_purge_gap)

        if "error" not in wf_results:
            logger.info(f"   Walk-forward accuracy: {wf_results['overall_accuracy']:.4f}")
            logger.info(f"   Walk-forward F1: {wf_results['overall_f1']:.4f}")
            logger.info(f"   Buy signal win rate: {wf_results['buy_signal_win_rate']:.4f}")

        logger.info(f"\n🏋️ Training {ensemble_method} model on full dataset...")
        ml_model.train(X, y)

    # Save model
    ml_model.save("forex_trading_model")
    logger.info(f"\n🎉 Forex model saved! Ready for backtesting and trading.")


@cli.command()
@click.option("--pair", default="EURUSD", help="Forex pair to backtest")
@click.option("--period", default="6mo", help="Historical period (default: 6mo)")
@click.option("--days", default=None, type=int, help="Limit backtest to last N days")
@click.option("--interval", default="1h", help="Bar timeframe (default: 1h)")
@click.option("--capital", default=None, type=float, help="Initial capital in USD")
@click.option("--confidence", default=None, type=float, help="Confidence threshold")
@click.option("--leverage", default=None, type=int, help="Leverage (default: 10)")
@click.option("--lots", "--quantity", default=None, type=float, help="Fixed lot size/quantity (e.g. 0.01, 0.02, 0.05)")
@click.pass_context
def backtest(ctx, pair, period, days, interval, capital, confidence, leverage, lots):
    """Backtest the trained model on forex historical data."""
    import pandas as pd
    from forex_data_loader import ForexDataLoader
    from forex_feature_engine import ForexFeatureEngine
    from forex_ml_model import ForexMLModel
    from forex_backtester import ForexBacktester

    config_path = ctx.obj["config"]

    data_loader = ForexDataLoader(config_path)
    feature_engine = ForexFeatureEngine()
    ml_model = ForexMLModel(config_path)
    backtester = ForexBacktester(config_path)

    if capital is not None:
        backtester.initial_capital = float(capital)
    if leverage is not None:
        backtester.leverage = int(leverage)
    if lots is not None:
        backtester.fixed_lots = float(lots)

    try:
        ml_model.load("forex_trading_model")
    except FileNotFoundError:
        logger.error("❌ No trained model found. Run 'python forex_main.py train' first.")
        return

    pairs_to_test = [p.strip().upper() for p in pair.split(",") if p.strip()]
    if pair.lower() == "all":
        with open(config_path) as f:
            cfg_temp = yaml.safe_load(f)
        pairs_to_test = cfg_temp.get("watchlist", {}).get("active_pairs", ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "USDCHF"])

    all_results = {}
    for p in pairs_to_test:
        logger.info("=" * 60)
        logger.info(f"FOREX BACKTEST — {p} | {period} ({interval}) | Capital: ${capital or 500:.2f}")
        logger.info("=" * 60)

        # Reset backtester capital and trades for each pair
        backtester.current_capital = backtester.initial_capital
        backtester.equity_curve = []
        backtester.trades = []

        fetch_period = "2y" if interval in ("1h", "1H") else period
        df = data_loader.fetch_forex_data(p, period=fetch_period, interval=interval)
        if df.empty:
            logger.error(f"❌ No data for {p}")
            continue

        macro_df = data_loader.fetch_macro_context(period=fetch_period, interval=interval)
        currency_strength = data_loader.compute_currency_strength(period="60d", interval="1d", lookback_bars=10)

        features = feature_engine.compute_features(
            df, pair=p, macro_df=macro_df, currency_strength=currency_strength,
        )
        if features.empty:
            logger.error(f"❌ Feature computation failed for {p}")
            continue

        predict_features = features.drop(columns=["Close"], errors="ignore")

        with open(config_path) as f:
            config = yaml.safe_load(f)
        threshold = confidence if confidence is not None else config["strategy"].get("confidence_threshold", 0.40)
        signals = ml_model.predict(predict_features, confidence_threshold=threshold)

        # Apply Hybrid Strategy Engine if enabled
        if config.get("hybrid", {}).get("enabled", True):
            from forex_strategy_engine import ForexStrategyEngine
            strat_engine = ForexStrategyEngine(config_path)

            eval_df = features.copy()
            for col in ["Open", "High", "Low"]:
                if col in df.columns and col not in eval_df.columns:
                    eval_df[col] = df.loc[eval_df.index, col]

            signals = strat_engine.generate_hybrid_signals(
                df=eval_df,
                ml_signals=signals,
                currency_strength=currency_strength,
                pair=p,
            )
            active_sig_count = len(signals[signals['signal'] != 0])
            logger.info(f"✨ Hybrid Engine filtered signals: {active_sig_count} high-conviction entries")

        prices = df.loc[signals.index]

        if days is not None and days > 0:
            cutoff = prices.index.max() - pd.Timedelta(days=days)
            prices = prices[prices.index >= cutoff]
            signals = signals.loc[prices.index]
        elif period and period != "2y":
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "ytd": 260}
            p_str = period.lower()
            if p_str.endswith("d") and p_str[:-1].isdigit():
                p_days = int(p_str[:-1])
            else:
                p_days = period_days.get(p_str)
            if p_days:
                cutoff = prices.index.max() - pd.Timedelta(days=p_days)
                prices = prices[prices.index >= cutoff]
                signals = signals.loc[prices.index]

        feat_slice = predict_features.loc[prices.index] if not predict_features.empty else None
        res = backtester.run(p, prices, signals, features=feat_slice)
        all_results[p] = res

    # If multiple pairs were backtested, print portfolio summary
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info(f"🌐 MULTI-PAIR PORTFOLIO SUMMARY — {interval} ({period})")
        logger.info("=" * 60)
        total_trades = sum(r["metrics"]["total_trades"] for r in all_results.values())
        total_wins = sum(r["metrics"]["winning_trades"] for r in all_results.values())
        total_pnl = sum(r["metrics"]["net_pnl_usd"] for r in all_results.values())
        total_pips = sum(r["metrics"]["net_pnl_pips"] for r in all_results.values())
        total_spread = sum(r["metrics"]["total_spread_cost"] for r in all_results.values())
        overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

        for p, r in all_results.items():
            m = r["metrics"]
            logger.info(f"  {p:<7} | Trades: {m['total_trades']:<2} | Win: {m['win_rate']:>5.1f}% | "
                         f"Pips: {m['net_pnl_pips']:>+6.1f} | PnL: ${m['net_pnl_usd']:>+6.2f} | "
                         f"PF: {m['profit_factor']:>4.2f}")
        logger.info("-" * 60)
        combined_return_pct = (total_pnl / (backtester.initial_capital) * 100)
        logger.info(f"  TOTALS  | Trades: {total_trades:<2} | Win: {overall_wr:>5.1f}% | "
                     f"Pips: {total_pips:>+6.1f} | PnL: ${total_pnl:>+6.2f} ({combined_return_pct:>+5.2f}%) | "
                     f"Spread: ${total_spread:.2f}")
        logger.info("=" * 60)


@cli.command("scan-pairs")
@click.option("--top", default=7, type=int, help="Number of top pairs to show")
@click.pass_context
def scan_pairs(ctx, top):
    """Rank forex pairs by current signal strength."""
    from forex_data_loader import ForexDataLoader
    from forex_feature_engine import ForexFeatureEngine
    from forex_ml_model import ForexMLModel

    config_path = ctx.obj["config"]

    logger.info("=" * 60)
    logger.info("🔍 FOREX PAIR SCANNER")
    logger.info("=" * 60)

    data_loader = ForexDataLoader(config_path)
    feature_engine = ForexFeatureEngine()

    # Try to load model
    ml_model = None
    try:
        ml_model = ForexMLModel(config_path)
        ml_model.load("forex_trading_model")
        logger.info("✅ ML model loaded for scoring")
    except FileNotFoundError:
        logger.info("ℹ️ No ML model — scanning without ML confidence")

    # Currency strength
    strength = data_loader.compute_currency_strength(period="30d", interval="1d", lookback_bars=5)

    all_pairs = data_loader.get_all_pairs()
    scores = []

    for pair in all_pairs:
        try:
            info = data_loader.fetch_pair_info(pair)
            if not info.get("is_valid"):
                continue

            score = 0.0
            # ADR score
            adr = info.get("adr_pips", 0)
            if adr > 40:
                score += 0.3

            # Currency strength divergence
            clean = data_loader.normalize_pair(pair)
            base_str = strength.get(clean[:3], 0)
            quote_str = strength.get(clean[3:], 0)
            divergence = abs(base_str - quote_str)
            score += divergence * 10

            # ML confidence (if model available)
            ml_conf = 0.0
            ml_signal = 0
            if ml_model:
                try:
                    df = data_loader.fetch_forex_data(pair, period="60d", interval="1h")
                    if not df.empty and len(df) > 100:
                        df_slice = df.tail(300)
                        features = feature_engine.compute_features(
                            df_slice, pair=pair, currency_strength=strength,
                        )
                        if not features.empty:
                            latest = features.iloc[[-1]]
                            pred = ml_model.predict(latest.drop(columns=["Close"], errors="ignore"))
                            ml_signal = int(pred["signal"].iloc[0])
                            ml_conf = float(pred["confidence"].iloc[0])
                            if ml_signal != 0:
                                score += ml_conf * 0.5
                except Exception:
                    pass

            scores.append({
                "pair": clean,
                "score": round(score, 3),
                "adr_pips": adr,
                "atr_pips": info.get("atr_14_pips", 0),
                "base_strength": round(base_str, 4),
                "quote_strength": round(quote_str, 4),
                "ml_signal": "BUY" if ml_signal == 1 else ("SELL" if ml_signal == -1 else "HOLD"),
                "ml_confidence": round(ml_conf * 100, 1),
            })
        except Exception as e:
            logger.debug(f"Scan failed for {pair}: {e}")

    # Sort by score
    scores.sort(key=lambda x: -x["score"])

    logger.info(f"\n📊 Top {top} Forex Pairs:")
    logger.info(f"{'Pair':<8} {'Score':>6} {'ADR':>5} {'ATR':>5} {'Base':>7} {'Quote':>7} {'Signal':>6} {'Conf':>5}")
    logger.info("─" * 60)
    for s in scores[:top]:
        logger.info(f"{s['pair']:<8} {s['score']:6.3f} {s['adr_pips']:5.0f} {s['atr_pips']:5.0f} "
                     f"{s['base_strength']:+7.4f} {s['quote_strength']:+7.4f} "
                     f"{s['ml_signal']:>6} {s['ml_confidence']:5.1f}%")


@cli.command("paper-trade")
@click.option("--lots", "--quantity", default=None, type=float, help="Fixed lot size/quantity override (e.g. 0.01, 0.02)")
@click.pass_context
def paper_trade(ctx, lots):
    """Start paper trading (requires TWS/Gateway running)."""
    from forex_bot import ForexTradingBot

    config_path = ctx.obj["config"]
    bot = ForexTradingBot(config_path)
    if lots is not None and lots > 0:
        bot.risk_manager.fixed_lots = lots
    bot.start(paper=True)


@cli.command()
@click.pass_context
def monitor(ctx):
    """Show model health report."""
    from forex_ml_model import ForexMLModel

    config_path = ctx.obj["config"]
    logger.info("=" * 60)
    logger.info("🧠 FOREX MODEL HEALTH")
    logger.info("=" * 60)

    try:
        model = ForexMLModel(config_path)
        model.load("forex_trading_model")
        health = model.get_model_health()
        logger.info(f"  Status: {health['status']}")
        if "rolling_accuracy" in health:
            logger.info(f"  Rolling accuracy: {health['rolling_accuracy']:.4f}")
            logger.info(f"  Training accuracy: {health['training_accuracy']:.4f}")
            logger.info(f"  Accuracy decay: {health['accuracy_decay_pct']:.2f}%")
        logger.info(f"  Predictions logged: {health.get('predictions_logged', 0)}")

        metrics = model.training_metrics
        if metrics:
            logger.info(f"\n  Training metrics:")
            logger.info(f"    Accuracy: {metrics.get('train_accuracy', 'N/A')}")
            logger.info(f"    F1: {metrics.get('train_f1', 'N/A')}")
            logger.info(f"    Samples: {metrics.get('n_samples', 'N/A')}")
            logger.info(f"    Features: {metrics.get('n_features_selected', 'N/A')}")
            logger.info(f"    Trained: {metrics.get('trained_at', 'N/A')}")

    except FileNotFoundError:
        logger.warning("❌ No trained forex model found")


@cli.command()
@click.pass_context
def info(ctx):
    """Show current forex configuration summary."""
    config_path = ctx.obj["config"]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()

        # Connection
        ibkr = config.get("ibkr", {})
        is_paper = ibkr.get("port", 7497) in (7497, 4002)
        console.print(Panel(
            f"[bold]IBKR Connection[/bold]\n"
            f"  Host: {ibkr.get('host')}:{ibkr.get('port')}\n"
            f"  Mode: {'📝 Paper' if is_paper else '💰 Live'}\n"
            f"  Contract: {ibkr.get('contract_type', 'CASH')}",
            title="🔌 Connection", border_style="blue",
        ))

        # Pairs
        watchlist = config.get("watchlist", {})
        console.print(Panel(
            f"  Active: {', '.join(watchlist.get('active_pairs', []))}\n"
            f"  Majors: {len(watchlist.get('majors', []))}\n"
            f"  Crosses: {len(watchlist.get('crosses', []))}",
            title="💱 Watchlist", border_style="cyan",
        ))

        # Strategy
        strat = config.get("strategy", {})
        model_cfg = config.get("model", {})
        console.print(Panel(
            f"  Timeframe: {strat.get('timeframe')}\n"
            f"  Model: {strat.get('model_type')}\n"
            f"  Ensemble: {model_cfg.get('ensemble_method', 'single')}\n"
            f"  Horizon: {strat.get('label_horizon_bars')} bars\n"
            f"  Label mode: {strat.get('label_mode')}\n"
            f"  Confidence: {strat.get('confidence_threshold')}",
            title="🧠 Strategy", border_style="green",
        ))

        # Leverage & Risk
        lev = config.get("leverage", {})
        risk = config.get("risk", {})
        console.print(Panel(
            f"  Max leverage: {lev.get('max_leverage')}:1\n"
            f"  Margin req: {lev.get('margin_requirement_pct')}%\n"
            f"  Max risk/trade: {risk.get('max_risk_per_trade_pct')}%\n"
            f"  Max positions: {risk.get('max_open_positions')}\n"
            f"  SL: {risk.get('stop_loss_atr_mult')}× ATR\n"
            f"  TP: {risk.get('take_profit_atr_mult')}× ATR\n"
            f"  Max spread: {risk.get('max_entry_spread_pips')} pips\n"
            f"  Max correlation: {risk.get('max_correlation')}",
            title="🛡️ Risk & Leverage", border_style="yellow",
        ))

        # Model status
        model_path = Path("models/forex/forex_trading_model.joblib")
        if model_path.exists():
            import json
            meta_path = Path("models/forex/forex_trading_model_meta.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                console.print(Panel(
                    f"  Status: ✅ Trained\n"
                    f"  Type: {meta.get('model_type')}\n"
                    f"  Ensemble: {meta.get('ensemble_method')}\n"
                    f"  Trained: {meta.get('saved_at', 'Unknown')}\n"
                    f"  Features: {len(meta.get('selected_features', meta.get('feature_names', [])))}",
                    title="🤖 Model Status", border_style="green",
                ))
        else:
            console.print(Panel(
                "  Status: ❌ Not trained\n  Run: python forex_main.py train",
                title="🤖 Model Status", border_style="red",
            ))

    except ImportError:
        print("\n=== Forex Configuration ===")
        print(yaml.dump(config, default_flow_style=False))


# ──────────────────────────────────────────────────────────
#  MT5 / XM COMMANDS
# ──────────────────────────────────────────────────────────

@cli.command("mt5-trade")
@click.option("--lots", "--quantity", default=None, type=float, help="Fixed lot size/quantity override (e.g. 0.01, 0.02)")
@click.pass_context
def mt5_trade(ctx, lots):
    """🚀 Start live/paper trading on MT5/XM.

    Connects to the MT5 terminal and runs the AI trading bot.
    Uses the same trained ML model and risk management as the IBKR bot.

    Prerequisites:
        1. MetaTrader 5 terminal installed and running
        2. Logged into your XM demo/live account
        3. Model trained: python forex_main.py train

    Example:
        python forex_main.py mt5-trade --lots 0.02
    """
    from forex_mt5_bot import ForexMT5Bot

    logger.info("=" * 60)
    logger.info("MT5/XM FOREX AI TRADING BOT")
    logger.info("=" * 60)

    bot = ForexMT5Bot(ctx.obj["config"], fixed_lots=lots)
    bot.start()


@cli.command("mt5-info")
@click.pass_context
def mt5_info(ctx):
    """📊 Show MT5 account info, balance, and open positions.

    Connects to the MT5 terminal and displays:
        - Account details (login, server, broker)
        - Balance, equity, margin, leverage
        - Open positions with P&L
        - Live prices and spreads for active pairs

    Example:
        python forex_main.py mt5-info
    """
    from forex_mt5_client import ForexMT5Client

    client = ForexMT5Client(ctx.obj["config"])
    if not client.connect():
        logger.error("Failed to connect to MT5. Is the terminal running?")
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Account info
        info = client.get_account_info()
        console.print(Panel(
            f"  Login: {info.get('login')}\n"
            f"  Name: {info.get('name')}\n"
            f"  Server: {info.get('server')}\n"
            f"  Broker: {info.get('company')}\n"
            f"  Mode: {info.get('trade_mode')}\n"
            f"  Currency: {info.get('currency')}\n"
            f"  Leverage: 1:{info.get('leverage', 0)}",
            title="🏦 MT5 Account", border_style="cyan",
        ))

        console.print(Panel(
            f"  Balance:      ${info.get('balance', 0):>12,.2f}\n"
            f"  Equity:       ${info.get('equity', 0):>12,.2f}\n"
            f"  Margin:       ${info.get('margin', 0):>12,.2f}\n"
            f"  Free Margin:  ${info.get('free_margin', 0):>12,.2f}\n"
            f"  Margin Level: {info.get('margin_level', 0):>12.1f}%\n"
            f"  Floating P&L: ${info.get('profit', 0):>+12,.2f}",
            title="💰 Balance & Margin", border_style="green",
        ))

        # Open positions
        positions = client.get_positions()
        if positions:
            table = Table(title="📈 Open Positions", border_style="blue")
            table.add_column("Pair", style="cyan")
            table.add_column("Dir", style="bold")
            table.add_column("Lots", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("SL", justify="right")
            table.add_column("TP", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("Swap", justify="right")

            total_pnl = 0
            for pos in positions:
                pnl_color = "green" if pos["pnl"] >= 0 else "red"
                table.add_row(
                    pos["pair"],
                    pos["direction"],
                    f"{pos['lots']:.2f}",
                    f"{pos['avg_cost']:.5f}",
                    f"{pos['current_price']:.5f}",
                    f"{pos['sl']:.5f}" if pos["sl"] > 0 else "—",
                    f"{pos['tp']:.5f}" if pos["tp"] > 0 else "—",
                    f"[{pnl_color}]${pos['pnl']:+.2f}[/]",
                    f"${pos['swap']:.2f}",
                )
                total_pnl += pos["pnl"]

            console.print(table)
            console.print(f"  Total floating P&L: ${total_pnl:+,.2f}")
        else:
            console.print("[dim]No open positions[/dim]")

        # Live prices
        with open(ctx.obj["config"]) as f:
            cfg = yaml.safe_load(f)
        active_pairs = cfg.get("watchlist", {}).get("active_pairs", ["GBPUSD", "EURUSD", "GBPJPY"])

        price_table = Table(title="📊 XM Live Market Watch", border_style="yellow")
        price_table.add_column("Pair", style="cyan")
        price_table.add_column("Bid (Sell)", justify="right", style="bright_white")
        price_table.add_column("Ask (Buy)", justify="right", style="bright_white")
        price_table.add_column("Mid Price", justify="right")
        price_table.add_column("Spread (pips)", justify="right")

        for pair in active_pairs:
            tick = client.get_tick(pair)
            if tick and tick.get("mid", 0) > 0:
                spread = tick["spread_pips"]
                spread_color = "green" if spread < 2.5 else ("yellow" if spread <= 4.5 else "red")
                price_table.add_row(
                    pair,
                    f"{tick['bid']:.5f}",
                    f"{tick['ask']:.5f}",
                    f"{tick['mid']:.5f}",
                    f"[{spread_color}]{spread:.1f}[/]",
                )
            else:
                price_table.add_row(pair, "N/A", "N/A", "N/A", "N/A")

        console.print(price_table)

    except ImportError:
        # Fallback without Rich
        info = client.get_account_info()
        print(f"\n=== MT5 Account ===")
        print(f"  Login: {info.get('login')} ({info.get('trade_mode')})")
        print(f"  Broker: {info.get('company')}")
        print(f"  Balance: ${info.get('balance', 0):,.2f}")
        print(f"  Equity: ${info.get('equity', 0):,.2f}")
        print(f"  Leverage: 1:{info.get('leverage', 0)}")

        positions = client.get_positions()
        print(f"\n  Open positions: {len(positions)}")
        for pos in positions:
            print(f"    {pos['pair']} {pos['direction']} {pos['lots']} lots | P&L: ${pos['pnl']:+.2f}")

    finally:
        client.disconnect()


@cli.command("mt5-close-all")
@click.pass_context
def mt5_close_all(ctx):
    """🚨 Emergency: close ALL open MT5 positions.

    Immediately closes every open position on the MT5 account.
    Use this as a kill switch if something goes wrong.

    Example:
        python forex_main.py mt5-close-all
    """
    from forex_mt5_client import ForexMT5Client

    client = ForexMT5Client(ctx.obj["config"])
    if not client.connect():
        logger.error("Failed to connect to MT5")
        return

    try:
        positions = client.get_positions()
        if not positions:
            logger.info("No open positions to close")
            return

        logger.warning(f"Closing {len(positions)} open positions...")
        result = client.close_all_positions()

        closed = result.get("closed", [])
        total_pnl = result.get("total_pnl", 0)
        logger.info(f"Closed {len(closed)} positions | Total P&L: ${total_pnl:+,.2f}")

    finally:
        client.disconnect()


@cli.command("mt5-history")
@click.option("--days", default=7, help="Number of past days of history to display (default: 7)")
@click.pass_context
def mt5_history(ctx, days):
    """📜 Show closed trades and deals history from MT5/XM.

    Displays ticket numbers, symbols, buy/sell, lots, entry/exit prices,
    closed time, commissions, swaps, and realized profit/loss ($).

    Example:
        python forex_main.py mt5-history
        python forex_main.py mt5-history --days 30
    """
    from forex_mt5_client import ForexMT5Client

    client = ForexMT5Client(ctx.obj["config"])
    if not client.connect():
        logger.error("Failed to connect to MT5. Is the terminal running?")
        return

    try:
        deals = client.get_trade_history(days=days)
        if not deals:
            print(f"\nNo closed deals found in the past {days} days.")
            return

        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"📜 MT5 Trade History (Last {days} Days)", border_style="cyan")
            table.add_column("Ticket", style="dim")
            table.add_column("Time", style="cyan")
            table.add_column("Symbol", style="bold")
            table.add_column("Type", style="bold")
            table.add_column("Entry", justify="center")
            table.add_column("Lots", justify="right")
            table.add_column("Price", justify="right")
            table.add_column("Profit ($)", justify="right")
            table.add_column("Swap/Comm", justify="right")
            table.add_column("Comment", style="dim")

            total_profit = 0.0
            total_closed = 0
            for d in deals:
                pnl = d["profit"]
                pnl_color = "green" if pnl > 0 else ("red" if pnl < 0 else "dim")
                fees = d.get("commission", 0) + d.get("swap", 0)
                table.add_row(
                    str(d["ticket"]),
                    d["time"],
                    d["symbol"],
                    d["type"],
                    d["entry"],
                    f"{d['lots']:.2f}",
                    f"{d['price']:.5f}",
                    f"[{pnl_color}]${pnl:+.2f}[/]",
                    f"${fees:+.2f}",
                    d.get("comment", "") or "—",
                )
                if d["entry"] in ("OUT", "INOUT") or pnl != 0:
                    total_profit += pnl + fees
                    total_closed += 1

            console.print(table)
            profit_color = "green" if total_profit >= 0 else "red"
            console.print(f"  Total Realized P&L: [{profit_color}]${total_profit:+,.2f}[/] across {total_closed} closed deal(s)\n")

        except ImportError:
            print(f"\n=== MT5 Trade History (Last {days} Days) ===")
            for d in deals:
                print(f"  {d['time']} | {d['symbol']} {d['type']} {d['lots']:.2f} lots @ {d['price']:.5f} | "
                      f"P&L: ${d['profit']:+.2f} | {d['entry']}")

    finally:
        client.disconnect()


if __name__ == "__main__":
    cli()

