"""
dashboard.py — Trading Bot Web Dashboard
==========================================
Flask web server that serves a premium dashboard UI for the
IBKR AI Trading Bot. Reads existing data files (JSON, CSV, Parquet)
from reports/, models/, logs/, and data/ directories.

Launch:  python main.py dashboard
"""

import json
import glob
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from flask import Flask, jsonify, send_from_directory, request

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="dashboard", static_url_path="/static")


def create_app(config_path: str = "config.yaml"):
    """Create and configure the Flask dashboard app."""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    app.config["BOT_CONFIG"] = config
    app.config["CONFIG_PATH"] = config_path

    # ──────────────────────────────────────────────
    #  SERVE DASHBOARD
    # ──────────────────────────────────────────────

    @app.route("/")
    def index():
        return send_from_directory("dashboard", "index.html")

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        return send_from_directory("dashboard", filename)

    # ──────────────────────────────────────────────
    #  API: MODEL INFO
    # ──────────────────────────────────────────────

    @app.route("/api/model")
    def api_model():
        """Return model metadata, training metrics, feature importance."""
        meta_path = Path("models/trading_model_meta.json")
        if not meta_path.exists():
            return jsonify({"error": "No trained model found"}), 404

        with open(meta_path) as f:
            meta = json.load(f)

        return jsonify(meta)

    # ──────────────────────────────────────────────
    #  API: BACKTEST LIST
    # ──────────────────────────────────────────────

    @app.route("/api/backtest/list")
    def api_backtest_list():
        """List all backtest runs with key metrics."""
        metrics_files = sorted(
            glob.glob("reports/backtest_metrics_*.json"),
            reverse=True
        )

        runs = []
        for mf in metrics_files[:50]:  # Limit to last 50
            try:
                with open(mf) as f:
                    data = json.load(f)

                # Extract timestamp from filename: backtest_metrics_YYYYMMDD_HHMMSS.json
                basename = Path(mf).stem
                parts = basename.replace("backtest_metrics_", "")
                ts_str = parts  # e.g. "20260909_202424"

                runs.append({
                    "timestamp": ts_str,
                    "total_trades": data.get("total_trades", 0),
                    "win_rate_pct": data.get("win_rate_pct", 0),
                    "total_return_pct": data.get("total_return_pct", 0),
                    "total_pnl": data.get("total_pnl", 0),
                    "sharpe_ratio": data.get("sharpe_ratio", 0),
                    "max_drawdown_pct": data.get("max_drawdown_pct", 0),
                    "initial_capital": data.get("initial_capital", 0),
                    "final_equity": data.get("final_equity", 0),
                })
            except Exception:
                continue

        return jsonify(runs)

    # ──────────────────────────────────────────────
    #  API: BACKTEST DETAIL
    # ──────────────────────────────────────────────

    @app.route("/api/backtest/<timestamp>")
    def api_backtest_detail(timestamp):
        """Return full backtest details: metrics, equity curve, trade log."""
        metrics_path = Path(f"reports/backtest_metrics_{timestamp}.json")
        equity_path = Path(f"reports/backtest_equity_{timestamp}.csv")
        trades_path = Path(f"reports/backtest_trades_{timestamp}.csv")

        if not metrics_path.exists():
            return jsonify({"error": "Backtest not found"}), 404

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Equity curve
        equity = []
        if equity_path.exists():
            try:
                df = pd.read_csv(equity_path)
                equity = df.to_dict(orient="records")
            except Exception:
                pass

        # Trades
        trades = []
        if trades_path.exists():
            try:
                df = pd.read_csv(trades_path)
                trades = df.to_dict(orient="records")
            except Exception:
                pass

        return jsonify({
            "metrics": metrics,
            "equity": equity,
            "trades": trades,
        })

    # ──────────────────────────────────────────────
    #  API: PENNY PICKS
    # ──────────────────────────────────────────────

    @app.route("/api/penny-picks")
    def api_penny_picks():
        """Return the latest penny stock picks."""
        pick_files = sorted(glob.glob("reports/penny_picks_*.json"), reverse=True)

        if not pick_files:
            return jsonify({"date": None, "picks": []})

        # Load the most recent
        latest = pick_files[0]
        date_str = Path(latest).stem.replace("penny_picks_", "")

        with open(latest) as f:
            picks = json.load(f)

        return jsonify({"date": date_str, "picks": picks})

    # ──────────────────────────────────────────────
    #  API: MODEL HEALTH
    # ──────────────────────────────────────────────

    @app.route("/api/health")
    def api_health():
        """Return model health status + prediction journal."""
        health_path = Path("logs/model_health.json")
        journal_path = Path("logs/prediction_journal.jsonl")

        health = {}
        if health_path.exists():
            with open(health_path) as f:
                health = json.load(f)

        journal = []
        if journal_path.exists():
            try:
                with open(journal_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            journal.append(json.loads(line))
                journal = journal[-50:]  # Last 50 entries
            except Exception:
                pass

        return jsonify({"health": health, "journal": journal})

    # ──────────────────────────────────────────────
    #  API: CONFIGURATION
    # ──────────────────────────────────────────────

    @app.route("/api/config")
    def api_config():
        """Return current bot configuration."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return jsonify(cfg)

    # ──────────────────────────────────────────────
    #  API: PRICE DATA
    # ──────────────────────────────────────────────

    @app.route("/api/price/<symbol>")
    def api_price(symbol):
        """Return cached OHLCV data for a symbol."""
        symbol = symbol.upper()
        interval = request.args.get("interval", "15m")

        # Find matching parquet file
        pattern = f"data/{symbol}_{interval}_*.parquet"
        files = sorted(glob.glob(pattern), reverse=True)

        if not files:
            # Try without interval
            pattern = f"data/{symbol}_*.parquet"
            files = sorted(glob.glob(pattern), reverse=True)

        if not files:
            return jsonify({"error": f"No data for {symbol}"}), 404

        try:
            df = pd.read_parquet(files[0])
            # Convert to JSON-serializable format
            df.index = df.index.astype(str)
            records = []
            for idx, row in df.iterrows():
                records.append({
                    "date": str(idx),
                    "open": round(float(row.get("Open", 0)), 4),
                    "high": round(float(row.get("High", 0)), 4),
                    "low": round(float(row.get("Low", 0)), 4),
                    "close": round(float(row.get("Close", 0)), 4),
                    "volume": int(row.get("Volume", 0)),
                })

            return jsonify({
                "symbol": symbol,
                "interval": interval,
                "source_file": Path(files[0]).name,
                "bars": len(records),
                "data": records[-500:],  # Limit to last 500 bars
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ──────────────────────────────────────────────
    #  API: WATCHLIST + LIVE PRICES
    # ──────────────────────────────────────────────

    @app.route("/api/watchlist")
    def api_watchlist():
        """Return watchlist symbols with cached price info."""
        symbols = config.get("watchlist", {}).get("symbols", [])

        # Also gather all symbols from data/ directory
        data_files = glob.glob("data/*_15m_*.parquet")
        data_symbols = set()
        for f in data_files:
            name = Path(f).stem
            sym = name.split("_")[0]
            data_symbols.add(sym)

        return jsonify({
            "watchlist": symbols,
            "available_symbols": sorted(data_symbols),
        })

    # ──────────────────────────────────────────────
    #  API: OVERVIEW STATS
    # ──────────────────────────────────────────────

    @app.route("/api/overview")
    def api_overview():
        """Return aggregated overview statistics for the dashboard header."""
        # Model info
        model_info = {"status": "not_trained", "accuracy": 0, "f1": 0, "trained_at": None}
        meta_path = Path("models/trading_model_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            tm = meta.get("training_metrics", {})
            wf = tm.get("walk_forward", {})
            model_info = {
                "status": "trained",
                "accuracy": wf.get("overall_accuracy", tm.get("train_accuracy", 0)),
                "f1": wf.get("overall_f1", tm.get("train_f1", 0)),
                "buy_win_rate": wf.get("buy_signal_win_rate", 0),
                "n_samples": tm.get("n_samples", 0),
                "n_features": tm.get("n_features_selected", 0),
                "ensemble": meta.get("ensemble_method", "single"),
                "trained_at": meta.get("saved_at", ""),
            }

        # Latest backtest
        backtest_info = {"status": "none"}
        metrics_files = sorted(glob.glob("reports/backtest_metrics_*.json"), reverse=True)
        if metrics_files:
            with open(metrics_files[0]) as f:
                bt = json.load(f)
            ts = Path(metrics_files[0]).stem.replace("backtest_metrics_", "")
            backtest_info = {
                "status": "available",
                "timestamp": ts,
                "total_trades": bt.get("total_trades", 0),
                "win_rate": bt.get("win_rate_pct", 0),
                "total_return": bt.get("total_return_pct", 0),
                "total_pnl": bt.get("total_pnl", 0),
                "sharpe": bt.get("sharpe_ratio", 0),
                "max_drawdown": bt.get("max_drawdown_pct", 0),
            }

        # Penny picks count
        pick_files = sorted(glob.glob("reports/penny_picks_*.json"), reverse=True)
        penny_info = {"count": 0, "date": None}
        if pick_files:
            with open(pick_files[0]) as f:
                picks = json.load(f)
            penny_info = {
                "count": len(picks),
                "date": Path(pick_files[0]).stem.replace("penny_picks_", ""),
            }

        # Model health
        health_info = {"status": "unknown"}
        health_path = Path("logs/model_health.json")
        if health_path.exists():
            with open(health_path) as f:
                health_info = json.load(f)

        # Config summary
        capital = config.get("backtest", {}).get("initial_capital", 30)
        daily_target = config.get("targets", {}).get("daily_profit_target", 5)

        return jsonify({
            "model": model_info,
            "backtest": backtest_info,
            "penny_picks": penny_info,
            "health": health_info,
            "capital": capital,
            "daily_target": daily_target,
            "total_backtests": len(metrics_files),
            "total_data_files": len(glob.glob("data/*.parquet")),
        })

    # ──────────────────────────────────────────────
    #  API: DAILY TRADE PLAN & REVIEW
    # ──────────────────────────────────────────────

    @app.route("/api/daily-plan")
    def api_daily_plan():
        """Return the latest daily pre-market trade plan."""
        plan_files = sorted(glob.glob("reports/daily_trade_plan_*.json"), reverse=True)
        if not plan_files:
            return jsonify({"date": None, "plan": [], "approval_gate": {}})

        latest = plan_files[0]
        with open(latest, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        return jsonify(plan_data)

    @app.route("/api/daily-review")
    def api_daily_review():
        """Return the latest daily post-market review."""
        review_files = sorted(glob.glob("reports/daily_review_*.json"), reverse=True)
        if not review_files:
            return jsonify({"date": None, "audit": {}})

        latest = review_files[0]
        with open(latest, "r", encoding="utf-8") as f:
            review_data = json.load(f)
        return jsonify(review_data)

    # ──────────────────────────────────────────────
    #  API: MODEL APPROVAL GATE
    # ──────────────────────────────────────────────

    @app.route("/api/model-gate")
    def api_model_gate():
        """Return ModelGate registry (Champion vs Challenger status, cumulative trials)."""
        from model_gate import ModelGate
        gate = ModelGate()
        return jsonify(gate.registry)

    return app
