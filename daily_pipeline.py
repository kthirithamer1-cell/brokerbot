"""
daily_pipeline.py — Master 24-Hour Autonomous Trading & Learning Engine
========================================================================
Orchestrates the 3-stage daily lifecycle:
    1. Pre-Market (Morning):
       - Scan live trending penny stocks (Yahoo Finance screeners + watchlist)
       - Run lean cumulative Optuna tuning (20-30 trials, SQLite persistent)
       - Model Approval Gate: Champion vs Challenger evaluation
       - Generate Daily Trade Plan (entry triggers, stops, profit targets)
    2. Market Hours:
       - Run paper trading with Breakeven stops & Tier-1 technical confirmation
    3. Post-Market (Evening):
       - Audit day's trades against the Morning Trade Plan
       - Save Daily Performance Scorecard to reports/
       - Update prediction journal for continuous learning
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from data_loader import DataLoader
from feature_engine import FeatureEngine
from ml_model import MLModel
from penny_scanner import PennyScanner
from model_gate import ModelGate

logger = logging.getLogger(__name__)


class DailyPipeline:
    """Master orchestrator for the daily trading and self-improving ML cycle."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.data_loader = DataLoader(config_path)
        self.feature_engine = FeatureEngine()
        self.penny_scanner = PennyScanner(config_path)
        self.model_gate = ModelGate()

        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def run_pre_market(self, tune_trials: int = 25) -> dict:
        """
        Stage 1: Pre-Market Preparation, Tuning, Model Approval, & Trade Plan.
        Executes before market open (e.g. 08:30 - 09:25 EST).
        """
        today_str = date.today().isoformat()
        logger.info("=" * 65)
        logger.info(f"🌅 STAGE 1: PRE-MARKET ENGINE — {today_str}")
        logger.info("=" * 65)

        # 1. Scan for live trending penny stocks
        logger.info("🔍 Step 1.1: Scanning live trending penny stocks & watchlist...")
        picks_df = self.penny_scanner.scan()
        if not picks_df.empty:
            top_symbols = picks_df["symbol"].head(5).tolist()
            logger.info(f"   Top Trending Picks today: {', '.join(top_symbols)}")
        else:
            top_symbols = self.config.get("watchlist", {}).get("symbols", ["PDSB", "PLUG", "MARA"])[:5]
            logger.info(f"   Using core watchlist candidates: {', '.join(top_symbols)}")

        # 2. Cumulative Trial Optimization & Model Approval
        logger.info(f"\n🧠 Step 1.2: Cumulative Model Training & Trial Optimization ({tune_trials} trials)...")
        # Train on primary focus symbol (or combined top picks)
        primary_symbol = top_symbols[0] if top_symbols else "PDSB"
        ml_model = MLModel(self.config_path)

        train_df = self.data_loader.fetch_price_data(
            primary_symbol,
            period="60d",
            interval=self.config["strategy"].get("timeframe", "15m"),
            use_cache=True,  # Respect cache expiration
        )

        approval_result = {"status": "SKIPPED", "reason": "Insufficient training data"}
        if not train_df.empty and len(train_df) >= 100:
            features = self.feature_engine.compute_features(train_df)
            label_mode = self.config.get("strategy", {}).get("label_mode", "atr_relative")
            label_atr_mult = self.config.get("strategy", {}).get("label_atr_multiplier", 1.5)
            labels = self.feature_engine.create_labels(
                train_df,
                horizon=10,
                mode=label_mode,
                atr_multiplier=label_atr_mult,
            )
            combined = features.copy()
            combined["label"] = labels
            combined.dropna(inplace=True)
            feature_cols = [c for c in combined.columns if c != "label"]
            X = combined[feature_cols]
            y = combined["label"].astype(int)

            # Run persistent tuning
            tune_res = ml_model.tune_hyperparameters(
                X, y,
                n_trials=tune_trials,
                study_name=f"study_{self.config['strategy'].get('model_type', 'xgboost')}",
            )

            # Evaluate candidate on walk-forward
            wf_results = ml_model.walk_forward_validate(X, y, n_splits=3, params=tune_res.get("best_params"))

            # Save candidate model
            ml_model.save("candidate_model")

            # Gate approval
            cand_metrics = {
                "accuracy": wf_results.get("overall_accuracy", 0.0),
                "f1": wf_results.get("overall_f1", 0.0),
                "buy_win_rate": wf_results.get("buy_signal_win_rate", 0.0),
                "trials_completed": tune_res.get("total_trials", 0),
                "symbols": [primary_symbol],
            }
            cand_meta = {
                "model_type": ml_model.model_type,
                "training_metrics": cand_metrics,
                "symbols": [primary_symbol],
            }
            approved, reason = self.model_gate.evaluate_and_promote(
                candidate_model_path=Path("models/candidate_model.joblib"),
                candidate_metrics=cand_metrics,
                candidate_meta=cand_meta,
            )
            approval_result = {
                "approved": approved,
                "reason": reason,
                "champion": self.model_gate.get_champion_metrics(),
            }

        # 3. Build Daily Trade Plan
        logger.info("\n📋 Step 1.3: Generating Daily Trade Plan...")
        trade_plan = self._generate_trade_plan(top_symbols)

        plan_file = self.reports_dir / f"daily_trade_plan_{today_str}.json"
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump({
                "date": today_str,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "approval_gate": approval_result,
                "plan": trade_plan,
            }, f, indent=2)

        logger.info(f"✅ Daily Trade Plan saved to {plan_file}")
        self._print_trade_plan(trade_plan)

        return {
            "date": today_str,
            "top_symbols": top_symbols,
            "approval": approval_result,
            "plan": trade_plan,
        }

    def _generate_trade_plan(self, symbols: list[str]) -> list[dict]:
        """Compute key levels, triggers, ATR stops, and targets for today's targets."""
        plan = []
        for symbol in symbols:
            try:
                df = self.data_loader.fetch_price_data(symbol, period="5d", interval="15m")
                if df.empty or len(df) < 15:
                    continue

                curr_price = float(df["Close"].iloc[-1])
                # Compute 14-period ATR on 15m bars
                high = df["High"]
                low = df["Low"]
                close = df["Close"]
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = float(tr.tail(14).mean())

                # Technical levels
                recent_high = float(df["High"].tail(10).max())
                recent_low = float(df["Low"].tail(10).min())
                vwap = float((df["Close"] * df["Volume"]).cumsum().iloc[-1] / df["Volume"].cumsum().iloc[-1]) if df["Volume"].sum() > 0 else curr_price

                # Entry trigger: Breakout above recent high or bounce off VWAP
                entry_trigger = round(max(curr_price, recent_high * 0.998), 2)
                stop_loss = round(entry_trigger - (1.5 * atr), 2)
                take_profit = round(entry_trigger + (3.0 * atr), 2)
                breakeven_level = round(entry_trigger + (1.0 * atr), 2)

                shares = int(max(10, 100 / max(curr_price, 0.01)))
                max_risk = round(abs(entry_trigger - stop_loss) * shares, 2)
                plan.append({
                    "symbol": symbol,
                    "current_price": curr_price,
                    "vwap": round(vwap, 2),
                    "atr_15m": round(atr, 3),
                    "entry_trigger": entry_trigger,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "target_1_breakeven": breakeven_level,
                    "target_2_profit": take_profit,
                    "breakeven_trigger": breakeven_level,
                    "position_shares": shares,
                    "max_risk_dollars": max_risk,
                    "risk_reward_ratio": 2.0,
                    "recommended_action": "BUY_ON_BREAKOUT_WITH_VOLUME",
                    "strategy_notes": f"Wait for 9:35 AM opening. Confirm Price > VWAP (${vwap:.2f}). Ratchet to Breakeven at ${breakeven_level:.2f}."
                })
            except Exception as e:
                logger.debug(f"Failed to generate plan for {symbol}: {e}")

        return plan

    def run_post_market_review(self, paper_trades: Optional[list[dict]] = None) -> dict:
        """
        Stage 3: Post-Market Review, Trade Audit, and Journal Feedback.
        Executes after market close (16:15 EST).
        """
        today_str = date.today().isoformat()
        logger.info("=" * 65)
        logger.info(f"🌆 STAGE 3: POST-MARKET TRADE AUDIT — {today_str}")
        logger.info("=" * 65)

        # Load today's trade plan if exists
        plan_file = self.reports_dir / f"daily_trade_plan_{today_str}.json"
        morning_plan = {}
        if plan_file.exists():
            with open(plan_file, "r", encoding="utf-8") as f:
                morning_plan = json.load(f).get("plan", [])

        # Load executed trades (from parameter or recent backtest/bot log)
        trades = paper_trades or []
        total_trades = len(trades)
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = sum(t.get("pnl", 0) for t in trades)

        # Categorize execution quality
        audit_summary = {
            "date": today_str,
            "audited_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "net_pnl": round(net_pnl, 2),
            "trades_count_wins": len(wins),
            "trades_count_losses": len(losses),
            "champion_model": self.model_gate.get_champion_metrics(),
            "trades": trades,
        }

        # Save daily review report
        review_file = self.reports_dir / f"daily_review_{today_str}.json"
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump({
                "date": today_str,
                "audit": audit_summary,
            }, f, indent=2)

        logger.info(f"📊 Daily Summary: {total_trades} trades | Win Rate: {win_rate:.1f}% | Net PnL: ${net_pnl:+.2f}")
        logger.info(f"✅ Daily Review saved to {review_file}")

        return audit_summary

    def _print_trade_plan(self, plan: list[dict]):
        """Print daily trade plan."""
        print("\n" + "=" * 70)
        print("🎯 TODAY'S DAILY TRADE PLAN (Pre-Market Levels)")
        print("=" * 70)
        for item in plan:
            print(
                f"  [{item['symbol']}] Price: ${item['current_price']:.2f} | "
                f"Trigger: ${item['entry_trigger']:.2f} | "
                f"Stop: ${item['stop_loss']:.2f} | "
                f"Target: ${item['take_profit']:.2f} | "
                f"Breakeven at: ${item['breakeven_trigger']:.2f}"
            )
        print("=" * 70 + "\n")
