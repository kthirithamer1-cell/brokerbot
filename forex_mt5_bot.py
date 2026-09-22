"""
forex_mt5_bot.py — MT5/XM Forex Trading Bot Loop
==================================================
Variant of forex_bot.py that uses ForexMT5Client instead of ForexIBKRClient.
Reuses the same ML model, feature engine, risk manager, and data loader.

Runs 24/5 with session awareness (Tokyo/London/NY).
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import yaml

from forex_data_loader import ForexDataLoader
from forex_feature_engine import ForexFeatureEngine
from forex_ml_model import ForexMLModel
from forex_risk_manager import ForexRiskManager
from forex_mt5_client import ForexMT5Client
from forex_strategy_engine import ForexStrategyEngine

logger = logging.getLogger(__name__)


class ForexMT5Bot:
    """
    MT5/XM forex trading bot.

    Pipeline per cycle:
        1. Check session awareness (Tokyo/London/NY)
        2. Update market regime from DXY/VIX
        3. Compute currency strength across all pairs
        4. For each active pair:
            a. Fetch latest 15m data
            b. Compute technical + macro features
            c. ML model generates signal
            d. Risk manager validates (spread, margin, correlation)
            e. Execute order via MT5
        5. Monitor positions (trailing stops, time stops)
        6. Track daily P&L
    """

    def __init__(self, config_path: str = "forex_config.yaml", fixed_lots: Optional[float] = None):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize modules
        self.data_loader = ForexDataLoader(config_path)
        self.feature_engine = ForexFeatureEngine()
        self.ml_model = ForexMLModel(config_path)
        try:
            self.ml_model.load("forex_trading_model")
            logger.info("Forex ML model preloaded successfully.")
        except Exception as e:
            logger.warning(f"Forex ML model could not be preloaded: {e}")
        self.risk_manager = ForexRiskManager(config_path)
        if fixed_lots is not None and fixed_lots > 0:
            self.risk_manager.fixed_lots = fixed_lots
            logger.info(f"⚖️ Fixed lot size override active: {fixed_lots:.2f} lots")
        self.mt5_client = ForexMT5Client(config_path)
        self.strategy_engine = ForexStrategyEngine(config_path)

        # Trading config
        self.active_pairs = self.data_loader.get_active_pairs()
        self.confidence_threshold = self.config["strategy"].get("confidence_threshold", 0.42)
        self.timeframe = self.config["strategy"].get("timeframe", "15m")

        self._running = False
        self._last_regime_update = None
        self._last_strength_update = None
        self._currency_strength = {}
        self._correlation_matrix = None
        self._last_sl_time = {}

        # Logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def start(self):
        """Start the MT5 forex trading bot."""
        logger.info("=" * 60)
        logger.info("MT5/XM FOREX AI TRADING BOT — Starting")
        logger.info(f"   Pairs: {', '.join(self.active_pairs)}")
        logger.info(f"   Timeframe: {self.timeframe}")
        logger.info(f"   Ensemble: {self.config.get('model', {}).get('ensemble_method', 'single')}")
        logger.info("=" * 60)

        # Load trained model
        try:
            self.ml_model.load("forex_trading_model")
            logger.info("Forex ML model loaded")
        except FileNotFoundError:
            logger.error("No trained model found. Run 'python forex_main.py train' first.")
            return

        # Connect to MT5
        connected = self.mt5_client.connect()
        if not connected:
            logger.error("Cannot start bot without MT5 connection")
            return

        # Initialize equity
        equity = self.mt5_client.get_equity()
        self.risk_manager.update_equity(equity)
        logger.info(f"Account equity: ${equity:,.2f}")

        # Start trading loop
        self._running = True
        try:
            self._trading_loop()
        except KeyboardInterrupt:
            logger.info("\nBot stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stop the bot gracefully."""
        self._running = False
        logger.info("Shutting down MT5 bot...")

        summary = self.risk_manager.get_daily_summary()
        logger.info(f"Daily Summary: PnL=${summary['daily_pnl']:+,.2f} | "
                     f"Trades={summary['trades_today']} | "
                     f"Target={summary['target_progress_pct']:.0f}%")

        self.mt5_client.disconnect()
        logger.info("MT5 bot stopped")

    def _trading_loop(self):
        """Main 24/5 trading loop."""
        cycle = 0

        # Determine sleep interval based on timeframe
        if self.timeframe in ("1m", "5m"):
            sleep_seconds = 15
        elif self.timeframe == "15m":
            sleep_seconds = 30
        else:
            sleep_seconds = 60

        while self._running:
            cycle += 1
            now = datetime.utcnow()
            logger.info(f"\n{'─'*50}")
            logger.info(f"Cycle {cycle} — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

            # Check if trading is allowed (weekends, kill switch)
            allowed, reason = self.risk_manager.is_trading_allowed()
            if not allowed:
                logger.warning(f"Trading paused: {reason}")
                time.sleep(60)
                continue

            # Update equity from MT5
            equity = self.mt5_client.get_equity()
            self.risk_manager.update_equity(equity)

            # Log current session
            session = self._get_current_session()
            logger.info(f"Session: {session}")

            # Periodic updates (every 30 min)
            self._periodic_updates()

            # Check trailing stops on MT5 positions
            self._check_stops()

            # Process each pair
            for pair in self.active_pairs:
                try:
                    self._process_pair(pair)
                except Exception as e:
                    logger.error(f"Error processing {pair}: {e}")

            # Cycle summary
            summary = self.risk_manager.get_daily_summary()
            positions = self.mt5_client.get_positions()
            logger.info(f"Cycle {cycle} | PnL: ${summary['daily_pnl']:+.2f} "
                         f"({summary['target_progress_pct']:.0f}% of target) | "
                         f"MT5 Positions: {len(positions)} | "
                         f"Equity: ${equity:,.2f}")

            # Sleep based on timeframe
            logger.info(f"Sleeping {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    def _process_pair(self, pair: str):
        """Process a single forex pair: data → features → signal → trade."""
        # 1. Fetch live candles DIRECTLY from XM MT5 broker server
        df = self.mt5_client.get_bars(pair, timeframe_str=self.timeframe, count=300)
        if df.empty or len(df) < 50:
            lookback_period = "60d" if self.timeframe in ("15m", "5m") else "2y"
            df = self.data_loader.fetch_forex_data(
                pair, period=lookback_period, interval=self.timeframe, use_cache=False
            )
        if df.empty or len(df) < 50:
            return

        # Fetch macro context
        lookback_period = "60d" if self.timeframe in ("15m", "5m") else "2y"
        macro_df = self.data_loader.fetch_macro_context(
            period=lookback_period, interval=self.timeframe
        )

        # Compute features
        features = self.feature_engine.compute_features(
            df, pair=pair,
            macro_df=macro_df,
            currency_strength=self._currency_strength,
        )
        if features.empty:
            return

        # Get ML prediction for latest bar
        latest = features.iloc[[-1]]
        prediction = self.ml_model.predict(latest, confidence_threshold=self.confidence_threshold)

        confidence = float(prediction["confidence"].iloc[0])
        prob_buy = float(prediction.get("prob_buy", pd.Series([0.0])).iloc[0])
        prob_sell = float(prediction.get("prob_sell", pd.Series([0.0])).iloc[0])

        # 2. Get real-time live price directly from XM MT5 tick
        tick = self.mt5_client.get_tick(pair)
        if tick and tick.get("mid", 0) > 0:
            current_price = tick["mid"]
            bid_price = tick["bid"]
            ask_price = tick["ask"]
            spread_pips = tick["spread_pips"]
        else:
            current_price = float(df["Close"].iloc[-1])
            bid_price = current_price
            ask_price = current_price
            spread_pips = 0.0
        pip_size = self.data_loader.get_pip_size(pair)

        # Hybrid Decision Gate: Rule Strategy Trigger + ML Validation
        hybrid_cfg = self.config.get("hybrid", {})
        if hybrid_cfg.get("enabled", True):
            eval_df = features.copy()
            for col in ["Open", "High", "Low"]:
                if col in df.columns and col not in eval_df.columns:
                    eval_df[col] = df.loc[eval_df.index, col]

            rule_result = self.strategy_engine.evaluate_rules(eval_df, pair=pair)
            rule_signal = rule_result["signal"]
            rule_reason = rule_result["reason"]

            signal, hybrid_reason = self.strategy_engine.evaluate_hybrid_gate(
                rule_signal=rule_signal,
                rule_reason=rule_reason,
                ml_prediction=prediction.to_dict(orient="records")[0],
                currency_strength=self._currency_strength,
                pair=pair,
            )

            if signal == 0:
                logger.info(f"{pair:<7} | HOLD | {hybrid_reason} | XM Bid: {bid_price:.5f} Ask: {ask_price:.5f} (Spread: {spread_pips:.1f} pips)")
                return
            logger.info(f"✨ {pair:<7} | {hybrid_reason} | XM Bid: {bid_price:.5f} Ask: {ask_price:.5f}")
        else:
            signal = int(prediction["signal"].iloc[0])
            if signal == 0:
                logger.info(f"{pair:<7} | HOLD | Conf: {confidence*100:.1f}% | "
                             f"Buy: {prob_buy*100:.1f}% Sell: {prob_sell*100:.1f}% | "
                             f"XM Bid: {bid_price:.5f} Ask: {ask_price:.5f} (Spread: {spread_pips:.1f} pips)")
                return

        # Check post-SL cooldown
        cooldown_bars = self.config.get("risk", {}).get("cooldown_bars", 8)
        # Convert bars to seconds based on timeframe
        bar_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(self.timeframe, 3600)
        if pair in self._last_sl_time:
            sl_time, sl_dir = self._last_sl_time[pair]
            if (datetime.utcnow() - sl_time).total_seconds() < cooldown_bars * bar_seconds and signal == sl_dir:
                logger.info(f"  {pair}: Post-SL cooldown active ({cooldown_bars} bars)")
                return

        # Check regime / trend filters
        regime_cfg = self.config.get("regime", {})
        if regime_cfg.get("enabled", True):
            min_adx = regime_cfg.get("trend_strength_min_adx", 22)
            if "adx" in latest.columns and min_adx > 0:
                cur_adx = float(latest["adx"].iloc[0])
                if cur_adx < min_adx:
                    logger.info(f"  {pair}: ADX={cur_adx:.1f} < {min_adx} (ranging — skipped)")
                    return

            trend_ema = regime_cfg.get("trend_filter_ema", 200)
            if trend_ema:
                ema_col = f"ema_{trend_ema}"
                if ema_col in latest.columns:
                    ema_val = float(latest[ema_col].iloc[0])
                    if signal == 1 and current_price < ema_val:
                        logger.info(f"  {pair}: BUY blocked below {ema_col}")
                        return
                    elif signal == -1 and current_price > ema_val:
                        logger.info(f"  {pair}: SELL blocked above {ema_col}")
                        return

            # RSI pullback filter
            rsi_max_buy = self.config.get("risk", {}).get("rsi_max_buy", 65.0)
            rsi_min_sell = self.config.get("risk", {}).get("rsi_min_sell", 35.0)
            if "rsi_14" in latest.columns:
                cur_rsi = float(latest["rsi_14"].iloc[0])
                if signal == 1 and cur_rsi > rsi_max_buy:
                    logger.info(f"  {pair}: BUY blocked by high RSI ({cur_rsi:.1f} > {rsi_max_buy})")
                    return
                elif signal == -1 and cur_rsi < rsi_min_sell:
                    logger.info(f"  {pair}: SELL blocked by low RSI ({cur_rsi:.1f} < {rsi_min_sell})")
                    return

        # Check current spread on MT5
        current_spread = self.mt5_client.get_spread_pips(pair)
        max_spread = self.config.get("risk", {}).get("max_entry_spread_pips", 3.0)
        if current_spread > max_spread:
            logger.info(f"  {pair}: Spread {current_spread:.1f} pips > {max_spread} max — skipped")
            return

        # ATR for stops
        atr = df["Close"].pct_change().rolling(14).std().iloc[-1] * current_price

        logger.info(f"{pair:<7} | {'BUY' if signal == 1 else 'SELL'} | "
                     f"Conf: {confidence*100:.1f}% | Price: {current_price:.5f} | "
                     f"Spread: {current_spread:.1f} pips | "
                     f"Regime: {self.risk_manager._current_regime}")

        # Execute trade
        self._execute_trade(pair, current_price, atr, signal)

    def _execute_trade(self, pair: str, price: float, atr: float, direction: int):
        """Execute a forex trade via MT5."""
        # Compute bracket levels
        bracket = self.risk_manager.compute_bracket_levels(pair, price, atr, direction)

        # Get session multiplier
        session_mult = self.risk_manager.get_session_multiplier(pair)

        # Position sizing
        sizing = self.risk_manager.calculate_position_size(
            pair, price, bracket["stop_loss"], session_mult
        )

        if sizing["units"] < 1000:  # Less than 1 micro lot
            logger.info(f"  {pair}: Position too small ({sizing['units']} units)")
            return

        # Validate trade
        valid, reason = self.risk_manager.validate_trade(
            pair, price, bracket["stop_loss"], sizing["units"], direction,
            current_spread_pips=self.mt5_client.get_spread_pips(pair),
            correlation_matrix=self._correlation_matrix,
        )
        if not valid:
            logger.info(f"  {pair}: {reason}")
            return

        # Place bracket order via MT5
        action = "BUY" if direction == 1 else "SELL"
        result = self.mt5_client.place_bracket_order(
            pair=pair, units=sizing["units"],
            entry_price=round(price, 5),
            stop_loss=bracket["stop_loss"],
            take_profit=bracket["take_profit"],
            action=action,
        )

        if "error" not in result:
            self.risk_manager.register_open(pair, price, sizing["units"], bracket["stop_loss"], direction)
            logger.info(f"  {action} {sizing['lots']:.2f} lots {pair} @ {price:.5f} | "
                         f"SL: {bracket['stop_loss']:.5f} ({bracket['sl_pips']} pips) | "
                         f"TP: {bracket['take_profit']:.5f} ({bracket['tp_pips']} pips) | "
                         f"R:R={bracket['risk_reward_ratio']} | "
                         f"Risk: ${sizing['risk_amount']:.2f}")
        else:
            logger.error(f"  Order failed: {result['error']}")

    def _check_stops(self):
        """Check MT5 positions and update internal tracking."""
        positions = self.mt5_client.get_positions()

        # Log open positions
        for pos in positions:
            if pos.get("pnl", 0) != 0:
                logger.debug(f"  {pos['pair']} {pos['direction']} {pos['lots']} lots | "
                              f"P&L: ${pos['pnl']:+.2f} | Swap: ${pos['swap']:.2f}")

    def _periodic_updates(self):
        """Update regime, currency strength, correlations periodically."""
        now = datetime.utcnow()

        # Regime update every 30 minutes
        if self._last_regime_update is None or now - self._last_regime_update > timedelta(minutes=30):
            self._update_regime()
            self._last_regime_update = now

        # Currency strength every 60 minutes
        if self._last_strength_update is None or now - self._last_strength_update > timedelta(minutes=60):
            try:
                self._currency_strength = self.data_loader.compute_currency_strength(
                    period="30d", interval="1d", lookback_bars=5
                )
                self._correlation_matrix = self.data_loader.compute_correlation_matrix(
                    self.active_pairs, period="30d", interval="1h"
                )
            except Exception as e:
                logger.debug(f"Currency strength update failed: {e}")
            self._last_strength_update = now

    def _update_regime(self):
        """Detect market regime from DXY/VIX."""
        try:
            import ta as ta_lib

            # Use DXY as primary regime indicator for forex
            dxy_df = self.data_loader.fetch_forex_data("DX-Y.NYB", period="3mo", interval="1d")
            if dxy_df.empty or len(dxy_df) < 50:
                return

            adx = ta_lib.trend.ADXIndicator(
                dxy_df["High"], dxy_df["Low"], dxy_df["Close"]
            ).adx().iloc[-1]

            atr = ta_lib.volatility.average_true_range(
                dxy_df["High"], dxy_df["Low"], dxy_df["Close"], window=14
            )
            atr_pct = atr / dxy_df["Close"]
            vol_percentile = atr_pct.rank(pct=True).iloc[-1] * 100

            regime_cfg = self.config.get("regime", {})
            high_vol = regime_cfg.get("high_vol_threshold", 75)

            if vol_percentile > high_vol:
                regime = "volatile"
            elif adx > 25:
                regime = "trending"
            elif adx < 15:
                regime = "ranging"
            else:
                regime = "medium"

            self.risk_manager.update_regime(regime)
            logger.info(f"Market regime: {regime.upper()} (DXY ADX={adx:.1f}, Vol%={vol_percentile:.0f})")

        except Exception as e:
            logger.debug(f"Regime detection failed: {e}")

    @staticmethod
    def _get_current_session() -> str:
        """Get current forex session name."""
        hour = datetime.utcnow().hour
        sessions = []
        if hour >= 21 or hour < 6:
            sessions.append("Sydney")
        if 0 <= hour < 9:
            sessions.append("Tokyo")
        if 7 <= hour < 16:
            sessions.append("London")
        if 12 <= hour < 21:
            sessions.append("New York")
        if 12 <= hour < 16:
            sessions.append("(Overlap)")
        return " + ".join(sessions) if sessions else "Off-hours"


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    bot = ForexMT5Bot()
    print(f"\nMT5 Forex Trading Bot initialized.")
    print(f"  Active pairs: {bot.active_pairs}")
    print(f"  Timeframe: {bot.timeframe}")
    print(f"  Session: {bot._get_current_session()}")
    print(f"\nTo start trading: bot.start()")
    print(f"Train a model first: python forex_main.py train")
