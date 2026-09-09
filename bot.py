"""
bot.py — Main Trading Bot Loop
================================
Orchestrates the trading pipeline: data → features → ML prediction
→ risk check → order execution. Runs in paper or live mode.

Upgraded with:
    - Multi-timeframe signal confirmation
    - Market session awareness
    - Regime-aware risk parameters
    - Daily P&L target integration
    - Model health monitoring
    - Cash account enforcement (no shorts, settled cash, day-trade tracking)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from data_loader import DataLoader
from feature_engine import FeatureEngine
from sentiment import SentimentAnalyzer
from ml_model import MLModel
from risk_manager import RiskManager
from ibkr_client import IBKRClient

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot that ties all modules together.

    Pipeline per cycle:
        1. Fetch latest price data
        2. Compute technical features + regime detection
        3. Fetch & score news sentiment
        4. Multi-timeframe signal confirmation
        5. ML model generates signals
        6. Risk manager validates trade (regime-aware)
        7. IBKR client executes order
        8. Monitor positions, trailing stops, & time stops
        9. Track daily P&L toward $5 target
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize modules
        self.data_loader = DataLoader(config_path)
        self.feature_engine = FeatureEngine()
        self.sentiment_analyzer = SentimentAnalyzer(
            self.config["sentiment"].get("model", "ProsusAI/finbert")
        )
        self.ml_model = MLModel(config_path)
        self.risk_manager = RiskManager(config_path)
        self.ibkr_client = IBKRClient(config_path)

        self.watchlist = self.config.get("watchlist", {}).get("symbols", [])
        self.confidence_threshold = self.config["strategy"].get("confidence_threshold", 0.60)
        self.timeframe = self.config["strategy"].get("timeframe", "1d")

        # Session config
        session_cfg = self.config.get("session", {})
        self.avoid_first_minutes = session_cfg.get("avoid_first_minutes", 5)

        self._running = False

        # Logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Cache for higher timeframe data (refreshed less frequently)
        self._htf_cache = {}
        self._htf_last_refresh = None
        self._htf_refresh_interval = timedelta(minutes=30)  # Refresh every 30 min

    def start(self, paper: bool = True):
        """
        Start the trading bot.

        Args:
            paper: If True, use paper trading port (default)
        """
        logger.info("=" * 60)
        logger.info("🤖 IBKR AI TRADING BOT — Starting")
        logger.info(f"   Mode: {'📝 PAPER' if paper else '💰 LIVE'}")
        logger.info(f"   Watchlist: {', '.join(self.watchlist)}")
        logger.info(f"   Timeframe: {self.timeframe}")
        logger.info(f"   Ensemble: {self.config.get('model', {}).get('ensemble_method', 'single')}")
        logger.info(f"   Daily target: ${self.config.get('targets', {}).get('daily_profit_target', 5.0)}")
        logger.info("=" * 60)

        # Load trained model
        try:
            self.ml_model.load("trading_model")
            logger.info("✅ ML model loaded")
        except FileNotFoundError:
            logger.error("❌ No trained model found. Run 'python main.py train' first.")
            return

        # Connect to IBKR
        if paper:
            self.ibkr_client.port = 7497  # TWS paper
        connected = self.ibkr_client.connect()
        if not connected:
            logger.error("❌ Cannot start bot without IBKR connection")
            return

        # Initialize equity
        equity = self.ibkr_client.get_equity()
        self.risk_manager.update_equity(equity)
        logger.info(f"💰 Account equity: ${equity:,.2f}")

        # Start trading loop
        self._running = True
        try:
            self._trading_loop()
        except KeyboardInterrupt:
            logger.info("\n⛔ Bot stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stop the trading bot gracefully."""
        self._running = False
        logger.info("Shutting down...")

        # Print daily summary
        summary = self.risk_manager.get_daily_summary()
        logger.info(f"📋 Daily Summary: PnL=${summary['daily_pnl']:+,.2f} "
                     f"({summary['daily_pnl_pct']:+.2f}%), "
                     f"Trades={summary['trades_today']}, "
                     f"Target={summary['target_progress_pct']:.0f}%")

        # Print model health
        health = self.ml_model.get_model_health()
        if health.get("status") != "insufficient_data":
            logger.info(f"🧠 Model Health: {health['status']} "
                         f"(accuracy={health.get('rolling_accuracy', 'N/A')}, "
                         f"decay={health.get('accuracy_decay_pct', 'N/A')}%)")

        self.ibkr_client.disconnect()
        logger.info("🛑 Bot stopped")

    def _trading_loop(self):
        """Main trading loop."""
        cycle = 0

        while self._running:
            cycle += 1
            logger.info(f"\n{'─'*50}")
            logger.info(f"📍 Cycle {cycle} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Update equity
            equity = self.ibkr_client.get_equity()
            self.risk_manager.update_equity(equity)

            # Update cash info for cash account enforcement
            if self.ibkr_client.is_cash_account():
                settled_cash = self.ibkr_client.get_settled_cash()
                buying_power = self.ibkr_client.get_buying_power()
                self.risk_manager.update_cash_info(settled_cash, buying_power)
                logger.debug(
                    f"💵 Cash account: settled=${settled_cash:.2f}, "
                    f"buying_power=${buying_power:.2f}"
                )

            # Check if trading is allowed
            allowed, reason = self.risk_manager.is_trading_allowed()
            if not allowed:
                logger.warning(f"⚠️ Trading paused: {reason}")
                time.sleep(60)
                continue

            # Refresh higher timeframe data periodically
            self._refresh_higher_timeframe_data()

            # Update market regime
            self._update_market_regime()

            # Check trailing stops on existing positions
            self._check_stops()

            # Scan watchlist for signals
            for symbol in self.watchlist:
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue

            # Log cycle summary
            summary = self.risk_manager.get_daily_summary()
            cash_info = ""
            if summary.get("account_type") == "cash":
                cash_info = (
                    f" | Settled: ${summary.get('settled_cash', 0):.2f}"
                    f" | DayTrades: {summary.get('day_trades_used', 0)}/{summary.get('day_trades_limit', 3)}"
                )
            logger.info(f"💰 Cycle {cycle} | PnL: ${summary['daily_pnl']:+.2f} "
                         f"({summary['target_progress_pct']:.0f}% of target) | "
                         f"Positions: {summary['open_positions']}{cash_info}")

            # Determine sleep interval based on timeframe
            if self.timeframe == "1d":
                sleep_seconds = 300  # 5 minutes
            elif self.timeframe in ("1h", "1H"):
                sleep_seconds = 60
            else:
                sleep_seconds = 30

            logger.info(f"💤 Sleeping {sleep_seconds}s until next cycle...")
            time.sleep(sleep_seconds)

    def _refresh_higher_timeframe_data(self):
        """
        Fetch higher-timeframe data (daily) for multi-timeframe confirmation.
        Cached and refreshed every 30 minutes to avoid excess API calls.
        """
        now = datetime.now()
        if (self._htf_last_refresh and
                now - self._htf_last_refresh < self._htf_refresh_interval):
            return  # Still fresh

        logger.debug("Refreshing higher-timeframe data...")

        for symbol in self.watchlist:
            try:
                daily_df = self.data_loader.fetch_price_data(
                    symbol, period="3mo", interval="1d", use_cache=True
                )
                if daily_df.empty or len(daily_df) < 20:
                    continue

                # Compute daily trend context
                import ta as ta_lib
                close = daily_df["Close"]

                ema_9 = ta_lib.trend.ema_indicator(close, window=9).iloc[-1]
                ema_21 = ta_lib.trend.ema_indicator(close, window=21).iloc[-1]
                daily_rsi = ta_lib.momentum.rsi(close, window=14).iloc[-1]
                weekly_momentum = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0

                self._htf_cache[symbol] = {
                    "daily_ema_trend": 1 if ema_9 > ema_21 else -1,
                    "daily_rsi": daily_rsi,
                    "weekly_momentum": weekly_momentum,
                    "daily_close": close.iloc[-1],
                }

            except Exception as e:
                logger.debug(f"HTF data failed for {symbol}: {e}")

        self._htf_last_refresh = now

    def _update_market_regime(self):
        """
        Detect current market regime from SPY/QQQ and update risk manager.
        """
        try:
            spy_df = self.data_loader.fetch_price_data(
                "SPY", period="3mo", interval="1d", use_cache=True
            )
            if spy_df.empty or len(spy_df) < 50:
                return

            import ta as ta_lib

            # ADX for trend strength
            adx = ta_lib.trend.ADXIndicator(
                spy_df["High"], spy_df["Low"], spy_df["Close"]
            ).adx().iloc[-1]

            # ATR percentile for volatility regime
            atr = ta_lib.volatility.average_true_range(
                spy_df["High"], spy_df["Low"], spy_df["Close"], window=14
            )
            atr_pct = atr / spy_df["Close"]
            vol_percentile = atr_pct.rank(pct=True).iloc[-1] * 100

            regime_cfg = self.config.get("regime", {})
            trend_min_adx = regime_cfg.get("trend_strength_min_adx", 25)
            high_vol = regime_cfg.get("high_vol_threshold", 75)
            low_vol = regime_cfg.get("low_vol_threshold", 25)

            if vol_percentile > high_vol:
                regime = "volatile"
            elif adx > trend_min_adx:
                regime = "trending"
            elif adx < 15:
                regime = "ranging"
            else:
                regime = "medium"

            self.risk_manager.update_regime(regime)
            logger.info(f"📊 Market regime: {regime.upper()} (ADX={adx:.1f}, Vol%={vol_percentile:.0f})")

        except Exception as e:
            logger.debug(f"Regime detection failed: {e}")

    def _check_multi_timeframe_alignment(self, symbol: str, signal: int) -> bool:
        """
        Check if the signal aligns with higher timeframe trends.

        For BUY signals: require daily trend to be bullish (EMA9 > EMA21)
        For SELL signals: require daily trend to be bearish
        """
        if symbol not in self._htf_cache:
            return True  # No HTF data — don't filter

        htf = self._htf_cache[symbol]
        daily_trend = htf.get("daily_ema_trend", 0)
        daily_rsi = htf.get("daily_rsi", 50)

        if signal == 1:  # BUY
            # Require daily trend alignment
            if daily_trend == -1 and daily_rsi < 40:
                logger.info(f"  ⏭️ {symbol}: BUY rejected — daily trend bearish (RSI={daily_rsi:.0f})")
                return False
        elif signal == -1:  # SELL
            # Require daily trend alignment
            if daily_trend == 1 and daily_rsi > 60:
                logger.info(f"  ⏭️ {symbol}: SELL rejected — daily trend bullish (RSI={daily_rsi:.0f})")
                return False

        return True

    def _process_symbol(self, symbol: str):
        """Process a single symbol: data → features → signal → trade."""

        # Step 1: Fetch latest data (ensure at least 250+ bars for 200-period indicators)
        lookback_period = "2y" if self.timeframe == "1d" else "730d"
        if self.timeframe in ("15m", "5m"):
            lookback_period = "60d"
        elif self.timeframe in ("1h", "1H"):
            lookback_period = "3mo"

        df = self.data_loader.fetch_price_data(
            symbol, period=lookback_period, interval=self.timeframe, use_cache=False
        )
        if df.empty or len(df) < 50:
            return

        # Step 2: Fetch sentiment
        sentiment_features = {}
        try:
            articles = self.data_loader.fetch_news(symbol, days_back=3)
            if articles:
                scored = self.sentiment_analyzer.score_articles(articles)
                sentiment_features = self.sentiment_analyzer.compute_aggregate_sentiment(scored)
        except Exception as e:
            logger.debug(f"Sentiment fetch failed for {symbol}: {e}")

        # Step 3: Get higher timeframe context
        higher_tf_data = self._htf_cache.get(symbol, {})

        # Step 4: Compute features (with multi-TF and sentiment)
        features = self.feature_engine.compute_features(
            df, sentiment_features, higher_tf_data
        )
        if features.empty:
            return

        # Step 5: Get ML prediction for the latest bar
        latest_features = features.iloc[[-1]]
        prediction = self.ml_model.predict(
            latest_features,
            confidence_threshold=self.confidence_threshold,
        )

        signal = int(prediction["signal"].iloc[0])
        raw_signal = int(prediction["raw_signal"].iloc[0])
        confidence = float(prediction["confidence"].iloc[0])
        prob_buy = float(prediction.get("prob_buy", pd.Series([0.0])).iloc[0])
        prob_sell = float(prediction.get("prob_sell", pd.Series([0.0])).iloc[0])
        prob_hold = float(prediction.get("prob_hold", pd.Series([0.0])).iloc[0])
        effective_threshold = float(prediction.get("effective_threshold", pd.Series([self.confidence_threshold])).iloc[0])
        current_price = float(df["Close"].iloc[-1])

        if signal == 0:
            logger.info(
                f"📊 {symbol:<5} | HOLD | Conf: {confidence*100:.1f}% (< {effective_threshold*100:.0f}%) "
                f"| Prob -> Buy: {prob_buy*100:.1f}%, Sell: {prob_sell*100:.1f}%, Hold: {prob_hold*100:.1f}% "
                f"| Price: ${current_price:.2f}"
            )
            return  # No high-conviction signal — skip

        # Step 6: Multi-timeframe confirmation
        if not self._check_multi_timeframe_alignment(symbol, signal):
            return

        atr = df["Close"].pct_change().rolling(14).std().iloc[-1] * current_price

        logger.info(
            f"🎯 {symbol:<5} | 🔥 SIGNAL: {'BUY' if signal == 1 else 'SELL'} | "
            f"Confidence: {confidence*100:.1f}% | Price: ${current_price:.2f} | "
            f"Regime: {self.risk_manager._current_regime}"
        )

        # Step 7: Execute trade
        if signal == 1:
            # If we hold a short position, close it first
            if symbol in self.risk_manager.open_positions and self.risk_manager.open_positions[symbol].get("direction") == -1:
                self._close_position(symbol, current_price, "Flip from Short to Long")
            self._execute_buy(symbol, current_price, atr)
        elif signal == -1:
            # On cash accounts: SELL signal only closes existing longs (no short selling)
            if self.ibkr_client.is_cash_account():
                if symbol in self.risk_manager.open_positions and self.risk_manager.open_positions[symbol].get("direction", 1) == 1:
                    self._close_position(symbol, current_price, "SELL signal (cash account — long exit only)")
                else:
                    logger.info(f"  ⏭️ {symbol}: SELL signal skipped — cash account cannot short")
            else:
                # Margin account: close long and open short
                if symbol in self.risk_manager.open_positions and self.risk_manager.open_positions[symbol].get("direction", 1) == 1:
                    self._close_position(symbol, current_price, "Flip from Long to Short")
                self._execute_short(symbol, current_price, atr)

    def _execute_buy(self, symbol: str, price: float, atr: float):
        """Execute a buy trade with risk management."""

        # Compute bracket levels (direction=1 for long, regime-aware)
        bracket = self.risk_manager.compute_bracket_levels(price, atr, direction=1)

        # Position sizing (regime-aware, session-aware)
        sizing = self.risk_manager.calculate_position_size(
            entry_price=price,
            stop_loss_price=bracket["stop_loss"],
        )

        if sizing["shares"] < 1:
            logger.info(f"  ⏭️ {symbol}: Position too small, skipping")
            return

        # Validate trade (direction=1 for long)
        valid, reason = self.risk_manager.validate_trade(
            symbol, price, bracket["stop_loss"], sizing["shares"], direction=1
        )
        if not valid:
            logger.info(f"  ⏭️ {symbol}: {reason}")
            return

        # Place bracket order
        result = self.ibkr_client.place_bracket_order(
            symbol=symbol,
            shares=sizing["shares"],
            entry_price=round(price, 2),
            stop_loss=bracket["stop_loss"],
            take_profit=bracket["take_profit"],
            action="BUY",
        )

        if "error" not in result:
            self.risk_manager.register_open(
                symbol, price, sizing["shares"], bracket["stop_loss"], direction=1
            )
            logger.info(
                f"  ✅ BUY {sizing['shares']} x {symbol} @ ${price:.2f} | "
                f"SL: ${bracket['stop_loss']:.2f} | TP: ${bracket['take_profit']:.2f} | "
                f"Risk: ${sizing['risk_amount']:.2f} ({sizing['risk_pct']:.1f}%) | "
                f"R:R={bracket['risk_reward_ratio']} | "
                f"Size mult: {sizing['size_multiplier']:.2f}"
            )
        else:
            logger.error(f"  ❌ Buy order failed: {result['error']}")

    def _execute_short(self, symbol: str, price: float, atr: float):
        """Execute a short sell trade with risk management."""

        # Compute bracket levels (direction=-1 for short, regime-aware)
        bracket = self.risk_manager.compute_bracket_levels(price, atr, direction=-1)

        # Position sizing (regime-aware, session-aware)
        sizing = self.risk_manager.calculate_position_size(
            entry_price=price,
            stop_loss_price=bracket["stop_loss"],
        )

        if sizing["shares"] < 1:
            logger.info(f"  ⏭️ {symbol}: Position too small for short, skipping")
            return

        # Validate trade (direction=-1 for short)
        valid, reason = self.risk_manager.validate_trade(
            symbol, price, bracket["stop_loss"], sizing["shares"], direction=-1
        )
        if not valid:
            logger.info(f"  ⏭️ {symbol}: {reason}")
            return

        # Place bracket order (action='SELL' for short entry)
        result = self.ibkr_client.place_bracket_order(
            symbol=symbol,
            shares=sizing["shares"],
            entry_price=round(price, 2),
            stop_loss=bracket["stop_loss"],
            take_profit=bracket["take_profit"],
            action="SELL",
        )

        if "error" not in result:
            self.risk_manager.register_open(
                symbol, price, sizing["shares"], bracket["stop_loss"], direction=-1
            )
            logger.info(
                f"  ✅ SHORT SELL {sizing['shares']} x {symbol} @ ${price:.2f} | "
                f"SL: ${bracket['stop_loss']:.2f} | TP: ${bracket['take_profit']:.2f} | "
                f"Risk: ${sizing['risk_amount']:.2f} ({sizing['risk_pct']:.1f}%) | "
                f"R:R={bracket['risk_reward_ratio']} | "
                f"Size mult: {sizing['size_multiplier']:.2f}"
            )
        else:
            logger.error(f"  ❌ Short order failed: {result['error']}")

    def _close_position(self, symbol: str, price: float, reason: str = ""):
        """Close an existing open position (long or short)."""
        if symbol in self.risk_manager.open_positions:
            pos = self.risk_manager.open_positions[symbol]
            # To close a Short, action is BUY; to close a Long, action is SELL
            close_action = "BUY" if pos.get("direction", 1) == -1 else "SELL"
            result = self.ibkr_client.place_market_order(
                symbol, pos["shares"], action=close_action
            )
            if "error" not in result:
                self.risk_manager.register_close(symbol, price, reason or "Signal exit")

                # Log prediction outcome for adaptive threshold
                direction = pos.get("direction", 1)
                pnl = (price - pos["entry_price"]) * direction
                actual = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
                self.ml_model.log_prediction_outcome(
                    predicted=direction,
                    actual=actual,
                    confidence=0.5,  # Will be improved with stored confidence
                )
            else:
                logger.error(f"  ❌ Close position failed: {result['error']}")

    def _check_stops(self):
        """Check trailing stops and time stops for all open positions."""
        if not self.risk_manager.open_positions:
            return

        current_prices = {}
        for symbol in list(self.risk_manager.open_positions.keys()):
            price = self.ibkr_client.get_current_price(symbol)
            if price > 0:
                current_prices[symbol] = price

        to_close = self.risk_manager.check_trailing_stops(current_prices)

        for symbol in to_close:
            price = current_prices.get(symbol, 0)
            self._close_position(symbol, price, reason="Trailing stop / stop-loss / time stop")


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    bot = TradingBot()
    print("\nTrading Bot initialized.")
    print(f"  Watchlist: {bot.watchlist}")
    print(f"  Timeframe: {bot.timeframe}")
    print(f"  Ensemble: {bot.config.get('model', {}).get('ensemble_method', 'single')}")
    print(f"  Daily target: ${bot.config.get('targets', {}).get('daily_profit_target', 5.0)}")
    print("\nTo start paper trading: bot.start(paper=True)")
    print("Make sure to train a model first: python main.py train")
