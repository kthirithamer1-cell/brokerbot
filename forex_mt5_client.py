"""
forex_mt5_client.py — MetaTrader 5 (XM) Forex API Wrapper
==========================================================
Drop-in replacement for forex_ibkr_client.py.
Routes orders through the MT5 Python bridge to XM (or any MT5 broker).

Requirements:
    - MetaTrader 5 terminal installed and running on Windows
    - pip install MetaTrader5
    - MT5 terminal logged into your XM demo/live account
"""

import logging
import time
from datetime import datetime
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class ForexMT5Client:
    """
    MetaTrader 5 wrapper for forex trading.

    Uses the official MetaTrader5 Python package to:
        - Connect to the MT5 terminal process
        - Place market orders with attached SL/TP
        - Query account equity, positions, and live prices
        - Calculate real-time spread in pips

    Compatible with any MT5 broker (XM, Exness, IC Markets, etc.)
    """

    # JPY pairs use 0.01 pip size, others use 0.0001
    JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}

    def __init__(self, config_path: str = "forex_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        mt5_cfg = self.config.get("mt5", {})
        self.terminal_path = mt5_cfg.get("terminal_path", "")
        self.login = mt5_cfg.get("login", 0)
        self.password = mt5_cfg.get("password", "")
        self.server = mt5_cfg.get("server", "")
        self.timeout = mt5_cfg.get("timeout", 10000)
        self.symbol_suffix = mt5_cfg.get("symbol_suffix", "")
        self.magic_number = mt5_cfg.get("magic_number", 234567)
        self.deviation = mt5_cfg.get("deviation", 20)
        self.fill_policy = mt5_cfg.get("fill_policy", "ioc")

        # Lot size mapping
        lot_type = mt5_cfg.get("lot_type", "micro")
        self._lot_sizes = {"standard": 1.0, "mini": 0.1, "micro": 0.01}
        self.base_lot_size = self._lot_sizes.get(lot_type, 0.01)

        self._connected = False
        self._mt5 = None

    # ──────────────────────────────────────────────
    #  SYMBOL UTILITIES
    # ──────────────────────────────────────────────

    def _symbol(self, pair: str) -> str:
        """Convert internal pair name to MT5 symbol name."""
        clean = pair.replace("=X", "").replace("/", "").upper()
        return f"{clean}{self.symbol_suffix}"

    def _pip_size(self, pair: str) -> float:
        """Get pip size for a pair."""
        clean = pair.replace("=X", "").replace("/", "").upper()
        return 0.01 if clean in self.JPY_PAIRS else 0.0001

    def _units_to_lots(self, units: int) -> float:
        """Convert currency units to MT5 lot size."""
        # 1 standard lot = 100,000 units
        lots = units / 100_000
        # Round to nearest 0.01 (micro lot)
        return round(max(lots, 0.01), 2)

    # ──────────────────────────────────────────────
    #  CONNECTION
    # ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize and connect to the MT5 terminal."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5

            # Initialize MT5 terminal
            init_kwargs = {}
            if self.terminal_path:
                init_kwargs["path"] = self.terminal_path
            if self.login:
                init_kwargs["login"] = int(self.login)
            if self.password:
                init_kwargs["password"] = self.password
            if self.server:
                init_kwargs["server"] = self.server
            if self.timeout:
                init_kwargs["timeout"] = int(self.timeout)

            if not mt5.initialize(**init_kwargs):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False

            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get MT5 account info")
                mt5.shutdown()
                return False

            self._connected = True

            # Determine if demo or live
            trade_mode = account_info.trade_mode
            mode_names = {0: "DEMO", 1: "CONTEST", 2: "LIVE"}
            mode_str = mode_names.get(trade_mode, f"Mode {trade_mode}")

            logger.info(f"Connected to MT5 | {mode_str}")
            logger.info(f"   Account: {account_info.login} | "
                         f"Server: {account_info.server}")
            logger.info(f"   Name: {account_info.name} | "
                         f"Broker: {account_info.company}")
            logger.info(f"   Balance: ${account_info.balance:,.2f} | "
                         f"Equity: ${account_info.equity:,.2f} | "
                         f"Leverage: 1:{account_info.leverage}")
            return True

        except ImportError:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False
        except Exception as e:
            logger.error(f"MT5 connection failed: {e}")
            return False

    def disconnect(self):
        """Shutdown MT5 connection."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("Disconnected from MT5")

    def is_connected(self) -> bool:
        """Check if MT5 terminal is connected."""
        if not self._connected or not self._mt5:
            return False
        try:
            info = self._mt5.terminal_info()
            return info is not None and info.connected
        except Exception:
            return False

    # ──────────────────────────────────────────────
    #  ORDER EXECUTION
    # ──────────────────────────────────────────────

    def place_market_order(self, pair: str, units: int, action: str = "BUY") -> dict:
        """
        Place a market order for a forex pair.

        Args:
            pair: Forex pair (e.g., 'EURUSD')
            units: Position size in currency units (will be converted to lots)
            action: 'BUY' or 'SELL'

        Returns:
            Dict with order info or error
        """
        if not self.is_connected():
            return {"error": "Not connected to MT5"}

        mt5 = self._mt5
        symbol = self._symbol(pair)
        lots = self._units_to_lots(units)

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Symbol {symbol} not available in MT5"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"error": f"Cannot get tick data for {symbol}"}

        order_type = mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action.upper() == "BUY" else tick.bid

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "ForexAI Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"error": f"Order send returned None: {mt5.last_error()}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"error": f"Order failed: {result.comment} (code {result.retcode})"}

        logger.info(f"Market {action} {lots} lots {pair} @ {result.price:.5f} "
                     f"(ticket #{result.order})")
        return {
            "order_id": result.order,
            "ticket": result.order,
            "status": "filled",
            "pair": pair,
            "units": units,
            "lots": lots,
            "action": action,
            "fill_price": result.price,
        }

    def place_bracket_order(
        self, pair: str, units: int, entry_price: float,
        stop_loss: float, take_profit: float, action: str = "BUY",
    ) -> dict:
        """
        Place a market order with attached SL and TP levels.

        In MT5, bracket orders are achieved by attaching sl/tp to the
        market order request directly (unlike IBKR which uses 3 separate orders).

        Args:
            pair: Forex pair
            units: Position size in currency units
            entry_price: Expected entry price (used for logging; actual fill is at market)
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            action: 'BUY' or 'SELL'

        Returns:
            Dict with order info or error
        """
        if not self.is_connected():
            return {"error": "Not connected to MT5"}

        mt5 = self._mt5
        symbol = self._symbol(pair)
        lots = self._units_to_lots(units)

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            return {"error": f"Symbol {symbol} not available in MT5"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"error": f"Cannot get tick data for {symbol}"}

        order_type = mt5.ORDER_TYPE_BUY if action.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if action.upper() == "BUY" else tick.bid

        # Normalize SL/TP to proper decimal precision
        symbol_info = mt5.symbol_info(symbol)
        digits = symbol_info.digits if symbol_info else 5

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": round(price, digits),
            "sl": round(stop_loss, digits),
            "tp": round(take_profit, digits),
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "ForexAI Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return {"error": f"Order send returned None: {mt5.last_error()}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"error": f"Order failed: {result.comment} (code {result.retcode})"}

        pip_size = self._pip_size(pair)
        sl_pips = abs(price - stop_loss) / pip_size
        tp_pips = abs(take_profit - price) / pip_size

        logger.info(f"Bracket {action} {lots} lots {pair} @ {result.price:.5f} "
                     f"(SL: {stop_loss:.5f} [{sl_pips:.1f} pips], "
                     f"TP: {take_profit:.5f} [{tp_pips:.1f} pips])")
        return {
            "order_ids": [result.order],
            "ticket": result.order,
            "status": "filled",
            "pair": pair,
            "units": units,
            "lots": lots,
            "action": action,
            "fill_price": result.price,
        }

    # ──────────────────────────────────────────────
    #  POSITION MANAGEMENT
    # ──────────────────────────────────────────────

    def close_position(self, pair: str) -> dict:
        """Close all positions for a specific pair."""
        if not self.is_connected():
            return {"error": "Not connected to MT5"}

        mt5 = self._mt5
        symbol = self._symbol(pair)
        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            return {"message": f"No open positions for {pair}"}

        results = []
        for pos in positions:
            # Determine close direction
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue

            price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": "ForexAI Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                pnl = pos.profit
                logger.info(f"Closed {pair} ticket #{pos.ticket} | P&L: ${pnl:+.2f}")
                results.append({"ticket": pos.ticket, "pnl": pnl})
            else:
                error_msg = result.comment if result else "Unknown error"
                logger.error(f"Failed to close {pair} #{pos.ticket}: {error_msg}")

        return {"closed": results}

    def close_all_positions(self) -> dict:
        """Emergency: close ALL open positions."""
        if not self.is_connected():
            return {"error": "Not connected to MT5"}

        mt5 = self._mt5
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            logger.info("No open positions to close")
            return {"message": "No open positions"}

        closed = []
        for pos in positions:
            result = self.close_position(pos.symbol.replace(self.symbol_suffix, ""))
            closed.extend(result.get("closed", []))

        total_pnl = sum(c.get("pnl", 0) for c in closed)
        logger.info(f"Closed {len(closed)} positions | Total P&L: ${total_pnl:+.2f}")
        return {"closed": closed, "total_pnl": total_pnl}

    # ──────────────────────────────────────────────
    #  ACCOUNT INFO
    # ──────────────────────────────────────────────

    def get_equity(self) -> float:
        """Get account equity."""
        if not self.is_connected():
            return 0.0

        try:
            info = self._mt5.account_info()
            return float(info.equity) if info else 0.0
        except Exception as e:
            logger.warning(f"Failed to get equity: {e}")
            return 0.0

    def get_balance(self) -> float:
        """Get account balance."""
        if not self.is_connected():
            return 0.0

        try:
            info = self._mt5.account_info()
            return float(info.balance) if info else 0.0
        except Exception as e:
            logger.warning(f"Failed to get balance: {e}")
            return 0.0

    def get_positions(self) -> list[dict]:
        """Get all open forex positions."""
        if not self.is_connected():
            return []

        try:
            positions = self._mt5.positions_get()
            if positions is None:
                return []

            forex_positions = []
            for pos in positions:
                direction = "LONG" if pos.type == 0 else "SHORT"
                pair_clean = pos.symbol.replace(self.symbol_suffix, "")
                forex_positions.append({
                    "pair": pair_clean,
                    "ticket": pos.ticket,
                    "direction": direction,
                    "units": int(pos.volume * 100_000),
                    "lots": pos.volume,
                    "avg_cost": float(pos.price_open),
                    "current_price": float(pos.price_current),
                    "pnl": float(pos.profit),
                    "sl": float(pos.sl),
                    "tp": float(pos.tp),
                    "swap": float(pos.swap),
                    "open_time": datetime.fromtimestamp(pos.time),
                })
            return forex_positions
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return []

    def get_tick(self, pair: str) -> dict:
        """Get live real-time tick (bid, ask, mid, spread) directly from XM MT5 terminal."""
        if not self.is_connected():
            return {}

        try:
            symbol = self._symbol(pair)
            if not self._mt5.symbol_select(symbol, True):
                return {}
            tick = self._mt5.symbol_info_tick(symbol)
            if tick and tick.bid > 0 and tick.ask > 0:
                pip_size = self._pip_size(pair)
                spread = round((tick.ask - tick.bid) / pip_size, 1)
                mid = round((tick.bid + tick.ask) / 2, 5)
                return {
                    "symbol": symbol,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "mid": mid,
                    "spread_pips": spread,
                    "time": tick.time,
                }
            return {}
        except Exception as e:
            logger.warning(f"Failed to get tick for {pair}: {e}")
            return {}

    def get_current_price(self, pair: str) -> float:
        """Get current mid price for a forex pair directly from MT5 tick."""
        tick = self.get_tick(pair)
        return tick.get("mid", 0.0)

    def get_spread_pips(self, pair: str) -> float:
        """Get current spread in pips for a forex pair directly from MT5 tick."""
        tick = self.get_tick(pair)
        return tick.get("spread_pips", 0.0)

    def get_bars(self, pair: str, timeframe_str: str = "1h", count: int = 300):
        """Fetch real-time historical candles directly from XM MT5 terminal server."""
        if not self.is_connected():
            import pandas as pd
            return pd.DataFrame()

        try:
            import pandas as pd
            mt5 = self._mt5
            symbol = self._symbol(pair)
            if not mt5.symbol_select(symbol, True):
                return pd.DataFrame()

            tf_map = {
                "1m": mt5.TIMEFRAME_M1,
                "5m": mt5.TIMEFRAME_M5,
                "15m": mt5.TIMEFRAME_M15,
                "30m": mt5.TIMEFRAME_M30,
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
            }
            mt5_tf = tf_map.get(timeframe_str.lower(), mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates is None or len(rates) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
            }, inplace=True)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            logger.warning(f"Failed to copy rates from MT5 for {pair}: {e}")
            import pandas as pd
            return pd.DataFrame()

    def get_account_info(self) -> dict:
        """Get comprehensive account information."""
        if not self.is_connected():
            return {"error": "Not connected"}

        try:
            info = self._mt5.account_info()
            if info is None:
                return {"error": "No account info"}

            mode_names = {0: "DEMO", 1: "CONTEST", 2: "LIVE"}
            return {
                "login": info.login,
                "name": info.name,
                "server": info.server,
                "company": info.company,
                "trade_mode": mode_names.get(info.trade_mode, "UNKNOWN"),
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "free_margin": info.margin_free,
                "margin_level": info.margin_level,
                "leverage": info.leverage,
                "currency": info.currency,
                "profit": info.profit,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_trade_history(self, days: int = 7) -> list[dict]:
        """Fetch closed trades / deals history from MT5.
        
        Args:
            days: Number of past days to query
            
        Returns:
            List of closed trade deal dicts with pnl, volume, prices
        """
        if not self.is_connected():
            return []

        try:
            from datetime import timedelta
            now = datetime.now()
            date_from = now - timedelta(days=days)
            deals = self._mt5.history_deals_get(date_from, now)
            if deals is None:
                return []

            history = []
            for d in deals:
                if not d.symbol:
                    continue
                # In MT5: 0=DEAL_ENTRY_IN, 1=DEAL_ENTRY_OUT, 2=DEAL_ENTRY_INOUT, 3=DEAL_ENTRY_OUT_BY
                entry_names = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}
                deal_time = datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M:%S")
                history.append({
                    "ticket": d.ticket,
                    "order": d.order,
                    "position_id": d.position_id,
                    "time": deal_time,
                    "symbol": d.symbol,
                    "type": "BUY" if d.type == 0 else "SELL",
                    "entry": entry_names.get(d.entry, str(d.entry)),
                    "lots": d.volume,
                    "price": d.price,
                    "profit": round(d.profit, 2),
                    "commission": round(d.commission, 2),
                    "swap": round(d.swap, 2),
                    "magic": d.magic,
                    "comment": d.comment,
                })
            return history
        except Exception as e:
            logger.warning(f"Failed to fetch trade history: {e}")
            return []



# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    client = ForexMT5Client()

    if client.connect():
        print("\n=== MT5 Client Test ===")

        # Account info
        info = client.get_account_info()
        print(f"\nAccount: {info.get('login')} ({info.get('trade_mode')})")
        print(f"  Broker: {info.get('company')}")
        print(f"  Balance: ${info.get('balance', 0):,.2f}")
        print(f"  Equity: ${info.get('equity', 0):,.2f}")
        print(f"  Leverage: 1:{info.get('leverage', 0)}")

        # Live prices
        for pair in ["GBPUSD", "EURUSD", "GBPJPY"]:
            price = client.get_current_price(pair)
            spread = client.get_spread_pips(pair)
            print(f"  {pair}: {price:.5f} (spread: {spread:.1f} pips)")

        # Open positions
        positions = client.get_positions()
        print(f"\nOpen positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos['pair']} {pos['direction']} {pos['lots']} lots | "
                  f"P&L: ${pos['pnl']:+.2f}")

        client.disconnect()
    else:
        print("Failed to connect to MT5. Is the terminal running?")
