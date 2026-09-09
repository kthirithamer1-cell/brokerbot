"""
ibkr_client.py — Interactive Brokers API Wrapper
=================================================
Handles connection to TWS / IB Gateway, order submission,
position tracking, and live data streaming via ib_insync.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class IBKRClient:
    """
    Wrapper around ib_insync for Interactive Brokers connectivity.

    Supports:
        - Paper and Live trading (TWS/Gateway)
        - Market/Limit/Stop orders with bracket logic
        - Real-time and historical bar data
        - Account and position queries
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        ibkr_cfg = self.config.get("ibkr", {})
        self.host = ibkr_cfg.get("host", "127.0.0.1")
        self.port = ibkr_cfg.get("port", 7497)  # 7497=TWS Paper
        self.client_id = ibkr_cfg.get("client_id", 1)
        self.account = ibkr_cfg.get("account", "")
        self.account_type = ibkr_cfg.get("account_type", "cash")  # "cash" or "margin"

        self.ib = None
        self._connected = False
        self._detected_account_type = None  # Auto-detected from IBKR

    # ──────────────────────────────────────────────
    #  CONNECTION
    # ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to IBKR TWS or IB Gateway."""
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=15,
            )
            self._connected = True

            # Detect account
            accounts = self.ib.managedAccounts()
            if not self.account and accounts:
                self.account = accounts[0]

            logger.info(f"✅ Connected to IBKR | Host: {self.host}:{self.port} | "
                         f"Account: {self.account}")

            # Log account type
            is_paper = self.port in (7497, 4002)
            mode = "📝 PAPER" if is_paper else "💰 LIVE"
            logger.info(f"   Mode: {mode}")

            # Auto-detect account type (Cash vs Margin)
            self._detect_account_type()

            return True

        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            logger.error("   Make sure TWS or IB Gateway is running and API is enabled.")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._connected and self.ib is not None and self.ib.isConnected()

    def _detect_account_type(self):
        """
        Auto-detect whether this is a Cash or Margin account from IBKR.
        Updates self.account_type and self._detected_account_type.
        """
        try:
            if not self.is_connected():
                return

            account_values = self.ib.accountSummary(self.account) if self.account else self.ib.accountSummary()
            for av in account_values:
                if av.tag == "AccountType":
                    detected = av.value.strip().upper()
                    self._detected_account_type = detected

                    # Map IBKR types: "INDIVIDUAL" is usually margin-capable,
                    # but we respect config override
                    if self.account_type == "cash":
                        logger.info(f"   Account type (config): CASH (IBKR reports: {detected})")
                        if "MARGIN" in detected:
                            logger.warning(
                                "   ⚠️ IBKR reports a margin account but config is set to 'cash'. "
                                "Cash account rules will be enforced."
                            )
                    else:
                        self.account_type = "margin"
                        logger.info(f"   Account type: MARGIN (IBKR: {detected})")
                    break

            logger.info(f"   🏦 Enforced mode: {self.account_type.upper()}")

        except Exception as e:
            logger.warning(f"   Could not detect account type: {e}. Using config: {self.account_type}")

    def is_cash_account(self) -> bool:
        """Check if running in cash account mode."""
        return self.account_type == "cash"

    # ──────────────────────────────────────────────
    #  ACCOUNT INFO
    # ──────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        """Get account equity, cash, and margin info."""
        if not self.is_connected():
            return {"error": "Not connected"}

        try:
            summary = {}
            if self.account:
                account_values = self.ib.accountSummary(self.account)
            else:
                account_values = self.ib.accountSummary()

            for av in account_values:
                if av.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                              "GrossPositionValue", "MaintMarginReq",
                              "SettledCash", "AvailableFunds", "AccountType"):
                    try:
                        if av.tag == "AccountType":
                            summary[av.tag] = av.value
                        else:
                            summary[av.tag] = float(av.value)
                    except (ValueError, TypeError):
                        pass

            return summary

        except Exception as e:
            logger.error(f"Account summary error: {e}")
            return {"error": str(e)}

    def get_equity(self) -> float:
        """Get current account net liquidation value."""
        summary = self.get_account_summary()
        return summary.get("NetLiquidation", 0.0)

    def get_settled_cash(self) -> float:
        """
        Get settled cash available for trading.

        On cash accounts, only settled funds can be used to buy.
        Unsettled funds from recent sells (T+1) are excluded.

        Returns:
            Settled cash in USD. Falls back to TotalCashValue if
            SettledCash is not available from IBKR.
        """
        summary = self.get_account_summary()
        settled = summary.get("SettledCash", None)
        if settled is not None:
            return settled
        # Fallback: AvailableFunds is usually settled on cash accounts
        available = summary.get("AvailableFunds", None)
        if available is not None:
            return available
        # Last resort: total cash (may include unsettled)
        logger.warning("⚠️ SettledCash not available from IBKR — using TotalCashValue (may include unsettled funds)")
        return summary.get("TotalCashValue", 0.0)

    def get_buying_power(self) -> float:
        """
        Get available buying power.

        On cash accounts, this equals settled cash.
        On margin accounts, this includes leverage.

        Returns:
            Buying power in USD.
        """
        if self.is_cash_account():
            return self.get_settled_cash()
        summary = self.get_account_summary()
        return summary.get("BuyingPower", 0.0)

    def get_positions(self) -> list[dict]:
        """Get current open positions."""
        if not self.is_connected():
            return []

        try:
            positions = self.ib.positions(self.account)
            result = []
            for pos in positions:
                result.append({
                    "symbol": pos.contract.symbol,
                    "shares": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_value": pos.position * pos.avgCost,
                    "contract": pos.contract,
                })
            return result

        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []

    # ──────────────────────────────────────────────
    #  MARKET DATA
    # ──────────────────────────────────────────────

    def get_stock_contract(self, symbol: str):
        """Create and qualify a US stock contract."""
        from ib_insync import Stock

        contract = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        return qualified[0] if qualified else contract

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        if not self.is_connected():
            return 0.0

        try:
            contract = self.get_stock_contract(symbol)
            ticker = self.ib.reqMktData(contract, snapshot=True)
            self.ib.sleep(2)  # Wait for data

            price = ticker.marketPrice()
            if price != price:  # NaN check
                price = ticker.last or ticker.close or 0.0

            self.ib.cancelMktData(contract)
            return float(price)

        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
            return 0.0

    def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
    ) -> pd.DataFrame:
        """
        Fetch historical bars from IBKR.

        Args:
            symbol: Ticker symbol
            duration: '1 Y', '6 M', '30 D', etc.
            bar_size: '1 day', '1 hour', '5 mins', etc.
            what_to_show: 'TRADES', 'MIDPOINT', 'BID', 'ASK'

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected():
            return pd.DataFrame()

        try:
            contract = self.get_stock_contract(symbol)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                return pd.DataFrame()

            from ib_insync import util
            df = util.df(bars)
            df.set_index("date", inplace=True)
            df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"
            }, inplace=True)

            return df[["Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    #  ORDER EXECUTION
    # ──────────────────────────────────────────────

    def place_market_order(self, symbol: str, shares: int, action: str = "BUY") -> dict:
        """
        Place a market order.

        Args:
            symbol: Ticker symbol
            shares: Number of shares
            action: 'BUY' or 'SELL'

        Returns:
            Trade status dict
        """
        if not self.is_connected():
            return {"error": "Not connected"}

        try:
            from ib_insync import MarketOrder

            contract = self.get_stock_contract(symbol)
            order = MarketOrder(action, shares)

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for fill

            logger.info(f"📤 {action} {shares} x {symbol} @ MARKET | "
                         f"Status: {trade.orderStatus.status}")

            return {
                "symbol": symbol,
                "action": action,
                "shares": shares,
                "order_type": "MARKET",
                "status": trade.orderStatus.status,
                "fill_price": trade.orderStatus.avgFillPrice,
                "order_id": trade.order.orderId,
            }

        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return {"error": str(e)}

    def place_bracket_order(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        action: str = "BUY",
    ) -> dict:
        """
        Place a bracket order (entry + stop-loss + take-profit).

        Args:
            symbol: Ticker symbol
            shares: Number of shares
            entry_price: Limit entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            action: 'BUY' or 'SELL'

        Returns:
            Bracket order status dict
        """
        if not self.is_connected():
            return {"error": "Not connected"}

        try:
            contract = self.get_stock_contract(symbol)

            bracket = self.ib.bracketOrder(
                action=action,
                quantity=shares,
                limitPrice=entry_price,
                takeProfitPrice=take_profit,
                stopLossPrice=stop_loss,
            )

            trades = []
            for order in bracket:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)

            self.ib.sleep(1)

            logger.info(
                f"📦 BRACKET {action} {shares} x {symbol} | "
                f"Entry: ${entry_price:.2f}, SL: ${stop_loss:.2f}, "
                f"TP: ${take_profit:.2f}"
            )

            return {
                "symbol": symbol,
                "action": action,
                "shares": shares,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "status": trades[0].orderStatus.status if trades else "Unknown",
                "parent_order_id": bracket[0].orderId if bracket else None,
            }

        except Exception as e:
            logger.error(f"Bracket order failed: {e}")
            return {"error": str(e)}

    def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self.is_connected():
            return

        try:
            self.ib.reqGlobalCancel()
            logger.info("🚫 All open orders cancelled")
        except Exception as e:
            logger.error(f"Cancel orders failed: {e}")

    def close_all_positions(self):
        """Close all open positions at market."""
        positions = self.get_positions()
        for pos in positions:
            if pos["shares"] > 0:
                self.place_market_order(pos["symbol"], abs(int(pos["shares"])), "SELL")
            elif pos["shares"] < 0:
                self.place_market_order(pos["symbol"], abs(int(pos["shares"])), "BUY")

        logger.info(f"Closed {len(positions)} positions")

    # ──────────────────────────────────────────────
    #  LIVE DATA STREAMING
    # ──────────────────────────────────────────────

    def subscribe_bars(self, symbol: str, bar_size: str = "5 mins", callback=None):
        """
        Subscribe to real-time bar updates.

        Args:
            symbol: Ticker symbol
            bar_size: Bar size ('5 secs', '1 min', '5 mins', etc.)
            callback: Function called with each new bar
        """
        if not self.is_connected():
            return None

        try:
            contract = self.get_stock_contract(symbol)
            bars = self.ib.reqRealTimeBars(contract, 5, "TRADES", False)

            if callback:
                bars.updateEvent += callback

            logger.info(f"📡 Subscribed to real-time bars: {symbol}")
            return bars

        except Exception as e:
            logger.error(f"Bar subscription failed for {symbol}: {e}")
            return None

    def sleep(self, seconds: float = 0):
        """Process IBKR messages for given duration."""
        if self.ib:
            self.ib.sleep(seconds)


# ──────────────────────────────────────────────
#  Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = IBKRClient()
    print("\nIBKR Client initialized.")
    print(f"  Host: {client.host}:{client.port}")
    print(f"  Client ID: {client.client_id}")

    is_paper = client.port in (7497, 4002)
    print(f"  Mode: {'Paper' if is_paper else 'Live'}")

    print("\nTo connect, make sure TWS or IB Gateway is running with API enabled.")
    print("Then call: client.connect()")
