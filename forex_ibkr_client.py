"""
forex_ibkr_client.py — IBKR Forex API Wrapper
===============================================
Handles IBKR connection for forex (CASH) contracts,
lot-based order submission, margin queries, and swap rates.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ForexIBKRClient:
    """
    IBKR wrapper for forex trading.

    Uses ib_insync with Forex (CASH) contract types.
    Supports market/limit/stop orders, bracket orders,
    position queries, and margin info.
    """

    def __init__(self, config_path: str = "forex_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        ibkr_cfg = self.config.get("ibkr", {})
        self.host = ibkr_cfg.get("host", "127.0.0.1")
        self.port = ibkr_cfg.get("port", 7497)
        self.client_id = ibkr_cfg.get("client_id", 2)
        self.account = ibkr_cfg.get("account", "")

        self.ib = None
        self._connected = False

    # ──────────────────────────────────────────────
    #  CONNECTION
    # ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to IBKR TWS or IB Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(
                host=self.host, port=self.port,
                clientId=self.client_id, timeout=15,
            )
            self._connected = True

            accounts = self.ib.managedAccounts()
            if not self.account and accounts:
                self.account = accounts[0]

            is_paper = self.port in (7497, 4002)
            mode = "📝 PAPER" if is_paper else "💰 LIVE"
            logger.info(f"✅ Connected to IBKR Forex | {self.host}:{self.port} | "
                         f"Account: {self.account} | {mode}")
            return True

        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("🔌 Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.ib and self.ib.isConnected()

    # ──────────────────────────────────────────────
    #  FOREX CONTRACTS
    # ──────────────────────────────────────────────

    def _create_forex_contract(self, pair: str):
        """
        Create an IBKR Forex contract.

        IBKR forex uses:
            - symbol = base currency (e.g., 'EUR')
            - currency = quote currency (e.g., 'USD')
            - exchange = 'IDEALPRO' (interbank forex)
        """
        from ib_insync import Forex

        clean = pair.replace("=X", "").replace("/", "").upper()
        # Forex() in ib_insync takes the pair as a single string
        contract = Forex(clean)
        return contract

    # ──────────────────────────────────────────────
    #  ORDER EXECUTION
    # ──────────────────────────────────────────────

    def place_market_order(self, pair: str, units: int, action: str = "BUY") -> dict:
        """
        Place a market order for a forex pair.

        Args:
            pair: Forex pair (e.g., 'EURUSD')
            units: Position size in currency units
            action: 'BUY' or 'SELL'

        Returns:
            Dict with order info or error
        """
        if not self.is_connected():
            return {"error": "Not connected to IBKR"}

        try:
            from ib_insync import MarketOrder

            contract = self._create_forex_contract(pair)
            order = MarketOrder(action, units)

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for fill

            logger.info(f"📤 Market {action} {units} {pair} submitted")
            return {
                "order_id": trade.order.orderId,
                "status": trade.orderStatus.status,
                "pair": pair,
                "units": units,
                "action": action,
            }

        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return {"error": str(e)}

    def place_bracket_order(
        self, pair: str, units: int, entry_price: float,
        stop_loss: float, take_profit: float, action: str = "BUY",
    ) -> dict:
        """
        Place a bracket order (entry + SL + TP) for a forex pair.

        Args:
            pair: Forex pair
            units: Position size in currency units
            entry_price: Limit entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            action: 'BUY' or 'SELL'

        Returns:
            Dict with order info or error
        """
        if not self.is_connected():
            return {"error": "Not connected to IBKR"}

        try:
            from ib_insync import LimitOrder

            contract = self._create_forex_contract(pair)
            bracket = self.ib.bracketOrder(
                action=action,
                quantity=units,
                limitPrice=round(entry_price, 5),
                takeProfitPrice=round(take_profit, 5),
                stopLossPrice=round(stop_loss, 5),
            )

            trades = []
            for order in bracket:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)

            self.ib.sleep(1)

            logger.info(f"📤 Bracket {action} {units} {pair} @ {entry_price:.5f} "
                         f"(SL: {stop_loss:.5f}, TP: {take_profit:.5f})")
            return {
                "order_ids": [t.order.orderId for t in trades],
                "status": "submitted",
                "pair": pair,
                "units": units,
                "action": action,
            }

        except Exception as e:
            logger.error(f"Bracket order failed: {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────
    #  ACCOUNT INFO
    # ──────────────────────────────────────────────

    def get_equity(self) -> float:
        """Get account equity (net liquidation value)."""
        if not self.is_connected():
            return 0.0

        try:
            account_values = self.ib.accountValues(self.account)
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    return float(av.value)
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to get equity: {e}")
            return 0.0

    def get_positions(self) -> list[dict]:
        """Get all open forex positions."""
        if not self.is_connected():
            return []

        try:
            positions = self.ib.positions(self.account)
            forex_positions = []
            for pos in positions:
                if pos.contract.secType == "CASH":
                    forex_positions.append({
                        "pair": f"{pos.contract.symbol}{pos.contract.currency}",
                        "units": int(pos.position),
                        "avg_cost": float(pos.avgCost),
                        "market_value": float(pos.marketValue) if hasattr(pos, "marketValue") else 0,
                    })
            return forex_positions
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return []

    def get_current_price(self, pair: str) -> float:
        """Get current mid price for a forex pair."""
        if not self.is_connected():
            return 0.0

        try:
            contract = self._create_forex_contract(pair)
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(2)

            mid = (ticker.bid + ticker.ask) / 2 if ticker.bid > 0 and ticker.ask > 0 else 0
            self.ib.cancelMktData(contract)
            return mid

        except Exception as e:
            logger.warning(f"Failed to get price for {pair}: {e}")
            return 0.0

    def get_spread_pips(self, pair: str) -> float:
        """Get current spread in pips for a forex pair."""
        if not self.is_connected():
            return 0.0

        try:
            contract = self._create_forex_contract(pair)
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(2)

            if ticker.bid > 0 and ticker.ask > 0:
                spread = ticker.ask - ticker.bid
                clean = pair.replace("=X", "").replace("/", "").upper()
                pip_size = 0.01 if clean in {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY"} else 0.0001
                spread_pips = spread / pip_size
                self.ib.cancelMktData(contract)
                return round(spread_pips, 1)

            self.ib.cancelMktData(contract)
            return 0.0

        except Exception as e:
            logger.warning(f"Failed to get spread for {pair}: {e}")
            return 0.0
