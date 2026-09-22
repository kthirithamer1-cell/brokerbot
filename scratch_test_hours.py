from forex_risk_manager import ForexRiskManager

rm = ForexRiskManager("forex_config.yaml")
allowed, reason = rm.is_trading_allowed()
print(f"Trading allowed: {allowed}, Reason: {reason}")
