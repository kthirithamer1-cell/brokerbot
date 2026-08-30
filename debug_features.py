import pandas as pd
from feature_engine import FeatureEngine

df = pd.read_parquet("data/SPY_1d_5y.parquet")
print(f"Loaded SPY parquet: {len(df)} rows")

engine = FeatureEngine()
feat = df.copy()

groups = [
    ("Moving Averages", engine._add_moving_averages),
    ("MACD", engine._add_macd),
    ("ADX", engine._add_adx),
    ("RSI", engine._add_rsi),
    ("Stochastic", engine._add_stochastic),
    ("Williams R", engine._add_williams_r),
    ("Bollinger", engine._add_bollinger_bands),
    ("ATR", engine._add_atr),
    ("Volume", engine._add_volume_features),
    ("Returns", engine._add_returns),
    ("Price", engine._add_price_features),
    ("Calendar", engine._add_calendar_features),
]

for name, func in groups:
    before_cols = set(feat.columns)
    func(feat)
    new_cols = set(feat.columns) - before_cols
    for col in new_cols:
        nan_count = feat[col].isna().sum()
        if nan_count > 0:
            print(f"[{name}] {col}: {nan_count}/{len(feat)} NaN")

print(f"Rows before dropna: {len(feat)}")
feat_clean = feat.dropna()
print(f"Rows after dropna: {len(feat_clean)}")

nan_counts = feat.isna().sum()
all_nan = nan_counts[nan_counts == len(feat)]
print(f"Columns that are 100% NaN: {list(all_nan.index)}")
