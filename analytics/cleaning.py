"""Raw -> clean price data.   *** STARTER VERSION. This file is Mikkel's to own and rewrite. ***

The app calls exactly two functions from here:

    clean_prices(raw)    -> DataFrame indexed by date with columns: close, distribution
    validate_prices(raw) -> list of human-readable issue strings (shown on the Data page)

`raw` is the rows for ONE ticker from the raw_prices table:
    date (YYYY-MM-DD str), close, adj_close, dividend, capital_gain, split

What the source data means (Yahoo via yfinance, auto_adjust=False):
  - close      is split-adjusted but NOT dividend-adjusted. Because every past price is
               restated in today's share units, the engine never has to apply splits.
  - dividend   is per-share cash, also in today's share units, on the ex-date.
  - adj_close  is split- AND dividend-adjusted (a total-return series).
  - capital_gain is fund capital-gain distributions. Yahoo sometimes ALSO folds these
               into `dividend` (double counting), so this starter ignores capital_gain
               and flags it instead. Deciding how to handle it is a cleaning call.

Ideas for the real version: reconcile close+dividends against adj_close and fix where they
disagree, cross-check against a second source (Tiingo), repair bad ticks, handle ticker
changes and delistings, decide how to treat capital-gain distributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df[df["close"].notna() & (df["close"] > 0)]
    df["distribution"] = df["dividend"].fillna(0.0).clip(lower=0.0)
    return df.set_index("date")[["close", "distribution"]]


def validate_prices(raw: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if raw.empty:
        return ["No data."]
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    bad = int((df["close"].isna() | (df["close"] <= 0)).sum())
    if bad:
        issues.append(f"{bad} rows with missing or non-positive close (dropped).")

    gaps = df["date"].diff().dt.days
    big_gaps = df.loc[gaps > 7, "date"]
    if len(big_gaps):
        issues.append(f"{len(big_gaps)} gaps longer than a week (e.g. before {big_gaps.iloc[0].date()}).")

    good = df[df["close"] > 0]
    r = good["close"].pct_change()
    spike = (r.abs() > 0.35) & (r.shift(-1).abs() > 0.25) & (np.sign(r) != np.sign(r.shift(-1)))
    if spike.any():
        issues.append(f"{int(spike.sum())} one-day spikes that reverse next day (possible bad ticks), "
                      f"e.g. {good.loc[spike, 'date'].iloc[0].date()}.")

    stale = (good["close"].diff() == 0).astype(int)
    run = stale.groupby((stale == 0).cumsum()).cumsum().max()
    if run and run >= 5:
        issues.append(f"Price unchanged for {int(run)} trading days in a row somewhere (stale data?).")

    if "capital_gain" in df and (df["capital_gain"].fillna(0) > 0).any():
        n = int((df["capital_gain"].fillna(0) > 0).sum())
        issues.append(f"{n} capital-gain distributions present but ignored by the starter cleaner "
                      f"(Yahoo may already include them in dividends).")

    # Reconciliation: total return rebuilt from close + dividends vs. the adj_close series.
    if "adj_close" in good and good["adj_close"].notna().sum() > 20:
        g = good.dropna(subset=["adj_close"])
        rebuilt = ((g["close"] + g["dividend"].fillna(0)) / g["close"].shift(1)).iloc[1:].prod()
        vendor = g["adj_close"].iloc[-1] / g["adj_close"].iloc[0]
        if vendor > 0 and abs(rebuilt / vendor - 1) > 0.01:
            issues.append(f"Total return from close+dividends ({rebuilt - 1:+.1%}) differs from "
                          f"adj_close ({vendor - 1:+.1%}) by more than 1%.")

    last = df["date"].iloc[-1]
    if (pd.Timestamp.today().normalize() - last).days > 10:
        issues.append(f"Data ends {last.date()}: delisted, renamed, or not updating? "
                      f"The engine holds the last price flat after this date.")
    return issues
