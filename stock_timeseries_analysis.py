#!/usr/bin/env python3
"""Single-stock time series analysis.

Features:
- Downloads historical OHLCV data via yfinance
- Or loads history from a local CSV (offline-friendly)
- Or generates a synthetic demo series (offline-friendly)
- Computes returns, rolling volatility, and moving averages
- Stationarity test (ADF) on log returns
- ACF/PACF plots on log returns
- Optional ARIMA forecasting on log price
- Saves CSV + plots to an output directory

Example:
  python stock_timeseries_analysis.py AAPL --start 2018-01-01 --out out
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


@dataclass
class Summary:
    ticker: str
    start: str
    end: str
    interval: str
    n_rows: int
    first_date: Optional[str]
    last_date: Optional[str]
    last_close: Optional[float]
    mean_daily_log_return: Optional[float]
    daily_log_return_std: Optional[float]
    annualized_volatility: Optional[float]
    adf_stat: Optional[float]
    adf_pvalue: Optional[float]
    ljungbox_pvalue_lag10: Optional[float]
    arima_order: Optional[Tuple[int, int, int]]
    arima_aic: Optional[float]


def _ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _to_iso(d: Any) -> Optional[str]:
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return None
    try:
        return pd.Timestamp(d).date().isoformat()
    except Exception:
        return str(d)


def download_history(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download data for {ticker} via yfinance: {e}. "
            f"If you're offline or blocked, use --csv PATH or --demo."
        ) from e

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} from yfinance. Check the symbol, date range, or network access. "
            f"If you're offline or blocked, use --csv PATH or --demo."
        )

    # Normalize column names and index
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Prefer Adj Close when present, otherwise Close
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Price"]).copy()

    return df


def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"CSV is empty: {path}")

    # Date column
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"], errors="coerce")
    else:
        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    if dt.isna().all():
        raise RuntimeError("Could not parse dates from CSV. Provide a 'Date' column or a date-like first column.")

    df = df.copy()
    df.index = dt
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.sort_index()

    # Price column mapping
    candidates = [
        "Adj Close",
        "AdjClose",
        "adj close",
        "adj_close",
        "Close",
        "close",
        "Price",
        "price",
    ]
    price_col = next((c for c in candidates if c in df.columns), None)
    if price_col is None:
        raise RuntimeError(
            "CSV must contain a price column such as 'Close' or 'Adj Close' (case sensitive for now). "
            f"Available columns: {list(df.columns)}"
        )

    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Price"]).copy()
    if df.empty:
        raise RuntimeError("CSV loaded but no valid numeric prices were found after cleaning.")
    return df


def simulate_history(ticker: str, start: str, end: str, interval: str, seed: int = 7) -> pd.DataFrame:
    # Simple geometric random walk on business days.
    # (Not intended to be a market model; only to enable offline demo of the analysis.)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if interval != "1d":
        # For non-daily intervals, approximate by daily anyway.
        pass
    idx = pd.bdate_range(start_ts, end_ts, inclusive="both")
    if len(idx) < 60:
        # Ensure we have enough points for the diagnostics/ARIMA selection to be meaningful.
        idx = pd.bdate_range(end_ts - pd.Timedelta(days=365), end_ts, inclusive="both")

    rng = np.random.default_rng(seed)
    mu = 0.0002  # drift
    sigma = 0.02  # daily vol
    log_rets = rng.normal(loc=mu, scale=sigma, size=len(idx))
    log_price = np.log(100.0) + np.cumsum(log_rets)
    price = np.exp(log_price)

    df = pd.DataFrame(index=idx, data={"Close": price})
    df["Price"] = df["Close"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Log price and log returns
    out["LogPrice"] = np.log(out["Price"])
    out["LogReturn"] = out["LogPrice"].diff()

    # Simple returns for convenience
    out["Return"] = out["Price"].pct_change()

    # Moving averages on price
    for w in (20, 50, 200):
        out[f"SMA_{w}"] = out["Price"].rolling(w).mean()

    # Rolling volatility (annualized; assumes ~252 trading days for daily interval)
    # For non-daily intervals this is still useful but less interpretable.
    out["RollingVol_20"] = out["LogReturn"].rolling(20).std() * np.sqrt(252.0)

    return out


def adf_test(series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    s = series.dropna()
    if len(s) < 30:
        return None, None
    stat, pvalue, *_ = adfuller(s.values, autolag="AIC")
    return float(stat), float(pvalue)


def ljungbox_test(series: pd.Series, lag: int = 10) -> Optional[float]:
    s = series.dropna()
    if len(s) < (lag + 10):
        return None
    res = acorr_ljungbox(s.values, lags=[lag], return_df=True)
    return float(res["lb_pvalue"].iloc[0])


def choose_arima_order(
    log_price: pd.Series,
    p_vals: Iterable[int] = (0, 1, 2),
    d_vals: Iterable[int] = (1,),
    q_vals: Iterable[int] = (0, 1, 2),
) -> Tuple[Optional[Tuple[int, int, int]], Optional[float]]:
    s = log_price.dropna()
    if len(s) < 100:
        return None, None

    best_order: Optional[Tuple[int, int, int]] = None
    best_aic: Optional[float] = None

    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                order = (p, d, q)
                try:
                    model = ARIMA(s, order=order)
                    fit = model.fit(method_kwargs={"warn_convergence": False})
                    aic = float(fit.aic)
                    if best_aic is None or aic < best_aic:
                        best_aic = aic
                        best_order = order
                except Exception:
                    continue

    return best_order, best_aic


def fit_forecast_arima(
    log_price: pd.Series, order: Tuple[int, int, int], steps: int
) -> pd.DataFrame:
    s = log_price.dropna()
    model = ARIMA(s, order=order)
    fit = model.fit(method_kwargs={"warn_convergence": False})

    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    out = pd.DataFrame(
        {
            "LogPrice_Forecast": mean,
            "LogPrice_Lower": ci.iloc[:, 0],
            "LogPrice_Upper": ci.iloc[:, 1],
        }
    )
    out["Price_Forecast"] = np.exp(out["LogPrice_Forecast"])
    out["Price_Lower"] = np.exp(out["LogPrice_Lower"])
    out["Price_Upper"] = np.exp(out["LogPrice_Upper"])
    return out


def save_plots(df: pd.DataFrame, ticker: str, out_dir: str, horizon: int, arima_fc: Optional[pd.DataFrame]) -> None:
    # 1) Price + SMAs
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Price"], label="Price", linewidth=1.5)
    for w in (20, 50, 200):
        col = f"SMA_{w}"
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.0)
    ax.set_title(f"{ticker} Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "price_and_sma.png"), dpi=150)
    plt.close(fig)

    # 2) Rolling annualized volatility
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df["RollingVol_20"], label="20-day rolling vol (ann.)", linewidth=1.2)
    ax.set_title(f"{ticker} Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rolling_volatility.png"), dpi=150)
    plt.close(fig)

    # 3) Log return distribution
    lr = df["LogReturn"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(lr.values, bins=60, alpha=0.85)
    ax.set_title(f"{ticker} Daily Log Return Histogram")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "log_return_hist.png"), dpi=150)
    plt.close(fig)

    # 4) ACF/PACF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_acf(lr.values, lags=40, ax=axes[0])
    axes[0].set_title("ACF (log returns)")
    plot_pacf(lr.values, lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("PACF (log returns)")
    fig.suptitle(f"{ticker} Autocorrelation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "acf_pacf.png"), dpi=150)
    plt.close(fig)

    # 5) ARIMA forecast (optional)
    if arima_fc is not None and not arima_fc.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        tail = df[["Price"]].dropna().tail(max(260, horizon * 6))
        ax.plot(tail.index, tail["Price"], label="History", linewidth=1.5)
        ax.plot(arima_fc.index, arima_fc["Price_Forecast"], label="Forecast", linewidth=1.5)
        ax.fill_between(
            arima_fc.index,
            arima_fc["Price_Lower"],
            arima_fc["Price_Upper"],
            alpha=0.2,
            label="95% CI",
        )
        ax.set_title(f"{ticker} ARIMA Forecast ({horizon} steps)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "arima_forecast.png"), dpi=150)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Time series analysis for a single stock ticker")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 1h)")
    parser.add_argument("--out", default="out", help="Output directory")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--csv", help="Path to CSV with Date + Close/Adj Close columns (offline-friendly)")
    src.add_argument("--demo", action="store_true", help="Use synthetic demo data (offline-friendly)")

    parser.add_argument(
        "--arima",
        action="store_true",
        help="Fit ARIMA on log price and forecast",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Forecast steps (only with --arima)",
    )

    args = parser.parse_args()

    out_dir = _ensure_out_dir(args.out)

    if args.csv:
        df_raw = load_history_csv(args.csv)
    elif args.demo:
        df_raw = simulate_history(args.ticker, args.start, args.end, args.interval)
    else:
        df_raw = download_history(args.ticker, args.start, args.end, args.interval)
    df = add_features(df_raw)

    adf_stat, adf_p = adf_test(df["LogReturn"])
    lb_p = ljungbox_test(df["LogReturn"], lag=10)

    arima_order: Optional[Tuple[int, int, int]] = None
    arima_aic: Optional[float] = None
    arima_fc: Optional[pd.DataFrame] = None

    if args.arima:
        arima_order, arima_aic = choose_arima_order(df["LogPrice"])
        if arima_order is not None:
            arima_fc = fit_forecast_arima(df["LogPrice"], arima_order, steps=max(1, args.horizon))

            # Try to build a sensible forecast index.
            # If the input index is business-day-like, extend with business days.
            last_idx = df.dropna(subset=["LogPrice"]).index[-1]
            freq = pd.infer_freq(df.dropna(subset=["LogPrice"]).index)
            if freq is None and args.interval == "1d":
                # Start from next business day after last_idx
                start_date = pd.bdate_range(last_idx, periods=2)[-1]
                idx = pd.bdate_range(start_date, periods=len(arima_fc))
            else:
                # Start from next period after last_idx
                if freq is None:
                    freq = "D"  # Default to daily
                # Get next period by generating 2 periods and taking the last one
                start_date = pd.date_range(last_idx, periods=2, freq=freq)[-1]
                idx = pd.date_range(start_date, periods=len(arima_fc), freq=freq)
            arima_fc.index = idx

            arima_fc.to_csv(os.path.join(out_dir, "arima_forecast.csv"), index=True)

    # Save full dataset
    df.to_csv(os.path.join(out_dir, f"{args.ticker}_history_with_features.csv"), index=True)

    # Summary
    last_close = float(df["Price"].dropna().iloc[-1]) if df["Price"].notna().any() else None
    lr = df["LogReturn"].dropna()
    mean_lr = float(lr.mean()) if len(lr) else None
    std_lr = float(lr.std(ddof=1)) if len(lr) > 1 else None
    ann_vol = float(std_lr * np.sqrt(252.0)) if std_lr is not None else None

    summary = Summary(
        ticker=args.ticker,
        start=str(args.start),
        end=str(args.end),
        interval=str(args.interval),
        n_rows=int(len(df)),
        first_date=_to_iso(df.index.min()) if len(df.index) else None,
        last_date=_to_iso(df.index.max()) if len(df.index) else None,
        last_close=last_close,
        mean_daily_log_return=mean_lr,
        daily_log_return_std=std_lr,
        annualized_volatility=ann_vol,
        adf_stat=adf_stat,
        adf_pvalue=adf_p,
        ljungbox_pvalue_lag10=lb_p,
        arima_order=arima_order,
        arima_aic=arima_aic,
    )

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    save_plots(df, args.ticker, out_dir, horizon=args.horizon, arima_fc=arima_fc)

    # Console summary
    print(json.dumps(asdict(summary), indent=2))
    print(f"\nWrote outputs to: {os.path.abspath(out_dir)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
