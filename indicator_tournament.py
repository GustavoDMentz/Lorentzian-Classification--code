"""Predeclared, causal, fixed-parameter indicator tournament.

These are simple research rules, not faithful replicas of full trading systems.
All signals are computed on completed candles and executed at the next open.
No parameters are selected using test data.
"""
import numpy as np
import pandas as pd
import talib

from backtest import backtest_next_open
from walk_forward import expanding_folds, cost_sensitivity


def indicator_targets(data: pd.DataFrame) -> dict[str, pd.Series]:
    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    c = pd.Series(close, index=data.index)
    h = pd.Series(high, index=data.index)
    l = pd.Series(low, index=data.index)

    def position(long, short):
        a = np.where(np.asarray(long, dtype=bool), 1,
                     np.where(np.asarray(short, dtype=bool), -1, 0))
        return pd.Series(a, index=data.index, dtype=int)

    ema_fast = talib.EMA(close, timeperiod=12)
    ema_slow = talib.EMA(close, timeperiod=26)
    sma50 = talib.SMA(close, timeperiod=50)
    sma200 = talib.SMA(close, timeperiod=200)
    macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    stoch_k, stoch_d = talib.STOCH(high, low, close)
    cci = talib.CCI(high, low, close, timeperiod=20)
    roc = talib.ROC(close, timeperiod=20)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
    adx = talib.ADX(high, low, close, timeperiod=14)
    # Previous completed 20-bar channel only; never includes current bar.
    channel_high = h.shift(1).rolling(20, min_periods=20).max()
    channel_low = l.shift(1).rolling(20, min_periods=20).min()

    return {
        "EMA 12/26 trend": position(ema_fast > ema_slow, ema_fast < ema_slow),
        "SMA 50/200 trend": position(sma50 > sma200, sma50 < sma200),
        "MACD 12/26/9": position(macd > signal, macd < signal),
        "Bollinger 20/2 reversal": position(close < lower, close > upper),
        "Stochastic 5/3/3": position((stoch_k < 20) & (stoch_d < 20),
                                     (stoch_k > 80) & (stoch_d > 80)),
        "CCI 20 reversal": position(cci < -100, cci > 100),
        "ROC 20 momentum": position(roc > 0, roc < 0),
        "ADX DI 14 trend": position((adx > 25) & (plus_di > minus_di),
                                     (adx > 25) & (minus_di > plus_di)),
        "Donchian 20 breakout": position(c > channel_high, c < channel_low),
    }


def tournament(data: pd.DataFrame, train_bars: int, test_bars: int,
               cost_bps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent test folds, no tuning; return ranked summary and fold details."""
    if cost_bps < 0 or not np.isfinite(cost_bps):
        raise ValueError("invalid cost")
    folds = expanding_folds(len(data), train_bars, test_bars)
    targets = indicator_targets(data)
    rows = []
    details = []
    for name, series in targets.items():
        fold_returns = []
        zero_returns = []
        turnover = 0.0
        long_bars = short_bars = flat_bars = 0
        for start, end in folds:
            local = backtest_next_open(data.iloc[start:end],
                                       series.iloc[start:end], cost_bps)
            net = float(local["equity"].iloc[-1] - 1)
            gross = float((1 + local["gross_return"]).prod() - 1)
            t = float(local["turnover"].sum())
            long = int((local["executed_position"] == 1).sum())
            short = int((local["executed_position"] == -1).sum())
            flat = int((local["executed_position"] == 0).sum())
            fold_returns.append(net)
            zero_returns.append(gross)
            turnover += t
            long_bars += long
            short_bars += short
            flat_bars += flat
            details.append({"strategy": name, "test_start": str(data.index[start]),
                            "test_end": str(data.index[end - 1]), "test_return": net,
                            "test_gross_return": gross, "turnover": t,
                            "long_bars": long, "short_bars": short, "flat_bars": flat})
        eq = np.cumprod(1 + np.asarray(fold_returns))
        peaks = np.maximum.accumulate(np.r_[1., eq])
        rows.append({
            "strategy": name, "folds": len(folds),
            "compounded_return": float(eq[-1] - 1),
            "gross_0bps": float(np.prod(1 + np.asarray(zero_returns)) - 1),
            "positive_folds": int((np.asarray(fold_returns) > 0).sum()),
            "worst_fold_return": float(min(fold_returns)),
            "fold_boundary_max_drawdown": float((eq / peaks[1:] - 1).min()),
            "turnover": turnover, "long_bars": long_bars,
            "short_bars": short_bars, "flat_bars": flat_bars,
        })
    summary = pd.DataFrame(rows).sort_values("compounded_return", ascending=False)
    return summary.reset_index(drop=True), pd.DataFrame(details)
