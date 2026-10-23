"""Stateful, predeclared trading systems. Research interpretations, not canonical replicas.

Signals are computed on closed candles, positions fill at the next open.
All systems are long/short/flat, unit exposure, no leverage beyond short notional.
Stops use closing prices (not intrabar fills); all thresholds are fixed ex ante.
"""
import numpy as np
import pandas as pd
import talib

from backtest import backtest_next_open
from walk_forward import expanding_folds


def _stateful(index, long_entry, short_entry, long_exit, short_exit,
              allow_reverse=True):
    """Finite state machine; exits take precedence, then entries on same close."""
    n = len(index)
    arrays = [np.asarray(x, dtype=bool) for x in
              (long_entry, short_entry, long_exit, short_exit)]
    if any(len(x) != n for x in arrays):
        raise ValueError("signal length mismatch")
    le, se, lx, sx = arrays
    out = np.zeros(n, dtype=np.int8)
    state = 0
    for i in range(n):
        if state == 1 and lx[i]:
            state = 0
        elif state == -1 and sx[i]:
            state = 0
        if state == 0 or allow_reverse:
            if le[i] and not se[i] and (state == 0 or allow_reverse):
                state = 1
            elif se[i] and not le[i] and (state == 0 or allow_reverse):
                state = -1
        out[i] = state
    return pd.Series(out, index=index, dtype=int)


def system_targets(data: pd.DataFrame) -> dict[str, pd.Series]:
    close = data["close"].to_numpy(dtype=float)
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    idx = data.index
    c = pd.Series(close, index=idx)
    h = pd.Series(high, index=idx)
    l = pd.Series(low, index=idx)
    cross_up = lambda a, b: (a > b) & (pd.Series(a, index=idx).shift(1).to_numpy() <= pd.Series(b, index=idx).shift(1).to_numpy())
    cross_dn = lambda a, b: (a < b) & (pd.Series(a, index=idx).shift(1).to_numpy() >= pd.Series(b, index=idx).shift(1).to_numpy())

    ef = talib.EMA(close, timeperiod=12)
    es = talib.EMA(close, timeperiod=26)
    sf = talib.SMA(close, timeperiod=50)
    ss = talib.SMA(close, timeperiod=200)
    macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    k, d = talib.STOCH(high, low, close)
    cci = talib.CCI(high, low, close, timeperiod=20)
    roc = talib.ROC(close, timeperiod=20)
    adx = talib.ADX(high, low, close, timeperiod=14)
    plus = talib.PLUS_DI(high, low, close, timeperiod=14)
    minus = talib.MINUS_DI(high, low, close, timeperiod=14)
    rsi = talib.RSI(close, timeperiod=14)
    prev_hi20 = h.shift(1).rolling(20, min_periods=20).max().to_numpy()
    prev_lo20 = l.shift(1).rolling(20, min_periods=20).min().to_numpy()
    prev_hi10 = h.shift(1).rolling(10, min_periods=10).max().to_numpy()
    prev_lo10 = l.shift(1).rolling(10, min_periods=10).min().to_numpy()
    zeros = np.zeros(len(data), dtype=bool)

    return {
        "EMA 12/26 crossover": _stateful(idx, cross_up(ef, es), cross_dn(ef, es),
                                          cross_dn(ef, es), cross_up(ef, es)),
        "SMA 50/200 crossover": _stateful(idx, cross_up(sf, ss), cross_dn(sf, ss),
                                           cross_dn(sf, ss), cross_up(sf, ss)),
        "MACD signal crossover": _stateful(idx, cross_up(macd, macd_signal),
                                            cross_dn(macd, macd_signal),
                                            cross_dn(macd, macd_signal),
                                            cross_up(macd, macd_signal)),
        "Bollinger reversion": _stateful(idx, close < lower, close > upper,
                                         close >= mid, close <= mid,
                                         allow_reverse=False),
        "Stochastic reversion": _stateful(idx, cross_up(k, d) & (k < 20),
                                          cross_dn(k, d) & (k > 80),
                                          k >= 50, k <= 50, allow_reverse=False),
        "CCI reversion": _stateful(idx, cross_up(cci, np.full(len(data), -100.)),
                                   cross_dn(cci, np.full(len(data), 100.)),
                                   cci >= 0, cci <= 0, allow_reverse=False),
        "ROC 20 zero-cross": _stateful(idx, cross_up(roc, zeros.astype(float)),
                                       cross_dn(roc, zeros.astype(float)),
                                       roc <= 0, roc >= 0),
        "ADX DI trend": _stateful(idx, (adx > 25) & cross_up(plus, minus),
                                  (adx > 25) & cross_dn(plus, minus),
                                  (plus < minus) | (adx < 20),
                                  (minus < plus) | (adx < 20)),
        "Donchian 20/10 turtle": _stateful(idx, close > prev_hi20, close < prev_lo20,
                                           close < prev_lo10, close > prev_hi10,
                                           allow_reverse=False),
        "RSI 14 30/70 reversion": _stateful(idx, rsi < 30, rsi > 70,
                                            rsi >= 50, rsi <= 50,
                                            allow_reverse=False),
    }


def evaluate_systems(data, train_bars=1440, test_bars=336, cost_bps=10.):
    folds = expanding_folds(len(data), train_bars, test_bars)
    targets = system_targets(data)
    rows = []
    fold_rows = []
    for name, target in targets.items():
        returns, gross, turnovers = [], [], []
        for start, end in folds:
            ledger = backtest_next_open(data.iloc[start:end], target.iloc[start:end],
                                        cost_bps=cost_bps)
            net = float(ledger["equity"].iloc[-1] - 1)
            raw = float((1 + ledger["gross_return"]).prod() - 1)
            turnover = float(ledger["turnover"].sum())
            returns.append(net)
            gross.append(raw)
            turnovers.append(turnover)
            fold_rows.append({"strategy": name, "test_start": str(data.index[start]),
                              "test_end": str(data.index[end - 1]),
                              "test_return": net, "gross_0bps": raw,
                              "turnover": turnover})
        eq = np.cumprod(1 + np.array(returns))
        peaks = np.maximum.accumulate(np.r_[1., eq])
        rows.append({"strategy": name, "folds": len(folds),
                     "compounded_return": float(eq[-1] - 1),
                     "gross_0bps": float(np.prod(1 + np.array(gross)) - 1),
                     "positive_folds": int((np.array(returns) > 0).sum()),
                     "worst_fold_return": float(min(returns)),
                     "fold_boundary_max_drawdown": float((eq / peaks[1:] - 1).min()),
                     "turnover": float(sum(turnovers))})
    return (pd.DataFrame(rows).sort_values("compounded_return", ascending=False).reset_index(drop=True),
            pd.DataFrame(fold_rows))
