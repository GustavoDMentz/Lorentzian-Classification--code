# Stateful trading systems — experimental protocol

This module is a **new experiment**. The old `indicator_tournament.py` remains unchanged as a simplified-rule baseline.

All ten systems use fixed, predeclared parameters, a state machine with persistent long/short/flat positions, signals at bar close, fills at next bar open, and the same 10-bps-per-unit turnover model. The backtest resets positions at each test fold. Indicators are computed using only current/past candles; the preceding history is used solely for warmup. There is **no parameter search** in the stateful tournament.

| System | Long entry / short entry | Exit |
| --- | --- | --- |
| EMA 12/26 | Fast EMA crosses above/below slow EMA | Opposite crossover |
| SMA 50/200 | Fast SMA crosses above/below slow SMA | Opposite crossover |
| MACD | MACD crosses above/below signal | Opposite crossover |
| Bollinger 20, 2σ | Close below lower / above upper band | Return to middle band |
| Stochastic 5/3/3 | %K crosses %D in oversold/overbought zone | %K crosses 50 |
| CCI 20 | CCI crosses above −100 / below +100 | CCI crosses zero |
| ROC 20 | ROC crosses above/below zero | Opposite zero crossing |
| ADX +DI/−DI | ADX>25 and +DI/−DI crosses | Opposite DI or ADX<20 |
| Donchian 20/10 | Close breaks prior 20-bar high/low | Close crosses prior 10-bar opposite channel |
| RSI 14 | RSI below 30 / above 70 | RSI crosses 50 |

**Important limitations:** These are explicitly defined *research interpretations*, not canonical replicas of every author's full trading methodology. Donchian uses close-confirmed breakouts rather than stop orders at channel levels; intrabar stop/limit fills are not modeled. No volatility position sizing, borrowing/funding, slippage, spread, or liquidation. Drawdown is measured at fold boundaries. Both long and short trades are permitted in the simulation; KuCoin spot does not itself provide uncollateralized shorting. Returns are not live-tradable performance.

**Fairness boundary:** The ten stateful systems are compared fairly to each other in execution and with fixed parameter rules, but the separately reported *optimized* RSI and Lorentzian use different search budgets and should not be interpreted as a controlled head-to-head. The old simple-rule leaderboard also remains separate. Comparing the full-system leaderboard against previous results is exploratory because the design was informed by earlier observed performance.

For genuine model selection, reserve a future unseen period or different asset/exchange as a locked validation set. Do not select a winner based solely on the historical ranking.
