# Kraken OOB/USD — read-only microstructure study

Run `python research/kraken_oob_observer.py --samples 30 --interval 10`.

The script first checks Kraken's public `AssetPairs` for an **active OOB/USD** market. If absent, it stops rather than substituting another pair. No credentials, private endpoints, or orders. Output is append-only `research_data/kraken_oob/snapshots.csv` and `raw.jsonl` (add this directory to your local gitignore; do not commit captured market data).

Each sample requests public `Depth` (25 levels) and `Ticker`, recording UTC timestamps, best bid/ask, percentage quoted spread, ten-level quote-notional depths, imbalance, last price and 24-hour volume. **REST responses are not synchronized**: they are observational snapshots, not executable quotes or a historical replay. Polling every 10 seconds cannot detect transient changes between samples.

The experiment has no predictive or profitability claim. First collect several days, then define fixed train/test time splits and compare momentum, reversal and order-book imbalance against a no-trade baseline, accounting for taker/maker fees, spread, latency, partial fills and impact. Avoid fitting and reporting on the same period. The Kraken public OHLC endpoint offers limited recent history, not years of historical microstructure data.
