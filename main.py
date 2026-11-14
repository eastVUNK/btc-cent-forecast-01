# btc-cent-forecast-01 — v2: median mean-reversion forecaster (Python 3.13, stdlib only)
#
# Formula (v2 — the v1 momentum term is GONE; see the evidence below):
#
#   Let P0 be the last close (the checkpoint close) and M the median of all
#   closes in the case (4,320 values, 72 hours).  Work in log prices.
#
#       dev = ln(P0 / M)
#       forecast = P0 * exp(-0.30 * dev)   if |dev| > 0.015
#       forecast = P0                      otherwise
#
#   In words: when the price sits more than 1.5% away from its 72-hour
#   median, move 30% of the way (in log terms) back toward that median;
#   otherwise forecast persistence (no move).  The result is clamped to the
#   scorer's (0, 10,000,000) range and rounded to the 5-decimal price grid.
#
# Why v2 drops v1's momentum (evidence from the Stage 2 rolling backtest,
# backtest.py — NOT shipped in the repo):
#   * Harness: for each practice case p00-p05, origins every 15 min from
#     minute 240 to 4065 (target origin+240 stays inside the case), 256
#     origins per case, 1,536 samples total. The harness was validated to
#     reproduce main.py v1's forecasts on the six full cases exactly.
#   * v1 on that backtest scored 35,091c — WORSE than pure persistence
#     (34,815c). Its 6-case practice win (39,184c vs 44,076c) was luck
#     concentrated in p02/p03, not a real edge: momentum hurts on average.
#   * Grid over shrink {0.04..0.12}, 1h/4h blend {0.5,0.6,0.7}, pull
#     {0.03,0.05,0.08}, threshold {0.01,0.02,0.03} (135 configs) plus a
#     120/480-min drift variant: smaller shrink was monotonically better,
#     stronger pull monotonically better. Probing past the grid edge
#     (pull 0.10/0.12/0.15/0.20/0.25/0.30/0.40, threshold 0.015/0.02,
#     median windows 24h/48h/72h) found a genuine interior peak at
#     pull 0.30, threshold 0.015, full-history median — pull 0.40 is worse
#     again (32,516c) and the pure-median forecast is terrible (70,334c),
#     so 0.30 is a peak, not a runaway edge.
#   * v2 backtest MAE: 31,915c pooled (vs persistence 34,815c, v1 35,091c);
#     late origins only (i >= 2880, 48h+ of history, closest to the real
#     72h case shape): 36,293c (vs persistence 39,820c, v1 40,212c);
#     beats persistence on ALL 6 case-windows (26,641 / 27,164 / 29,171 /
#     34,045 / 36,810 / 37,662c) and in both split halves.
#   * A token drift term (shrink 0.02) adds nothing at pull 0.30
#     (31,941c pooled), so it was dropped: fewer knobs, less overfit.
#   * Practice 6-case MAE (p05 via a Coinbase close proxy, approx):
#     v2 38,893c vs v1 39,184c vs persistence 44,076c (official sealed
#     persistence: 44,521.58c). The 6-case number is noisy (SE ~7,000c);
#     the backtest is the ranking evidence, the practice number is a check.
#   * Known risk: in a hard one-way trend (waterfall) the median lags and
#     the pull bets against the trend; the backtest's volatile late half
#     still favoured it, but the final window (15-17 Nov) is a new regime.
#
# Deterministic: no randomness, no time, no network; the same case file
# always yields the same bytes.  Reads every *.in file in the input
# directory (MAKEFUN_INPUT_DIR or /input) and writes one *.out per case
# (MAKEFUN_OUTPUT_DIR or /output) as {"forecasts": [{"case": id, "price": p}]}.
"""Forecast BTC's price 240 minutes after each checkpoint."""

import json
import math
import os
from pathlib import Path

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# v2 constants (tuned by the Stage 2 rolling backtest; see header).
REVERT_THRESHOLD = 0.015   # |ln(P0 / median)| above which we pull back
REVERT_STRENGTH = 0.30     # fraction of the deviation pulled back (log terms)


def price5(x: float) -> float:
    """Clamp to the scorer's range and round to the 5-decimal price grid."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def median_of(prices) -> float:
    """Median of the closes (finite, positive only)."""
    ordered = sorted(p for p in prices if math.isfinite(p) and p > 0)
    if not ordered:
        raise ValueError("market has no usable prices")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def forecast(prices) -> float:
    """One forecast from a list of 1-minute closes, oldest first."""
    if not prices:
        raise ValueError("market has no prices")
    p0 = float(prices[-1])
    if not math.isfinite(p0) or p0 <= 0:
        raise ValueError("last price is not a finite positive number")

    # Mean reversion toward the full-history median, only on a real stretch.
    # Short histories (<100 closes) skip the pull and stay at persistence.
    if len(prices) >= 100:
        dev = math.log(p0 / median_of(prices))
        if abs(dev) > REVERT_THRESHOLD:
            return price5(p0 * math.exp(-REVERT_STRENGTH * dev))
    return price5(p0)


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(p.name for p in input_dir.glob("*.in")):
        case_id = name[: -len(".in")]
        with open(input_dir / name, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{name} does not hold exactly one market")
        price = forecast(markets[0]["prices"])
        out = {"forecasts": [{"case": case_id, "price": price}]}
        with open(output_dir / f"{case_id}.out", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
