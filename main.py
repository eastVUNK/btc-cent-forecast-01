# btc-cent-forecast-01 — v1: damped-momentum forecaster (Python 3.13, stdlib only)
#
# Formula (v1, deliberately simple — to be tuned against the practice cases
# in Stage 2 with rolling backtests inside each case's own 72h window):
#
#   Let P0 be the last close (the checkpoint close) and P(-k) the close k
#   minutes earlier.  Work in log prices so the forecast stays positive.
#
#   1. Short-window drift.  r60 = ln(P0 / P(-60)) and r240 = ln(P0 / P(-240))
#      are the log-returns over the last hour and the last four hours.
#      Scale each to a 240-minute horizon and blend them:
#          drift240 = 0.6 * (4 * r60) + 0.4 * r240
#      (the 1h return gets the bigger weight: it is the freshest signal, but
#      scaling it x4 also scales its noise, which is why step 2 shrinks it.)
#
#   2. Hard shrinkage toward persistence.  BTC's 4h drift is mostly noise and
#      persistence (forecast = P0) is a strong baseline (practice MAE
#      44,521.58 cents).  Keep only 8% of the estimated drift:
#          drift_term = 0.08 * drift240
#
#   3. Extreme-deviation mean reversion.  Let M be the median of all 4,320
#      closes (72h) and dev = ln(P0 / M).  Only when the price sits more than
#      2% away from its 72h median do we pull a little back toward it:
#          pull = -0.05 * dev   if |dev| > 0.02, else 0
#      (5% of the deviation — a nudge, not a snap-back; ordinary days are
#      untouched so the forecaster stays close to persistence.)
#
#   4. Forecast:  P = P0 * exp(drift_term + pull), clamped to the scorer's
#      (0, 10,000,000) range and rounded to the price grid's 5 decimals.
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

# v1 constants (see header comment for the reasoning).
DRIFT_SHRINK = 0.08        # fraction of the estimated 240-min drift kept
SHORT_WEIGHT = 0.6         # weight on the scaled 1h return vs the 4h return
REVERT_THRESHOLD = 0.02    # |ln(P0 / median72)| above which we pull back
REVERT_STRENGTH = 0.05     # fraction of the deviation pulled back


def price5(x: float) -> float:
    """Clamp to the scorer's range and round to the 5-decimal price grid."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def forecast(prices) -> float:
    """One forecast from a list of 1-minute closes, oldest first."""
    if not prices:
        raise ValueError("market has no prices")
    p0 = float(prices[-1])
    if not math.isfinite(p0) or p0 <= 0:
        raise ValueError("last price is not a finite positive number")

    n = len(prices)

    def log_return(k: int) -> float:
        """ln(P0 / P(-k)); 0.0 when the window is unavailable or invalid."""
        if n <= k:
            return 0.0
        pk = float(prices[-1 - k])
        if not math.isfinite(pk) or pk <= 0:
            return 0.0
        return math.log(p0 / pk)

    # 1-2. Damped momentum: blend the 1h and 4h log-returns, scale to the
    # 240-minute horizon, then shrink hard toward zero (persistence).
    r60 = log_return(60)
    r240 = log_return(240)
    drift240 = SHORT_WEIGHT * (4.0 * r60) + (1.0 - SHORT_WEIGHT) * r240
    drift_term = DRIFT_SHRINK * drift240

    # 3. Mean reversion, only on extreme deviation from the 72h median.
    pull = 0.0
    if n >= 100:
        ordered = sorted(float(p) for p in prices if math.isfinite(p) and p > 0)
        if ordered:
            mid = len(ordered) // 2
            median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
            dev = math.log(p0 / median)
            if abs(dev) > REVERT_THRESHOLD:
                pull = -REVERT_STRENGTH * dev

    # 4. Combine and clamp.
    return price5(p0 * math.exp(drift_term + pull))


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
