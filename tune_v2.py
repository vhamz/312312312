"""
V2 Threshold tuning - uses scipy differential_evolution for global optimization.
Optimizes all scoring weights + decision thresholds simultaneously.
"""

import json
import math
import numpy as np
from scipy.optimize import differential_evolution
from collections import defaultdict

STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001

with open("llm_cache.json") as f:
    cache = json.load(f)

# Pre-index cache by (date, ticker) for lookback
by_ticker = defaultdict(list)
for item in cache:
    by_ticker[item["ticker"]].append(item)
for tk in by_ticker:
    by_ticker[tk].sort(key=lambda x: x["date"])

prev_data = {}
for tk, rows in by_ticker.items():
    for i in range(1, len(rows)):
        key = (rows[i]["date"], tk)
        prev_data[key] = rows[i-1]

# Pre-extract all numeric features once
features = []
for item in cache:
    a = item["analysis"]
    ind = item["indicators"]
    features.append({
        "date": item["date"],
        "ticker": item["ticker"],
        "price": item["price"],
        "sentiment": float(a.get("sentiment_score", 0) or 0),
        "reversal_prob": float(a.get("reversal_probability", 0) or 0),
        "risk_level": str(a.get("risk_level", "medium")).lower(),
        "llm_action": str(a.get("recommended_action", "hold")).lower(),
        "rsi": float(ind.get("rsi", 50)),
        "macd_hist": float(ind.get("macd_hist", 0)),
        "bb_pos": float(ind.get("bb_position", 0.5)),
        "volatility": float(ind.get("volatility_7d", 0)),
    })


def evaluate(params_vec):
    """
    Objective function: returns -Sharpe (we minimize).

    params_vec layout (14 continuous params):
    [0]  w_rsi         - weight for RSI component
    [1]  w_macd        - weight for MACD component
    [2]  w_bb          - weight for BB component
    [3]  w_sentiment   - weight for LLM sentiment
    [4]  w_action      - weight for LLM action
    [5]  w_volatility  - weight for volatility penalty
    [6]  w_reversal    - weight for reversal probability
    [7]  buy_score     - buy threshold
    [8]  buy_rsi_max   - max RSI for buy
    [9]  sell_score    - strong sell threshold
    [10] sell_score_mild - mild sell threshold
    [11] sell_rsi_min  - min RSI for mild sell
    [12] sell_bb       - BB sell threshold
    [13] alloc_pct     - allocation per trade
    """
    (w_rsi, w_macd, w_bb, w_sentiment, w_action, w_volatility, w_reversal,
     buy_score, buy_rsi_max, sell_score, sell_score_mild, sell_rsi_min,
     sell_bb, alloc_pct) = params_vec

    # Generate decisions
    decisions = []
    for f in features:
        if f["risk_level"] == "extreme":
            decisions.append({"date": f["date"], "ticker": f["ticker"], "price": f["price"], "decision": "sell"})
            continue

        # Weighted scoring
        score = 0.0
        rsi, macd_hist, bb_pos = f["rsi"], f["macd_hist"], f["bb_pos"]
        volatility, sentiment = f["volatility"], f["sentiment"]
        reversal_prob = f["reversal_prob"]

        # RSI component (normalized to -1..+1)
        if rsi < 30:
            score += w_rsi * (30 - rsi) / 30  # 0..+1 when oversold
        elif rsi > 70:
            score -= w_rsi * (rsi - 70) / 30  # 0..-1 when overbought
        else:
            score += w_rsi * (50 - rsi) / 50 * 0.3  # small bias

        # MACD component
        if macd_hist > 0:
            score += w_macd
        elif macd_hist > -0.001:
            score += w_macd * 0.25
        else:
            score -= w_macd * 0.5

        # BB component
        if bb_pos < 0.2:
            score += w_bb * (0.2 - bb_pos) / 0.2
        elif bb_pos > 0.8:
            score -= w_bb * (bb_pos - 0.8) / 0.2

        # LLM sentiment
        score += w_sentiment * sentiment

        # LLM action
        if f["llm_action"] == "buy":
            score += w_action
        elif f["llm_action"] == "sell":
            score -= w_action

        # Volatility penalty
        if volatility > 0.035:
            score -= w_volatility * min((volatility - 0.035) / 0.025, 1.0)

        # Reversal probability
        if reversal_prob > 0.5 and sentiment < 0 and rsi < 40:
            score += w_reversal * reversal_prob
        elif reversal_prob > 0.5 and sentiment > 0 and rsi > 60:
            score -= w_reversal * reversal_prob

        # MACD momentum confirmation (compare to prev day)
        prev = prev_data.get((f["date"], f["ticker"]))
        macd_improving = True
        if prev:
            prev_macd = float(prev["indicators"].get("macd_hist", 0))
            macd_improving = macd_hist > prev_macd

        # Decision
        decision = "hold"
        if score >= buy_score and rsi < buy_rsi_max and macd_improving:
            decision = "buy"
        if score <= sell_score:
            decision = "sell"
        elif score <= sell_score_mild and rsi > sell_rsi_min:
            decision = "sell"
        elif bb_pos > sell_bb:
            decision = "sell"
        if rsi > 75:
            decision = "sell"

        decisions.append({"date": f["date"], "ticker": f["ticker"], "price": f["price"], "decision": decision})

    # Run backtest
    capital = STARTING_CAPITAL
    positions = {}
    daily_values = []
    buys = 0
    sells = 0

    dates = sorted(set(d["date"] for d in decisions))
    for date in dates:
        day_rows = [d for d in decisions if d["date"] == date]
        for r in day_rows:
            ticker, price, dec = r["ticker"], r["price"], r["decision"]
            if dec == "buy" and capital > 100:
                alloc = capital * alloc_pct
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price
                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0}
                old = positions[ticker]
                old_cost = old["qty"] * old["avg_price"]
                new_qty = old["qty"] + qty
                positions[ticker] = {"qty": new_qty, "avg_price": (old_cost + invest) / new_qty if new_qty > 0 else 0}
                capital -= alloc
                buys += 1
            elif dec == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                qty = positions[ticker]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                capital += proceeds - fee
                positions[ticker] = {"qty": 0, "avg_price": 0}
                sells += 1

        pv = capital
        prices_today = {r["ticker"]: r["price"] for r in day_rows}
        for ticker, pos in positions.items():
            if pos["qty"] > 0 and ticker in prices_today:
                pv += pos["qty"] * prices_today[ticker]
        daily_values.append(pv)

    # Require minimum trade activity for robustness
    if buys < 5 or sells < 3 or len(daily_values) < 2:
        return 0.0

    returns = [(daily_values[i] / daily_values[i-1] - 1) for i in range(1, len(daily_values))]
    avg_ret = sum(returns) / len(returns)
    std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
    sharpe = (avg_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0
    total_ret = (daily_values[-1] / STARTING_CAPITAL - 1) * 100

    max_val = daily_values[0]
    max_dd = 0
    for v in daily_values:
        max_val = max(max_val, v)
        dd = (v - max_val) / max_val
        max_dd = min(max_dd, dd)

    # Penalize extreme drawdown
    if max_dd < -0.10:
        sharpe *= 0.5

    # Return negative Sharpe (we minimize)
    return -sharpe


# === OPTIMIZATION ===
print("V2 Optimization with differential_evolution...")
print("Optimizing 14 parameters: 7 weights + 7 thresholds")
print("=" * 80)

# Bounds: (min, max) for each parameter
bounds = [
    (5, 40),      # w_rsi
    (5, 30),      # w_macd
    (5, 30),      # w_bb
    (5, 35),      # w_sentiment
    (2, 20),      # w_action
    (0, 20),      # w_volatility
    (0, 15),      # w_reversal
    (5, 50),      # buy_score
    (25, 60),     # buy_rsi_max
    (-35, 0),     # sell_score (strong)
    (-20, 10),    # sell_score_mild
    (30, 55),     # sell_rsi_min
    (0.6, 1.5),   # sell_bb
    (0.02, 0.12), # alloc_pct
]

callback_count = [0]
best_so_far = [0]

def callback(xk, convergence):
    callback_count[0] += 1
    val = evaluate(xk)
    if -val > best_so_far[0]:
        best_so_far[0] = -val
    if callback_count[0] % 5 == 0:
        print(f"  Generation {callback_count[0]}: best Sharpe so far = {best_so_far[0]:+.4f} (convergence={convergence:.4f})")

result = differential_evolution(
    evaluate,
    bounds,
    seed=42,
    maxiter=200,
    popsize=25,
    tol=1e-6,
    mutation=(0.5, 1.5),
    recombination=0.8,
    callback=callback,
    workers=1,  # features list is shared
)

print(f"\n{'='*80}")
print(f"Optimization finished: {result.message}")
print(f"Best Sharpe: {-result.fun:+.4f}")

names = ["w_rsi", "w_macd", "w_bb", "w_sentiment", "w_action", "w_volatility", "w_reversal",
         "buy_score", "buy_rsi_max", "sell_score", "sell_score_mild", "sell_rsi_min",
         "sell_bb", "alloc_pct"]

print(f"\nOptimal parameters:")
for name, val in zip(names, result.x):
    print(f"  {name:20s} = {val:.4f}")

# Run final evaluation with best params and print detailed metrics
neg_sharpe = evaluate(result.x)
# Re-run backtest to get all metrics
(w_rsi, w_macd, w_bb, w_sentiment, w_action, w_volatility, w_reversal,
 buy_score, buy_rsi_max, sell_score, sell_score_mild, sell_rsi_min,
 sell_bb, alloc_pct) = result.x

# Count decisions
decisions = []
for f_item in features:
    if f_item["risk_level"] == "extreme":
        decisions.append("sell")
        continue
    score = 0.0
    rsi, macd_hist, bb_pos = f_item["rsi"], f_item["macd_hist"], f_item["bb_pos"]
    volatility, sentiment = f_item["volatility"], f_item["sentiment"]
    reversal_prob = f_item["reversal_prob"]
    if rsi < 30: score += w_rsi * (30 - rsi) / 30
    elif rsi > 70: score -= w_rsi * (rsi - 70) / 30
    else: score += w_rsi * (50 - rsi) / 50 * 0.3
    if macd_hist > 0: score += w_macd
    elif macd_hist > -0.001: score += w_macd * 0.25
    else: score -= w_macd * 0.5
    if bb_pos < 0.2: score += w_bb * (0.2 - bb_pos) / 0.2
    elif bb_pos > 0.8: score -= w_bb * (bb_pos - 0.8) / 0.2
    score += w_sentiment * sentiment
    if f_item["llm_action"] == "buy": score += w_action
    elif f_item["llm_action"] == "sell": score -= w_action
    if volatility > 0.035: score -= w_volatility * min((volatility - 0.035) / 0.025, 1.0)
    if reversal_prob > 0.5 and sentiment < 0 and rsi < 40: score += w_reversal * reversal_prob
    elif reversal_prob > 0.5 and sentiment > 0 and rsi > 60: score -= w_reversal * reversal_prob
    prev = prev_data.get((f_item["date"], f_item["ticker"]))
    macd_improving = True
    if prev:
        prev_macd = float(prev["indicators"].get("macd_hist", 0))
        macd_improving = macd_hist > prev_macd
    decision = "hold"
    if score >= buy_score and rsi < buy_rsi_max and macd_improving: decision = "buy"
    if score <= sell_score: decision = "sell"
    elif score <= sell_score_mild and rsi > sell_rsi_min: decision = "sell"
    elif bb_pos > sell_bb: decision = "sell"
    if rsi > 75: decision = "sell"
    decisions.append(decision)

from collections import Counter
dist = Counter(decisions)
print(f"\nDecision distribution: {dict(dist)}")
print(f"  Buy:  {dist.get('buy', 0)} ({dist.get('buy', 0)/len(decisions)*100:.1f}%)")
print(f"  Sell: {dist.get('sell', 0)} ({dist.get('sell', 0)/len(decisions)*100:.1f}%)")
print(f"  Hold: {dist.get('hold', 0)} ({dist.get('hold', 0)/len(decisions)*100:.1f}%)")

# Compare with v1
print(f"\n{'='*80}")
print(f"V1 baseline: Sharpe=1.8622, Return=+3.37%")
print(f"V2 optimized: Sharpe={-result.fun:+.4f}")
if -result.fun > 1.8622:
    print(f"  IMPROVEMENT: +{(-result.fun - 1.8622):.4f} Sharpe")
else:
    print(f"  No improvement over v1")
