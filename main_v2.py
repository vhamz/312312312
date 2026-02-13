"""
Crypto Trading Bot V2 - Regression-Optimized Strategy
Uses cached LLM analysis (llm_cache.json) with scipy-optimized continuous weights.
V2 improvements: continuous scoring weights found via differential_evolution,
MACD momentum confirmation, sentiment-dominated strategy.
"""

import json
import math
import pandas as pd
from collections import defaultdict

OUTPUT_CSV = "trades_log.csv"
STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001  # 0.1%

# ============================================================
# V2 OPTIMIZED WEIGHTS (found by scipy.optimize.differential_evolution)
# ============================================================
W_RSI = 5.8722
W_MACD = 9.9980
W_BB = 10.4224
W_SENTIMENT = 33.3975
W_ACTION = 16.2695
W_VOLATILITY = 7.0974
W_REVERSAL = 5.8520

BUY_SCORE = 9.6302
BUY_RSI_MAX = 54.3964
SELL_SCORE = -8.5240
SELL_SCORE_MILD = 5.8735
SELL_RSI_MIN = 45.4337
SELL_BB = 1.3758
ALLOC_PCT = 0.0200


# ============================================================
# DATA LOADING
# ============================================================

print("Loading LLM cache...")
with open("llm_cache.json") as f:
    cache = json.load(f)
print(f"  Loaded {len(cache)} cached LLM analyses")

# Build lookback for MACD momentum confirmation
by_ticker = defaultdict(list)
for item in cache:
    by_ticker[item["ticker"]].append(item)
for tk in by_ticker:
    by_ticker[tk].sort(key=lambda x: x["date"])

prev_data = {}
for tk, rows in by_ticker.items():
    for i in range(1, len(rows)):
        key = (rows[i]["date"], tk)
        prev_data[key] = rows[i - 1]


# ============================================================
# V2 TRADING LOGIC (Continuous Weighted Scoring)
# ============================================================

def trading_decision_v2(item):
    analysis = item["analysis"]
    indicators = item["indicators"]

    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    reversal_prob = float(analysis.get("reversal_probability", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()
    confidence = float(analysis.get("confidence", 0) or 0)

    rsi = float(indicators.get("rsi", 50))
    macd_hist = float(indicators.get("macd_hist", 0))
    bb_pos = float(indicators.get("bb_position", 0.5))
    volatility = float(indicators.get("volatility_7d", 0))

    # === RISK GATE ===
    if risk_level == "extreme":
        return "sell", "EXTREME RISK flagged by LLM", confidence

    # === CONTINUOUS WEIGHTED SCORING ===
    score = 0.0
    breakdown = []

    # Component 1: RSI (continuous, weight-scaled)
    if rsi < 30:
        c = W_RSI * (30 - rsi) / 30
        score += c
        breakdown.append(f"RSI_oversold:+{c:.1f}")
    elif rsi > 70:
        c = W_RSI * (rsi - 70) / 30
        score -= c
        breakdown.append(f"RSI_overbought:-{c:.1f}")
    else:
        c = W_RSI * (50 - rsi) / 50 * 0.3
        score += c
        breakdown.append(f"RSI_bias:{c:+.1f}")

    # Component 2: MACD Histogram
    if macd_hist > 0:
        score += W_MACD
        breakdown.append(f"MACD_pos:+{W_MACD:.1f}")
    elif macd_hist > -0.001:
        c = W_MACD * 0.25
        score += c
        breakdown.append(f"MACD_flat:+{c:.1f}")
    else:
        c = W_MACD * 0.5
        score -= c
        breakdown.append(f"MACD_neg:-{c:.1f}")

    # Component 3: Bollinger Band Position
    if bb_pos < 0.2:
        c = W_BB * (0.2 - bb_pos) / 0.2
        score += c
        breakdown.append(f"BB_low:+{c:.1f}")
    elif bb_pos > 0.8:
        c = W_BB * (bb_pos - 0.8) / 0.2
        score -= c
        breakdown.append(f"BB_high:-{c:.1f}")

    # Component 4: LLM Sentiment (dominant factor in v2)
    c = W_SENTIMENT * sentiment
    score += c
    breakdown.append(f"Sent:{c:+.1f}")

    # Component 5: LLM Recommended Action
    if llm_action == "buy":
        score += W_ACTION
        breakdown.append(f"LLM_buy:+{W_ACTION:.1f}")
    elif llm_action == "sell":
        score -= W_ACTION
        breakdown.append(f"LLM_sell:-{W_ACTION:.1f}")

    # Component 6: Volatility Penalty
    if volatility > 0.035:
        c = W_VOLATILITY * min((volatility - 0.035) / 0.025, 1.0)
        score -= c
        breakdown.append(f"Vol:-{c:.1f}")

    # Component 7: Reversal Probability
    if reversal_prob > 0.5 and sentiment < 0 and rsi < 40:
        c = W_REVERSAL * reversal_prob
        score += c
        breakdown.append(f"Rev_bounce:+{c:.1f}")
    elif reversal_prob > 0.5 and sentiment > 0 and rsi > 60:
        c = W_REVERSAL * reversal_prob
        score -= c
        breakdown.append(f"Rev_top:-{c:.1f}")

    # === MACD MOMENTUM CONFIRMATION ===
    prev = prev_data.get((item["date"], item["ticker"]))
    macd_improving = True
    if prev:
        prev_macd = float(prev["indicators"].get("macd_hist", 0))
        macd_improving = macd_hist > prev_macd

    # === DECISION THRESHOLDS (regression-optimized) ===
    decision = "hold"
    bd = ", ".join(breakdown)
    reason = f"Neutral (score={score:.1f})"

    # BUY
    if score >= BUY_SCORE and rsi < BUY_RSI_MAX and macd_improving:
        decision = "buy"
        reason = f"BUY: score={score:.1f} [{bd}]"

    # SELL (overrides buy)
    if score <= SELL_SCORE:
        decision = "sell"
        reason = f"SELL: score={score:.1f} [{bd}]"
    elif score <= SELL_SCORE_MILD and rsi > SELL_RSI_MIN:
        decision = "sell"
        reason = f"SELL_MILD: score={score:.1f}, RSI={rsi:.1f} [{bd}]"
    elif bb_pos > SELL_BB:
        decision = "sell"
        reason = f"SELL_BB: BB={bb_pos:.2f} [{bd}]"

    # Emergency RSI override
    if rsi > 75:
        decision = "sell"
        reason = f"EMERGENCY: RSI={rsi:.1f} extreme overbought"

    return decision, reason, confidence


# ============================================================
# GENERATE DECISIONS
# ============================================================

print("\nApplying V2 optimized trading logic...")
results = []
for item in cache:
    decision, reason, confidence = trading_decision_v2(item)
    results.append({
        "date": item["date"],
        "ticker": item["ticker"],
        "price": item["price"],
        "decision": decision,
        "reason": reason,
        "sentiment": f"{float(item['analysis'].get('sentiment_score', 0)):.3f}",
        "rsi": f"{float(item['indicators'].get('rsi', 0)):.2f}",
        "confidence": f"{confidence:.3f}",
    })

# Export
trades_df = pd.DataFrame(results)
trades_df.to_csv(OUTPUT_CSV, index=False)
total = len(trades_df)
buy_count = len(trades_df[trades_df["decision"] == "buy"])
sell_count = len(trades_df[trades_df["decision"] == "sell"])
hold_count = len(trades_df[trades_df["decision"] == "hold"])
print(f"\nExported {total} trades to {OUTPUT_CSV}")
print(f"\nDecision Distribution:")
print(f"  Buy:  {buy_count} ({buy_count/total*100:.1f}%)")
print(f"  Sell: {sell_count} ({sell_count/total*100:.1f}%)")
print(f"  Hold: {hold_count} ({hold_count/total*100:.1f}%)")


# ============================================================
# BACKTESTING
# ============================================================

print("\n=== BACKTESTING ===")
capital = STARTING_CAPITAL
positions = {}
daily_portfolio_values = []
trade_results = []

dates = sorted(trades_df["date"].unique())
for date in dates:
    day_trades = trades_df[trades_df["date"] == date]
    for _, trade in day_trades.iterrows():
        ticker = trade["ticker"]
        price = float(trade["price"])
        decision = trade["decision"]

        if decision == "buy" and capital > 0:
            alloc = capital * ALLOC_PCT
            fee = alloc * TRANSACTION_FEE
            invest = alloc - fee
            qty = invest / price
            if ticker not in positions:
                positions[ticker] = {"qty": 0, "avg_price": 0}
            old_qty = positions[ticker]["qty"]
            old_cost = old_qty * positions[ticker]["avg_price"]
            new_qty = old_qty + qty
            positions[ticker]["qty"] = new_qty
            positions[ticker]["avg_price"] = (old_cost + invest) / new_qty if new_qty > 0 else 0
            capital -= alloc
            trade_results.append({"date": date, "ticker": ticker, "action": "buy", "price": price, "qty": qty, "fee": fee})

        elif decision == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
            qty = positions[ticker]["qty"]
            proceeds = qty * price
            fee = proceeds * TRANSACTION_FEE
            net = proceeds - fee
            pnl = net - (qty * positions[ticker]["avg_price"])
            capital += net
            trade_results.append({"date": date, "ticker": ticker, "action": "sell", "price": price, "qty": qty, "fee": fee, "pnl": pnl})
            positions[ticker] = {"qty": 0, "avg_price": 0}

    portfolio_value = capital
    for ticker, pos in positions.items():
        if pos["qty"] > 0:
            ticker_row = day_trades[day_trades["ticker"] == ticker]
            if not ticker_row.empty:
                portfolio_value += pos["qty"] * float(ticker_row.iloc[0]["price"])
    daily_portfolio_values.append({"date": date, "value": portfolio_value})

pv = pd.DataFrame(daily_portfolio_values)
if len(pv) > 1:
    pv["daily_return"] = pv["value"].pct_change()
    pv = pv.dropna()
    avg_return = pv["daily_return"].mean()
    std_return = pv["daily_return"].std()
    sharpe = (avg_return / std_return) * math.sqrt(365) if std_return > 0 else 0
    total_return = (pv["value"].iloc[-1] / STARTING_CAPITAL - 1) * 100
    max_val = pv["value"].cummax()
    drawdown = ((pv["value"] - max_val) / max_val).min() * 100
    sell_trades = [t for t in trade_results if t["action"] == "sell"]
    wins = len([t for t in sell_trades if t.get("pnl", 0) > 0])
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
    buy_trades = [t for t in trade_results if t["action"] == "buy"]

    print(f"  Sharpe Ratio:  {sharpe:.4f}")
    print(f"  Total Return:  {total_return:+.2f}%")
    print(f"  Max Drawdown:  {drawdown:.2f}%")
    print(f"  Win Rate:      {win_rate:.1f}%")
    print(f"  Total Trades:  {len(trade_results)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
    print(f"  Final Value:   ${pv['value'].iloc[-1]:.2f}")

    print(f"\n{'='*60}")
    print(f"V1 baseline:  Sharpe=1.8622, Return=+3.37%")
    print(f"V2 optimized: Sharpe={sharpe:.4f}, Return={total_return:+.2f}%")
    if sharpe > 1.8622:
        print(f"  IMPROVEMENT: +{(sharpe - 1.8622):.4f} Sharpe")
