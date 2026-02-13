"""
Threshold tuning script - grid search over trading logic parameters.
Uses cached LLM analysis (llm_cache.json) - no API calls needed.
"""

import json
import math

STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001

with open("llm_cache.json") as f:
    cache = json.load(f)


def trading_decision(analysis, indicators, params):
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    confidence = float(analysis.get("confidence", 0) or 0)
    reversal_prob = float(analysis.get("reversal_probability", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()

    rsi = float(indicators.get("rsi", 50))
    macd_hist = float(indicators.get("macd_hist", 0))
    bb_pos = float(indicators.get("bb_position", 0.5))
    volatility = float(indicators.get("volatility_7d", 0))

    # Risk gates
    if volatility > params["vol_gate"]:
        return "hold"
    if risk_level == "extreme":
        return "sell"

    # Scoring
    score = 0
    if rsi < 25: score += 25
    elif rsi < 30: score += 18
    elif rsi < 40: score += 5
    elif rsi > 70: score -= 25
    elif rsi > 60: score -= 8

    if macd_hist > 0: score += 20
    elif macd_hist > -0.001: score += 5
    else: score -= 10

    if bb_pos < 0.1: score += 20
    elif bb_pos < 0.2: score += 12
    elif bb_pos > 0.9: score -= 20
    elif bb_pos > 0.8: score -= 12

    score += round(sentiment * 25)

    if llm_action == "buy": score += 10
    elif llm_action == "sell": score -= 10

    if volatility > 0.045: score -= 15
    elif volatility > 0.035: score -= 8

    if reversal_prob > 0.6 and sentiment < 0 and rsi < 35: score += 10
    elif reversal_prob > 0.6 and sentiment > 0 and rsi > 60: score -= 10

    # Decision
    decision = "hold"

    # Buy conditions
    if score >= params["buy_score"] and rsi < params["buy_rsi_max"]:
        decision = "buy"

    # Sell conditions (checked after buy, can override)
    if score <= params["sell_score"]:
        decision = "sell"
    elif score <= params["sell_score_mild"] and rsi > params["sell_rsi_min"]:
        decision = "sell"
    elif bb_pos > params["sell_bb"]:
        decision = "sell"

    if rsi > 75:
        decision = "sell"

    return decision


def run_backtest(results, alloc_pct=0.1):
    capital = STARTING_CAPITAL
    positions = {}
    daily_values = []
    trade_results = []

    dates = sorted(set(r["date"] for r in results))
    for date in dates:
        day_rows = [r for r in results if r["date"] == date]
        for r in day_rows:
            ticker, price, decision = r["ticker"], r["price"], r["decision"]
            if decision == "buy" and capital > 100:
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
                trade_results.append({"action": "buy", "pnl": 0})
            elif decision == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                qty = positions[ticker]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                net = proceeds - fee
                pnl = net - (qty * positions[ticker]["avg_price"])
                capital += net
                trade_results.append({"action": "sell", "pnl": pnl})
                positions[ticker] = {"qty": 0, "avg_price": 0}

        pv = capital
        prices_today = {r["ticker"]: r["price"] for r in day_rows}
        for ticker, pos in positions.items():
            if pos["qty"] > 0 and ticker in prices_today:
                pv += pos["qty"] * prices_today[ticker]
        daily_values.append(pv)

    if len(daily_values) > 1:
        returns = [(daily_values[i] / daily_values[i-1] - 1) for i in range(1, len(daily_values))]
        avg_ret = sum(returns) / len(returns)
        std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0
        total_ret = (daily_values[-1] / STARTING_CAPITAL - 1) * 100
        buys = sum(1 for t in trade_results if t["action"] == "buy")
        sells_list = [t for t in trade_results if t["action"] == "sell"]
        wins = sum(1 for t in sells_list if t["pnl"] > 0)
        return {
            "sharpe": sharpe, "total_return": total_ret,
            "buys": buys, "sells": len(sells_list),
            "win_rate": (wins / len(sells_list) * 100) if sells_list else 0,
            "final_value": daily_values[-1],
        }
    return None


# === GRID SEARCH ===
print("Running grid search over threshold parameters...")
print("=" * 80)

best_sharpe = -999
best_params = None
best_metrics = None
results_list = []

for buy_score in [15, 20, 25, 30, 40, 50]:
    for buy_rsi_max in [28, 35, 50]:
        for sell_score in [-25, -15, -5, 0]:
            for sell_score_mild in [-10, -5, 0, 5]:
                for sell_rsi_min in [40, 50]:
                    for sell_bb in [0.8, 1.0, 1.1]:
                        for vol_gate in [0.06, 0.10, 1.0]:
                            for alloc in [0.05, 0.10]:
                                params = {
                                    "buy_score": buy_score,
                                    "buy_rsi_max": buy_rsi_max,
                                    "sell_score": sell_score,
                                    "sell_score_mild": sell_score_mild,
                                    "sell_rsi_min": sell_rsi_min,
                                    "sell_bb": sell_bb,
                                    "vol_gate": vol_gate,
                                }

                                decisions = []
                                for item in cache:
                                    d = trading_decision(item["analysis"], item["indicators"], params)
                                    decisions.append({"date": item["date"], "ticker": item["ticker"], "price": item["price"], "decision": d})

                                metrics = run_backtest(decisions, alloc_pct=alloc)
                                if metrics and metrics["buys"] > 0 and metrics["sells"] > 0:
                                    if metrics["sharpe"] > best_sharpe:
                                        best_sharpe = metrics["sharpe"]
                                        best_params = {**params, "alloc": alloc}
                                        best_metrics = metrics
                                        results_list.append((metrics["sharpe"], params, alloc, metrics))

# Sort by Sharpe
results_list.sort(key=lambda x: x[0], reverse=True)

print(f"\nTop 10 parameter sets:")
for i, (sharpe, params, alloc, metrics) in enumerate(results_list[:10]):
    print(f"  #{i+1} Sharpe={sharpe:+.4f} Ret={metrics['total_return']:+.2f}% "
          f"Buys={metrics['buys']} Sells={metrics['sells']} WR={metrics['win_rate']:.0f}% "
          f"| buy>={params['buy_score']} rsi<{params['buy_rsi_max']} "
          f"sell<={params['sell_score']}/{params['sell_score_mild']} rsi>{params['sell_rsi_min']} "
          f"bb>{params['sell_bb']} vol>{params['vol_gate']} alloc={alloc}")

if best_params:
    print(f"\n{'='*80}")
    print(f"BEST: Sharpe={best_sharpe:+.4f}")
    print(f"  Params: {best_params}")
    print(f"  Metrics: {best_metrics}")
