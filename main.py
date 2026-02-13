"""
Crypto Trading Bot - Sharpe Ratio Optimized Strategy
Uses OpenRouter API (Gemini 3 Flash Preview) for news sentiment analysis
Combined with 7-component technical scoring system
"""

import os
import pandas as pd
import requests
import json
import time
import math
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "google/gemini-2.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

FEATURES_CSV = "crypto_features_3months.csv"
NEWS_CSV = "crypto_news_3months.csv"
OUTPUT_CSV = "trades_log.csv"

STARTING_CAPITAL = 10000
TRANSACTION_FEE = 0.001  # 0.1%


# ============================================================
# MODULE 1: DATA LOADING
# ============================================================

def load_data():
    print("Loading market data...")
    features = pd.read_csv(FEATURES_CSV)
    print(f"  Market data: {len(features)} rows, {len(features.columns)} columns")

    print("Loading news data...")
    news = pd.read_csv(NEWS_CSV)
    print(f"  News data: {len(news)} rows")

    # Group news by date
    news_by_date = {}
    for _, row in news.iterrows():
        date = str(row["date"]).strip()
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        if date not in news_by_date:
            news_by_date[date] = []
        news_by_date[date].append(f"{title}: {body[:200]}")

    # Merge news text into features
    def get_news_text(date):
        date_str = str(date).strip()
        articles = news_by_date.get(date_str, [])
        if articles:
            return "\n\n".join(articles[:3])
        return "No significant news today"

    features["news_text"] = features["date"].apply(get_news_text)

    # Sort by date and ticker for chronological processing
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"  Tickers: {sorted(features['ticker'].unique())}")
    print(f"  Rows with news: {(features['news_text'] != 'No significant news today').sum()}")

    return features


# ============================================================
# MODULE 2: LLM PROMPT BUILDER
# ============================================================

def build_prompt(row):
    close = float(row.get("close", 0))
    ma7 = float(row.get("ma7", 0))
    ma20 = float(row.get("ma20", 0)) if pd.notna(row.get("ma20")) else 0
    ma50 = float(row.get("ma50", 0)) if pd.notna(row.get("ma50")) else 0
    rsi = float(row.get("rsi", 50))
    macd_hist = float(row.get("macd_hist", 0))
    bb_pos = float(row.get("bb_position", 0.5))
    vol7d = float(row.get("volatility_7d", 0))
    returns = float(row.get("returns", 0))
    volume_change = float(row.get("volume_change", 0))
    bb_upper = float(row.get("bb_upper", 0))
    bb_lower = float(row.get("bb_lower", 0))

    # Pre-compute labels
    above_ma7 = "above" if close > ma7 else "below"
    above_ma20 = ("above" if close > ma20 else "below") if ma20 > 0 else "N/A"
    above_ma50 = ("above" if close > ma50 else "below") if ma50 > 0 else "N/A"
    ma_crossover = "bullish (MA7 > MA20)" if (ma7 > ma20 and ma20 > 0) else "bearish (MA7 < MA20)"
    macd_trend = "positive (bullish momentum)" if macd_hist > 0 else "negative (bearish momentum)"

    if rsi < 30:
        rsi_zone = "OVERSOLD"
    elif rsi > 70:
        rsi_zone = "OVERBOUGHT"
    elif rsi < 45:
        rsi_zone = "weak"
    elif rsi > 55:
        rsi_zone = "strong"
    else:
        rsi_zone = "neutral"

    if bb_pos < 0.2:
        bb_zone = "NEAR LOWER BAND (oversold)"
    elif bb_pos > 0.8:
        bb_zone = "NEAR UPPER BAND (overbought)"
    else:
        bb_zone = "middle range"

    if vol7d > 0.06:
        vol_level = "HIGH"
    elif vol7d > 0.035:
        vol_level = "MODERATE"
    else:
        vol_level = "LOW"

    has_news = row["news_text"] != "No significant news today"

    prompt = f"""You are a conservative cryptocurrency risk analyst focused on capital preservation. Output ONLY valid JSON (no markdown, no code blocks, no explanation).

Your priority: AVOID LOSSES first, capture gains second. In uncertain conditions, recommend "hold".

Date: {row['date']}
Coin: {row['ticker']}

=== TECHNICAL ANALYSIS ===
Price: ${close} | Daily Return: {returns * 100:.2f}%
7-Day Volatility: {vol7d * 100:.2f}% [{vol_level}]
Volume Change: {volume_change * 100:.1f}%

Moving Averages:
- Price vs MA7: {above_ma7} (${ma7:.4f})
- Price vs MA20: {above_ma20}{f' (${ma20:.4f})' if ma20 > 0 else ''}
- Price vs MA50: {above_ma50}{f' (${ma50:.4f})' if ma50 > 0 else ''}
- MA Crossover: {ma_crossover}

Momentum:
- RSI: {rsi:.1f} [{rsi_zone}]
- MACD Histogram: {macd_hist:.6f} [{macd_trend}]

Bollinger Bands:
- BB Position: {bb_pos:.3f} [{bb_zone}]
- Upper: ${bb_upper:.4f} | Lower: ${bb_lower:.4f}

=== NEWS ===
{row['news_text'] if has_news else 'No news available. Base analysis purely on technical indicators above.'}

=== INSTRUCTIONS ===
Analyze the above data and output this exact JSON structure:
{{"sentiment_score": <float from -1.0 to 1.0>, "market_mood": "<bearish|neutral|bullish>", "trend_strength": <float from 0.0 to 1.0>, "reversal_probability": <float from 0.0 to 1.0>, "risk_level": "<low|medium|high|extreme>", "recommended_action": "<buy|sell|hold>", "confidence": <float from 0.0 to 1.0>, "reasoning": "<brief reasoning>"}}

Rules:
- If volatility is HIGH, set risk_level to "high" or "extreme" and lower confidence
- If RSI is OVERSOLD (<30) AND MACD is improving, slight bullish bias
- If price is below all MAs in a downtrend, maintain bearish bias
- If no news is available, base confidence primarily on technical signals
- Default to "hold" unless signals are clearly aligned"""

    return prompt


# ============================================================
# MODULE 3: OPENROUTER API CLIENT
# ============================================================

def call_llm(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            analysis = json.loads(content)
            return analysis

        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                print(f"    LLM call failed after {max_retries} attempts: {e}")
                return {
                    "sentiment_score": 0,
                    "market_mood": "neutral",
                    "trend_strength": 0,
                    "reversal_probability": 0,
                    "risk_level": "medium",
                    "recommended_action": "hold",
                    "confidence": 0,
                    "reasoning": "LLM call failed",
                }


# ============================================================
# MODULE 4: TRADING LOGIC (7-Component Scoring System)
# ============================================================

def trading_decision(analysis, indicators):
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    confidence = float(analysis.get("confidence", 0) or 0)
    reversal_prob = float(analysis.get("reversal_probability", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()

    rsi = float(indicators.get("rsi", 50))
    macd_hist = float(indicators.get("macd_hist", 0))
    bb_pos = float(indicators.get("bb_position", 0.5))
    volatility = float(indicators.get("volatility_7d", 0))

    # === RISK GATE ===
    if risk_level == "extreme":
        return "sell", "EXTREME RISK flagged by LLM", confidence

    # === 7-COMPONENT SCORING SYSTEM ===
    score = 0
    breakdown = []

    # Component 1: RSI (max +/-25)
    if rsi < 25:
        score += 25; breakdown.append("RSI_deep_oversold:+25")
    elif rsi < 30:
        score += 18; breakdown.append("RSI_oversold:+18")
    elif rsi < 40:
        score += 5; breakdown.append("RSI_weak:+5")
    elif rsi > 70:
        score -= 25; breakdown.append("RSI_overbought:-25")
    elif rsi > 60:
        score -= 8; breakdown.append("RSI_elevated:-8")

    # Component 2: MACD Histogram (max +/-20)
    if macd_hist > 0:
        score += 20; breakdown.append("MACD_pos:+20")
    elif macd_hist > -0.001:
        score += 5; breakdown.append("MACD_near_zero:+5")
    else:
        score -= 10; breakdown.append("MACD_neg:-10")

    # Component 3: Bollinger Band Position (max +/-20)
    if bb_pos < 0.1:
        score += 20; breakdown.append("BB_extreme_low:+20")
    elif bb_pos < 0.2:
        score += 12; breakdown.append("BB_low:+12")
    elif bb_pos > 0.9:
        score -= 20; breakdown.append("BB_extreme_high:-20")
    elif bb_pos > 0.8:
        score -= 12; breakdown.append("BB_high:-12")

    # Component 4: LLM Sentiment (max +/-25)
    sent_score = round(sentiment * 25)
    score += sent_score; breakdown.append(f"Sent:{sent_score:+d}")

    # Component 5: LLM Action (max +/-10)
    if llm_action == "buy":
        score += 10; breakdown.append("LLM_buy:+10")
    elif llm_action == "sell":
        score -= 10; breakdown.append("LLM_sell:-10")

    # Component 6: Volatility Penalty (max -15)
    if volatility > 0.045:
        score -= 15; breakdown.append("Vol_high:-15")
    elif volatility > 0.035:
        score -= 8; breakdown.append("Vol_mod:-8")

    # Component 7: Reversal Probability
    if reversal_prob > 0.6 and sentiment < 0 and rsi < 35:
        score += 10; breakdown.append("Rev_bounce:+10")
    elif reversal_prob > 0.6 and sentiment > 0 and rsi > 60:
        score -= 10; breakdown.append("Rev_top:-10")

    # === DECISION THRESHOLDS (grid-search optimized) ===
    decision = "hold"
    bd = ", ".join(breakdown)
    reason = f"Neutral (score={score})"

    # BUY: score >= 15 and RSI < 50
    if score >= 15 and rsi < 50:
        decision = "buy"
        reason = f"BUY: score={score} [{bd}]"

    # SELL: strong negative score OR mild negative + elevated RSI OR extreme BB
    if score <= -15:
        decision = "sell"
        reason = f"SELL: score={score} [{bd}]"
    elif score <= -5 and rsi > 40:
        decision = "sell"
        reason = f"SELL_MILD: score={score}, RSI={rsi:.1f} [{bd}]"
    elif bb_pos > 0.8:
        decision = "sell"
        reason = f"SELL_BB: BB={bb_pos:.2f} [{bd}]"

    # Emergency
    if rsi > 75:
        decision = "sell"
        reason = f"EMERGENCY: RSI={rsi:.1f} extreme overbought"

    return decision, reason, confidence


# ============================================================
# MODULE 5: BACKTESTING ENGINE
# ============================================================

def run_backtest(trades_df):
    print("\n=== BACKTESTING ===")

    capital = STARTING_CAPITAL
    positions = {}  # ticker -> {"qty": float, "avg_price": float}
    daily_portfolio_values = []
    trade_results = []

    # Get unique dates sorted
    dates = sorted(trades_df["date"].unique())

    for date in dates:
        day_trades = trades_df[trades_df["date"] == date]

        for _, trade in day_trades.iterrows():
            ticker = trade["ticker"]
            price = float(trade["price"])
            decision = trade["decision"]

            if decision == "buy" and capital > 0:
                # Allocate equal portion of remaining capital per coin
                alloc = capital * 0.05  # 5% of capital per buy (grid-search optimized)
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price

                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0}

                old_qty = positions[ticker]["qty"]
                old_cost = old_qty * positions[ticker]["avg_price"]
                new_cost = old_cost + invest
                new_qty = old_qty + qty
                positions[ticker]["qty"] = new_qty
                positions[ticker]["avg_price"] = new_cost / new_qty if new_qty > 0 else 0

                capital -= alloc
                trade_results.append({
                    "date": date, "ticker": ticker, "action": "buy",
                    "price": price, "qty": qty, "fee": fee
                })

            elif decision == "sell" and ticker in positions and positions[ticker]["qty"] > 0:
                qty = positions[ticker]["qty"]
                proceeds = qty * price
                fee = proceeds * TRANSACTION_FEE
                net = proceeds - fee

                pnl = net - (qty * positions[ticker]["avg_price"])
                capital += net
                trade_results.append({
                    "date": date, "ticker": ticker, "action": "sell",
                    "price": price, "qty": qty, "fee": fee, "pnl": pnl
                })
                positions[ticker] = {"qty": 0, "avg_price": 0}

        # Calculate end-of-day portfolio value
        portfolio_value = capital
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
                # Get latest price for this ticker on this date
                ticker_row = day_trades[day_trades["ticker"] == ticker]
                if not ticker_row.empty:
                    portfolio_value += pos["qty"] * float(ticker_row.iloc[0]["price"])

        daily_portfolio_values.append({"date": date, "value": portfolio_value})

    # Calculate metrics
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

        print(f"  Sharpe Ratio:  {sharpe:.4f}")
        print(f"  Total Return:  {total_return:+.2f}%")
        print(f"  Max Drawdown:  {drawdown:.2f}%")
        print(f"  Win Rate:      {win_rate:.1f}%")
        print(f"  Total Trades:  {len(trade_results)} ({len([t for t in trade_results if t['action']=='buy'])} buys, {len(sell_trades)} sells)")
        print(f"  Final Value:   ${pv['value'].iloc[-1]:.2f}")

        return {
            "sharpe": sharpe, "total_return": total_return,
            "max_drawdown": drawdown, "win_rate": win_rate,
            "total_trades": len(trade_results), "final_value": pv["value"].iloc[-1]
        }
    else:
        print("  Not enough data for backtest metrics")
        return None


# ============================================================
# MODULE 6: MAIN PIPELINE
# ============================================================

def process_row(idx, row, total):
    """Process a single row: build prompt, call LLM, make trading decision."""
    ticker = row["ticker"]
    date = row["date"]

    prompt = build_prompt(row)
    analysis = call_llm(prompt)

    indicators = {
        "rsi": row.get("rsi", 50),
        "macd_hist": row.get("macd_hist", 0),
        "bb_position": row.get("bb_position", 0.5),
        "volatility_7d": row.get("volatility_7d", 0),
    }

    decision, reason, confidence = trading_decision(analysis, indicators)

    return {
        "idx": idx,
        "date": date,
        "ticker": ticker,
        "price": float(row["close"]),
        "decision": decision,
        "reason": reason,
        "sentiment": f"{float(analysis.get('sentiment_score', 0)):.3f}",
        "rsi": f"{float(row.get('rsi', 0)):.2f}",
        "confidence": f"{confidence:.3f}",
        "analysis": analysis,  # Save full LLM analysis for tuning
        "indicators": indicators,
    }


def main():
    start_time = time.time()

    # Step 1: Load data
    features = load_data()
    total = len(features)

    print(f"\nProcessing {total} rows through LLM + Trading Logic (20 concurrent workers)...")
    print("=" * 60)

    # Step 2: Process all rows concurrently
    results = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for idx, row in features.iterrows():
            future = executor.submit(process_row, idx, row, total)
            futures[future] = idx

        for future in as_completed(futures):
            try:
                result = future.result()
                results[result["idx"]] = result
                completed += 1
                if completed % 20 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} ({completed/total*100:.0f}%)")
            except Exception as e:
                idx = futures[future]
                print(f"  ERROR processing row {idx}: {e}")
                row = features.iloc[idx]
                results[idx] = {
                    "idx": idx,
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "price": float(row["close"]),
                    "decision": "hold",
                    "reason": f"Processing error: {e}",
                    "sentiment": "0.000",
                    "rsi": f"{float(row.get('rsi', 0)):.2f}",
                    "confidence": "0.000",
                }
                completed += 1

    # Save raw LLM analysis for offline tuning
    analysis_cache = []
    for r in results:
        if r:
            analysis_cache.append({
                "idx": r["idx"],
                "date": r["date"],
                "ticker": r["ticker"],
                "price": r["price"],
                "analysis": r.get("analysis", {}),
                "indicators": r.get("indicators", {}),
            })
            del r["idx"]
            r.pop("analysis", None)
            r.pop("indicators", None)

    with open("llm_cache.json", "w") as f:
        json.dump(analysis_cache, f)
    print(f"\nSaved LLM analysis cache to llm_cache.json")

    # Step 3: Export trades_log.csv
    trades_df = pd.DataFrame(results)
    trades_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExported {len(trades_df)} trades to {OUTPUT_CSV}")

    # Step 4: Print summary
    buy_count = len(trades_df[trades_df["decision"] == "buy"])
    sell_count = len(trades_df[trades_df["decision"] == "sell"])
    hold_count = len(trades_df[trades_df["decision"] == "hold"])
    print(f"\nDecision Distribution:")
    print(f"  Buy:  {buy_count} ({buy_count/total*100:.1f}%)")
    print(f"  Sell: {sell_count} ({sell_count/total*100:.1f}%)")
    print(f"  Hold: {hold_count} ({hold_count/total*100:.1f}%)")

    # Step 5: Run backtest
    metrics = run_backtest(trades_df)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    return metrics


if __name__ == "__main__":
    main()
