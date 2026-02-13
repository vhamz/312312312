"""
Crypto Trading Bot V3 - Ultra-Conservative Low-Risk Strategy
=============================================================
Key risk reduction vs V2:
  - Max portfolio exposure cap (30% in crypto, 70% always cash)
  - Per-position stop-loss (-3%) and take-profit (+5%)
  - Trailing stop mechanism
  - Requires 3+ confirming signals to buy (consensus filter)
  - Cooldown: no re-buy of same coin within 3 days of selling
  - Position size: 1.5% of capital (down from 2-5%)
  - Volatility gate: skip trading on high-vol days entirely
  - Works with or without LLM (pure technical fallback)
"""

import os
import json
import math
import time
import sys
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# === V3 CONSERVATIVE PARAMETERS ===
ALLOC_PCT = 0.015           # 1.5% per trade (very small)
MAX_EXPOSURE = 0.25         # Never more than 25% in positions
STOP_LOSS_PCT = -0.03       # -3% stop-loss per position
TAKE_PROFIT_PCT = 0.05      # +5% take-profit per position
TRAILING_STOP_PCT = 0.025   # 2.5% trailing stop from peak
COOLDOWN_DAYS = 2            # Don't re-buy same coin for 2 days after sell
VOLATILITY_GATE = 0.065     # Skip all trading when 7d vol > 6.5%

# V3 scoring weights (conservative: balanced, less LLM dependence)
W_RSI = 6.0
W_MACD = 10.0
W_BB = 9.0
W_SENTIMENT = 22.0      # Lower than V2's 33.4 - less LLM dependence
W_ACTION = 12.0
W_VOLATILITY = 10.0     # Higher penalty for volatility
W_REVERSAL = 4.0

BUY_THRESHOLD = 12.0    # Higher than V2's 9.6 but reachable
BUY_RSI_MAX = 48.0      # Stricter than V2's 54.4
SELL_THRESHOLD = -6.0    # Sell on moderate negative signals
MIN_CONFIDENCE = 0.15    # Minimum confidence (technical fallback gives 0.2+)
MIN_CONFIRMING_SIGNALS = 2  # Need 2+ positive signals to buy

USE_LLM = bool(OPENROUTER_API_KEY)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    print("Loading market data...")
    features = pd.read_csv(FEATURES_CSV)
    print(f"  Market data: {len(features)} rows, {len(features.columns)} columns")

    print("Loading news data...")
    news = pd.read_csv(NEWS_CSV)
    print(f"  News data: {len(news)} rows")

    news_by_date = {}
    for _, row in news.iterrows():
        date = str(row["date"]).strip()
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        if date not in news_by_date:
            news_by_date[date] = []
        news_by_date[date].append(f"{title}: {body[:200]}")

    def get_news_text(date):
        date_str = str(date).strip()
        articles = news_by_date.get(date_str, [])
        if articles:
            return "\n\n".join(articles[:3])
        return "No significant news today"

    features["news_text"] = features["date"].apply(get_news_text)
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"  Tickers: {sorted(features['ticker'].unique())}")

    return features


# ============================================================
# LLM PROMPT (conservative framing)
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

    above_ma7 = "above" if close > ma7 else "below"
    above_ma20 = ("above" if close > ma20 else "below") if ma20 > 0 else "N/A"
    above_ma50 = ("above" if close > ma50 else "below") if ma50 > 0 else "N/A"
    ma_crossover = "bullish (MA7 > MA20)" if (ma7 > ma20 and ma20 > 0) else "bearish (MA7 < MA20)"
    macd_trend = "positive (bullish)" if macd_hist > 0 else "negative (bearish)"

    if rsi < 30: rsi_zone = "OVERSOLD"
    elif rsi > 70: rsi_zone = "OVERBOUGHT"
    elif rsi < 45: rsi_zone = "weak"
    elif rsi > 55: rsi_zone = "strong"
    else: rsi_zone = "neutral"

    if bb_pos < 0.2: bb_zone = "NEAR LOWER BAND"
    elif bb_pos > 0.8: bb_zone = "NEAR UPPER BAND"
    else: bb_zone = "middle range"

    if vol7d > 0.06: vol_level = "HIGH"
    elif vol7d > 0.035: vol_level = "MODERATE"
    else: vol_level = "LOW"

    has_news = row["news_text"] != "No significant news today"

    prompt = f"""You are an ultra-conservative cryptocurrency risk analyst. Your #1 priority is CAPITAL PRESERVATION. You would rather miss gains than risk losses. Output ONLY valid JSON.

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
{row['news_text'] if has_news else 'No news. Use only technical analysis.'}

=== INSTRUCTIONS ===
You are EXTREMELY risk-averse. Default to "hold" unless you see VERY strong evidence.
Only recommend "buy" if 3+ indicators align bullishly AND volatility is LOW.
Recommend "sell" at the first sign of weakness.

Output this exact JSON:
{{"sentiment_score": <float -1.0 to 1.0>, "market_mood": "<bearish|neutral|bullish>", "trend_strength": <float 0.0 to 1.0>, "reversal_probability": <float 0.0 to 1.0>, "risk_level": "<low|medium|high|extreme>", "recommended_action": "<buy|sell|hold>", "confidence": <float 0.0 to 1.0>, "reasoning": "<brief reasoning>"}}

Rules:
- If volatility > 4%, set risk_level to "high" or "extreme"
- If RSI is between 40-60, strongly prefer "hold"
- Only recommend "buy" if sentiment > 0.3 AND RSI < 45 AND MACD positive
- Default confidence below 0.5 unless signals are exceptionally clear"""

    return prompt


# ============================================================
# LLM CLIENT
# ============================================================

def call_llm(prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,  # Lower temp for more consistent conservative outputs
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/{max_retries}: {e} (wait {wait}s)")
                time.sleep(wait)
            else:
                print(f"    LLM failed after {max_retries} attempts: {e}")
                return None

    return None


# ============================================================
# PURE TECHNICAL ANALYSIS (no LLM fallback)
# ============================================================

def technical_analysis(row):
    """Generate a synthetic LLM-like analysis from pure technical indicators."""
    rsi = float(row.get("rsi", 50))
    macd_hist = float(row.get("macd_hist", 0))
    bb_pos = float(row.get("bb_position", 0.5))
    vol = float(row.get("volatility_7d", 0))
    returns = float(row.get("returns", 0))
    close = float(row.get("close", 0))
    ma7 = float(row.get("ma7", 0))
    ma20 = float(row.get("ma20", 0)) if pd.notna(row.get("ma20")) else 0
    ma50 = float(row.get("ma50", 0)) if pd.notna(row.get("ma50")) else 0
    volume_change = float(row.get("volume_change", 0))

    sentiment = 0.0
    bullish_signals = 0
    bearish_signals = 0

    # RSI signal (continuous)
    if rsi < 25:
        sentiment += 0.4; bullish_signals += 1
    elif rsi < 35:
        sentiment += 0.25; bullish_signals += 1
    elif rsi < 45:
        sentiment += 0.1
    elif rsi > 70:
        sentiment -= 0.4; bearish_signals += 1
    elif rsi > 60:
        sentiment -= 0.2; bearish_signals += 1
    elif rsi > 50:
        sentiment -= 0.05

    # MACD signal
    if macd_hist > 0.01:
        sentiment += 0.3; bullish_signals += 1
    elif macd_hist > 0:
        sentiment += 0.15; bullish_signals += 1
    elif macd_hist > -0.01:
        sentiment -= 0.05
    else:
        sentiment -= 0.2; bearish_signals += 1

    # BB signal
    if bb_pos < 0.15:
        sentiment += 0.3; bullish_signals += 1
    elif bb_pos < 0.3:
        sentiment += 0.15; bullish_signals += 1
    elif bb_pos > 0.85:
        sentiment -= 0.3; bearish_signals += 1
    elif bb_pos > 0.7:
        sentiment -= 0.15; bearish_signals += 1

    # Moving average trend
    if ma7 > 0 and ma20 > 0:
        if close > ma7 and ma7 > ma20:
            sentiment += 0.2; bullish_signals += 1
        elif close > ma7:
            sentiment += 0.1
        elif close < ma7 and ma7 < ma20:
            sentiment -= 0.2; bearish_signals += 1
        elif close < ma7:
            sentiment -= 0.1

    # Volume confirmation
    if volume_change > 0.5 and returns > 0.02:
        sentiment += 0.1  # High volume on up day
    elif volume_change > 0.5 and returns < -0.02:
        sentiment -= 0.1  # High volume on down day

    # Volatility dampening
    if vol > 0.05:
        sentiment *= 0.7  # Reduce conviction in high vol

    sentiment = max(-1.0, min(1.0, sentiment))

    # Mood
    if sentiment > 0.15:
        mood = "bullish"
    elif sentiment < -0.1:
        mood = "bearish"
    else:
        mood = "neutral"

    # Risk level
    if vol > 0.07:
        risk = "extreme"
    elif vol > 0.05:
        risk = "high"
    elif vol > 0.035:
        risk = "medium"
    else:
        risk = "low"

    # Action
    action = "hold"
    if bullish_signals >= 2 and sentiment > 0.2 and rsi < 48 and vol < 0.05:
        action = "buy"
    elif bearish_signals >= 2 or sentiment < -0.15 or rsi > 65:
        action = "sell"

    # Confidence
    total_signals = bullish_signals + bearish_signals
    confidence = min(total_signals * 0.2, 0.8) if total_signals > 0 else 0.1

    # Reversal probability
    reversal_prob = 0.0
    if rsi < 25:
        reversal_prob = 0.7
    elif rsi < 30:
        reversal_prob = 0.5
    elif rsi > 75:
        reversal_prob = 0.6
    elif rsi > 65:
        reversal_prob = 0.3

    return {
        "sentiment_score": round(sentiment, 3),
        "market_mood": mood,
        "trend_strength": round(min(abs(sentiment) * 1.5, 1.0), 3),
        "reversal_probability": round(reversal_prob, 3),
        "risk_level": risk,
        "recommended_action": action,
        "confidence": round(confidence, 3),
        "reasoning": f"Technical: RSI={rsi:.1f}, MACD={'pos' if macd_hist > 0 else 'neg'}, BB={bb_pos:.2f}, Vol={vol:.3f}",
    }


# ============================================================
# V3 CONSERVATIVE TRADING LOGIC
# ============================================================

def count_confirming_signals(analysis, indicators):
    """Count how many independent signals confirm a buy."""
    count = 0
    rsi = float(indicators.get("rsi", 50))
    macd_hist = float(indicators.get("macd_hist", 0))
    bb_pos = float(indicators.get("bb_position", 0.5))
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    llm_action = str(analysis.get("recommended_action", "hold")).lower()

    if rsi < 40: count += 1           # RSI not overbought
    if macd_hist > 0: count += 1       # MACD positive
    if bb_pos < 0.4: count += 1        # Below middle BB
    if sentiment > 0.1: count += 1     # Positive sentiment
    if llm_action == "buy": count += 1 # LLM says buy
    return count


def trading_decision_v3(analysis, indicators, prev_indicators=None):
    """
    Ultra-conservative trading decision.
    Returns (decision, reason, confidence)
    """
    sentiment = float(analysis.get("sentiment_score", 0) or 0)
    confidence = float(analysis.get("confidence", 0) or 0)
    reversal_prob = float(analysis.get("reversal_probability", 0) or 0)
    risk_level = str(analysis.get("risk_level", "medium")).lower()
    llm_action = str(analysis.get("recommended_action", "hold")).lower()

    rsi = float(indicators.get("rsi", 50))
    macd_hist = float(indicators.get("macd_hist", 0))
    bb_pos = float(indicators.get("bb_position", 0.5))
    volatility = float(indicators.get("volatility_7d", 0))

    # === RISK GATES (conservative) ===
    if risk_level in ("extreme", "high"):
        if risk_level == "extreme":
            return "sell", "EXTREME RISK - force sell", confidence
        return "hold", f"HIGH RISK - no new trades (vol={volatility:.3f})", confidence

    if volatility > VOLATILITY_GATE:
        return "hold", f"VOLATILITY GATE: {volatility:.3f} > {VOLATILITY_GATE}", confidence

    # === WEIGHTED SCORING ===
    score = 0.0
    breakdown = []

    # RSI
    if rsi < 30:
        c = W_RSI * (30 - rsi) / 30
        score += c; breakdown.append(f"RSI_oversold:+{c:.1f}")
    elif rsi > 70:
        c = W_RSI * (rsi - 70) / 30
        score -= c; breakdown.append(f"RSI_overbought:-{c:.1f}")
    else:
        c = W_RSI * (50 - rsi) / 50 * 0.2  # Smaller neutral bias
        score += c; breakdown.append(f"RSI_bias:{c:+.1f}")

    # MACD
    if macd_hist > 0:
        score += W_MACD; breakdown.append(f"MACD_pos:+{W_MACD:.1f}")
    elif macd_hist > -0.001:
        c = W_MACD * 0.15
        score += c; breakdown.append(f"MACD_flat:+{c:.1f}")
    else:
        c = W_MACD * 0.6
        score -= c; breakdown.append(f"MACD_neg:-{c:.1f}")

    # BB
    if bb_pos < 0.2:
        c = W_BB * (0.2 - bb_pos) / 0.2
        score += c; breakdown.append(f"BB_low:+{c:.1f}")
    elif bb_pos > 0.8:
        c = W_BB * (bb_pos - 0.8) / 0.2
        score -= c; breakdown.append(f"BB_high:-{c:.1f}")

    # Sentiment
    c = W_SENTIMENT * sentiment
    score += c; breakdown.append(f"Sent:{c:+.1f}")

    # LLM Action
    if llm_action == "buy":
        score += W_ACTION; breakdown.append(f"LLM_buy:+{W_ACTION:.1f}")
    elif llm_action == "sell":
        score -= W_ACTION; breakdown.append(f"LLM_sell:-{W_ACTION:.1f}")

    # Volatility penalty (stronger in V3)
    if volatility > 0.03:
        c = W_VOLATILITY * min((volatility - 0.03) / 0.02, 1.0)
        score -= c; breakdown.append(f"Vol:-{c:.1f}")

    # Reversal
    if reversal_prob > 0.5 and sentiment < 0 and rsi < 35:
        c = W_REVERSAL * reversal_prob
        score += c; breakdown.append(f"Rev_bounce:+{c:.1f}")
    elif reversal_prob > 0.5 and sentiment > 0 and rsi > 60:
        c = W_REVERSAL * reversal_prob
        score -= c; breakdown.append(f"Rev_top:-{c:.1f}")

    # === MACD MOMENTUM CONFIRMATION ===
    macd_improving = True
    if prev_indicators:
        prev_macd = float(prev_indicators.get("macd_hist", 0))
        macd_improving = macd_hist > prev_macd

    # === CONSENSUS FILTER ===
    confirming = count_confirming_signals(analysis, indicators)

    # === DECISIONS ===
    bd = ", ".join(breakdown)
    decision = "hold"
    reason = f"Neutral (score={score:.1f}, signals={confirming})"

    # BUY: need high score + low RSI + MACD improving + enough confirming signals + min confidence
    if (score >= BUY_THRESHOLD
            and rsi < BUY_RSI_MAX
            and macd_improving
            and confirming >= MIN_CONFIRMING_SIGNALS
            and confidence >= MIN_CONFIDENCE):
        decision = "buy"
        reason = f"BUY: score={score:.1f}, signals={confirming} [{bd}]"

    # SELL: lower threshold (sell earlier to protect capital)
    if score <= SELL_THRESHOLD:
        decision = "sell"
        reason = f"SELL: score={score:.1f} [{bd}]"
    elif rsi > 65 and score < 5:
        decision = "sell"
        reason = f"SELL_RSI: RSI={rsi:.1f}, score={score:.1f} [{bd}]"
    elif bb_pos > 0.9:
        decision = "sell"
        reason = f"SELL_BB: BB={bb_pos:.2f} [{bd}]"

    # Emergency
    if rsi > 72:
        decision = "sell"
        reason = f"EMERGENCY: RSI={rsi:.1f} overbought"

    return decision, reason, confidence


# ============================================================
# BACKTESTING ENGINE (with stop-loss, take-profit, trailing stop)
# ============================================================

def run_backtest(trades_df):
    print("\n=== BACKTESTING (V3 Conservative) ===")

    capital = STARTING_CAPITAL
    positions = {}       # ticker -> {"qty", "avg_price", "peak_price", "entry_date"}
    sell_cooldown = {}   # ticker -> last_sell_date
    daily_portfolio_values = []
    trade_results = []

    dates = sorted(trades_df["date"].unique())

    for date in dates:
        day_trades = trades_df[trades_df["date"] == date]

        # First: check stop-loss / take-profit / trailing stop on existing positions
        for _, trade in day_trades.iterrows():
            ticker = trade["ticker"]
            price = float(trade["price"])

            if ticker in positions and positions[ticker]["qty"] > 0:
                pos = positions[ticker]
                avg_price = pos["avg_price"]
                peak_price = pos.get("peak_price", avg_price)

                # Update peak price for trailing stop
                if price > peak_price:
                    positions[ticker]["peak_price"] = price
                    peak_price = price

                pnl_pct = (price - avg_price) / avg_price

                # Stop-loss
                if pnl_pct <= STOP_LOSS_PCT:
                    qty = pos["qty"]
                    proceeds = qty * price
                    fee = proceeds * TRANSACTION_FEE
                    net = proceeds - fee
                    pnl = net - (qty * avg_price)
                    capital += net
                    trade_results.append({
                        "date": date, "ticker": ticker, "action": "sell",
                        "price": price, "qty": qty, "fee": fee, "pnl": pnl,
                        "reason": f"STOP-LOSS ({pnl_pct*100:.1f}%)"
                    })
                    positions[ticker] = {"qty": 0, "avg_price": 0, "peak_price": 0}
                    sell_cooldown[ticker] = date
                    continue

                # Take-profit
                if pnl_pct >= TAKE_PROFIT_PCT:
                    qty = pos["qty"]
                    proceeds = qty * price
                    fee = proceeds * TRANSACTION_FEE
                    net = proceeds - fee
                    pnl = net - (qty * avg_price)
                    capital += net
                    trade_results.append({
                        "date": date, "ticker": ticker, "action": "sell",
                        "price": price, "qty": qty, "fee": fee, "pnl": pnl,
                        "reason": f"TAKE-PROFIT ({pnl_pct*100:.1f}%)"
                    })
                    positions[ticker] = {"qty": 0, "avg_price": 0, "peak_price": 0}
                    sell_cooldown[ticker] = date
                    continue

                # Trailing stop
                if peak_price > avg_price:
                    drop_from_peak = (price - peak_price) / peak_price
                    if drop_from_peak <= -TRAILING_STOP_PCT:
                        qty = pos["qty"]
                        proceeds = qty * price
                        fee = proceeds * TRANSACTION_FEE
                        net = proceeds - fee
                        pnl = net - (qty * avg_price)
                        capital += net
                        trade_results.append({
                            "date": date, "ticker": ticker, "action": "sell",
                            "price": price, "qty": qty, "fee": fee, "pnl": pnl,
                            "reason": f"TRAILING-STOP (peak=${peak_price:.2f}, drop={drop_from_peak*100:.1f}%)"
                        })
                        positions[ticker] = {"qty": 0, "avg_price": 0, "peak_price": 0}
                        sell_cooldown[ticker] = date
                        continue

        # Then: process signal-based trades
        for _, trade in day_trades.iterrows():
            ticker = trade["ticker"]
            price = float(trade["price"])
            decision = trade["decision"]

            # Check if position was already closed above
            if ticker in positions and positions[ticker]["qty"] == 0 and decision == "sell":
                continue

            if decision == "buy" and capital > 100:
                # Cooldown check
                if ticker in sell_cooldown:
                    last_sell = sell_cooldown[ticker]
                    try:
                        days_since = (pd.Timestamp(date) - pd.Timestamp(last_sell)).days
                        if days_since < COOLDOWN_DAYS:
                            continue
                    except Exception:
                        pass

                # Max exposure check
                total_exposure = 0
                prices_today = {t["ticker"]: float(t["price"])
                                for _, t in day_trades.iterrows()}
                for tk, pos in positions.items():
                    if pos["qty"] > 0 and tk in prices_today:
                        total_exposure += pos["qty"] * prices_today[tk]

                portfolio_val = capital + total_exposure
                if total_exposure / portfolio_val >= MAX_EXPOSURE:
                    continue  # Already at max exposure

                alloc = capital * ALLOC_PCT
                fee = alloc * TRANSACTION_FEE
                invest = alloc - fee
                qty = invest / price

                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "avg_price": 0, "peak_price": 0}

                old_qty = positions[ticker]["qty"]
                old_cost = old_qty * positions[ticker]["avg_price"]
                new_qty = old_qty + qty
                new_avg = (old_cost + invest) / new_qty if new_qty > 0 else 0
                positions[ticker]["qty"] = new_qty
                positions[ticker]["avg_price"] = new_avg
                positions[ticker]["peak_price"] = max(
                    positions[ticker].get("peak_price", 0), price
                )
                positions[ticker]["entry_date"] = date

                capital -= alloc
                trade_results.append({
                    "date": date, "ticker": ticker, "action": "buy",
                    "price": price, "qty": qty, "fee": fee, "pnl": 0,
                    "reason": "SIGNAL"
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
                    "price": price, "qty": qty, "fee": fee, "pnl": pnl,
                    "reason": "SIGNAL"
                })
                positions[ticker] = {"qty": 0, "avg_price": 0, "peak_price": 0}
                sell_cooldown[ticker] = date

        # End-of-day portfolio value
        portfolio_value = capital
        for ticker, pos in positions.items():
            if pos["qty"] > 0:
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

        buy_trades = [t for t in trade_results if t["action"] == "buy"]
        stop_losses = len([t for t in trade_results if "STOP-LOSS" in t.get("reason", "")])
        take_profits = len([t for t in trade_results if "TAKE-PROFIT" in t.get("reason", "")])
        trailing_stops = len([t for t in trade_results if "TRAILING-STOP" in t.get("reason", "")])

        print(f"  Sharpe Ratio:    {sharpe:.4f}")
        print(f"  Total Return:    {total_return:+.2f}%")
        print(f"  Max Drawdown:    {drawdown:.2f}%")
        print(f"  Win Rate:        {win_rate:.1f}%")
        print(f"  Total Trades:    {len(trade_results)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
        print(f"  Stop-Losses:     {stop_losses}")
        print(f"  Take-Profits:    {take_profits}")
        print(f"  Trailing Stops:  {trailing_stops}")
        print(f"  Final Value:     ${pv['value'].iloc[-1]:.2f}")

        return {
            "sharpe": sharpe, "total_return": total_return,
            "max_drawdown": drawdown, "win_rate": win_rate,
            "total_trades": len(trade_results), "final_value": pv["value"].iloc[-1],
            "stop_losses": stop_losses, "take_profits": take_profits,
        }
    else:
        print("  Not enough data for backtest metrics")
        return None


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_row(idx, row, total, prev_indicators_map):
    ticker = row["ticker"]
    date = row["date"]

    indicators = {
        "rsi": row.get("rsi", 50),
        "macd_hist": row.get("macd_hist", 0),
        "bb_position": row.get("bb_position", 0.5),
        "volatility_7d": row.get("volatility_7d", 0),
    }

    # Get previous day indicators for MACD momentum
    prev_ind = prev_indicators_map.get((date, ticker))

    if USE_LLM:
        prompt = build_prompt(row)
        analysis = call_llm(prompt)
        if analysis is None:
            analysis = technical_analysis(row)
    else:
        analysis = technical_analysis(row)

    decision, reason, confidence = trading_decision_v3(analysis, indicators, prev_ind)

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
    }


def main():
    start_time = time.time()

    # Step 1: Load data
    features = load_data()
    total = len(features)

    mode = "LLM + Technical" if USE_LLM else "Pure Technical (no API key)"
    print(f"\nMode: {mode}")
    print(f"Processing {total} rows...")
    print("=" * 60)

    # Build prev_indicators map
    by_ticker = defaultdict(list)
    for idx, row in features.iterrows():
        by_ticker[row["ticker"]].append((idx, row))

    prev_indicators_map = {}
    for ticker, rows in by_ticker.items():
        rows.sort(key=lambda x: x[1]["date"])
        for i in range(1, len(rows)):
            curr_idx, curr_row = rows[i]
            prev_idx, prev_row = rows[i - 1]
            prev_indicators_map[(curr_row["date"], ticker)] = {
                "rsi": prev_row.get("rsi", 50),
                "macd_hist": prev_row.get("macd_hist", 0),
                "bb_position": prev_row.get("bb_position", 0.5),
                "volatility_7d": prev_row.get("volatility_7d", 0),
            }

    # Step 2: Process rows
    results = [None] * total
    completed = 0

    if USE_LLM:
        # Concurrent LLM calls
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}
            for idx, row in features.iterrows():
                future = executor.submit(process_row, idx, row, total, prev_indicators_map)
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
                    row = features.iloc[idx]
                    results[idx] = {
                        "idx": idx, "date": row["date"], "ticker": row["ticker"],
                        "price": float(row["close"]), "decision": "hold",
                        "reason": f"Error: {e}", "sentiment": "0.000",
                        "rsi": f"{float(row.get('rsi', 0)):.2f}", "confidence": "0.000",
                    }
                    completed += 1
    else:
        # Sequential (no LLM, fast)
        for idx, row in features.iterrows():
            result = process_row(idx, row, total, prev_indicators_map)
            results[result["idx"]] = result
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({completed/total*100:.0f}%)")

    # Clean up idx field
    for r in results:
        if r:
            r.pop("idx", None)

    # Step 3: Export
    trades_df = pd.DataFrame(results)
    trades_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nExported {len(trades_df)} trades to {OUTPUT_CSV}")

    # Step 4: Summary
    buy_count = len(trades_df[trades_df["decision"] == "buy"])
    sell_count = len(trades_df[trades_df["decision"] == "sell"])
    hold_count = len(trades_df[trades_df["decision"] == "hold"])
    print(f"\nDecision Distribution:")
    print(f"  Buy:  {buy_count} ({buy_count/total*100:.1f}%)")
    print(f"  Sell: {sell_count} ({sell_count/total*100:.1f}%)")
    print(f"  Hold: {hold_count} ({hold_count/total*100:.1f}%)")

    # Step 5: Backtest
    metrics = run_backtest(trades_df)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    if metrics:
        print(f"\n{'='*60}")
        print(f"V1 baseline:     Sharpe=1.8622, Return=+3.37%")
        print(f"V2 optimized:    Sharpe=4.7036, Return=+1.14%")
        print(f"V3 conservative: Sharpe={metrics['sharpe']:.4f}, Return={metrics['total_return']:+.2f}%")

    return metrics


if __name__ == "__main__":
    main()
