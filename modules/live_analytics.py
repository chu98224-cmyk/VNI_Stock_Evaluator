"""
Live Analytics & Decision Engine
Provides Order Flow Analysis (Active Buy vs Sell), VWAP, 15-Factor Technical Signal Consensus,
Dynamic Trade Setup (Entry, Targets, Stop-loss, R:R), Pivot Points, and Daily Market Pulse Wrap.
"""

import pandas as pd
import numpy as np


def analyze_order_flow(intraday_df: pd.DataFrame, ref_price: float = None):
    """
    Analyze intraday tick data for Active Buy vs Active Sell volume, VWAP, and liquidity pressure.
    """
    if intraday_df is None or intraday_df.empty or len(intraday_df) < 2:
        return {
            "has_data": False,
            "buy_vol": 0,
            "sell_vol": 0,
            "neutral_vol": 0,
            "total_vol": 0,
            "buy_pct": 50.0,
            "sell_pct": 50.0,
            "pressure": "Cân bằng",
            "vwap_latest": 0,
            "vwap_diff_pct": 0,
            "df_vwap": pd.DataFrame()
        }

    df = intraday_df.copy()
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)

    match_col = 'match_type' if 'match_type' in df.columns else None
    
    buy_vol = 0
    sell_vol = 0
    neutral_vol = 0
    
    if match_col:
        buy_mask = df[match_col].astype(str).str.upper().str.contains('BUY|B', regex=True)
        sell_mask = df[match_col].astype(str).str.upper().str.contains('SELL|S', regex=True)
        
        buy_vol = float(df.loc[buy_mask, 'volume'].sum())
        sell_vol = float(df.loc[sell_mask, 'volume'].sum())
        neutral_vol = float(df.loc[~buy_mask & ~sell_mask, 'volume'].sum())
    else:
        df['p_diff'] = df['price'].diff().fillna(0)
        buy_vol = float(df.loc[df['p_diff'] > 0, 'volume'].sum())
        sell_vol = float(df.loc[df['p_diff'] < 0, 'volume'].sum())
        neutral_vol = float(df.loc[df['p_diff'] == 0, 'volume'].sum())

    total_matched = buy_vol + sell_vol
    if total_matched > 0:
        buy_pct = (buy_vol / total_matched) * 100
        sell_pct = (sell_vol / total_matched) * 100
    else:
        buy_pct = 50.0
        sell_pct = 50.0

    if buy_pct >= 62.0:
        pressure = "Bên Mua áp đảo (Bullish Aggressive)"
        badge_type = "green"
    elif buy_pct >= 53.0:
        pressure = "Nghiêng về Mua (Moderate Buy)"
        badge_type = "green"
    elif sell_pct >= 62.0:
        pressure = "Bên Bán áp đảo (Bearish Aggressive)"
        badge_type = "red"
    elif sell_pct >= 53.0:
        pressure = "Nghiêng về Bán (Moderate Sell)"
        badge_type = "red"
    else:
        pressure = "Thế trận Giằng co (Balanced)"
        badge_type = "yellow"

    # Compute VWAP
    df['cum_val'] = (df['price'] * df['volume']).cumsum()
    df['cum_vol'] = df['volume'].cumsum()
    df['vwap'] = df['cum_val'] / df['cum_vol'].replace(0, np.nan)
    
    latest_p = float(df['price'].iloc[-1])
    latest_vwap = float(df['vwap'].dropna().iloc[-1]) if not df['vwap'].dropna().empty else latest_p
    vwap_diff = ((latest_p - latest_vwap) / latest_vwap * 100) if latest_vwap > 0 else 0

    return {
        "has_data": True,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "neutral_vol": neutral_vol,
        "total_vol": float(df['volume'].sum()),
        "buy_pct": round(buy_pct, 1),
        "sell_pct": round(sell_pct, 1),
        "pressure": pressure,
        "badge_type": badge_type,
        "latest_price": latest_p,
        "vwap_latest": latest_vwap,
        "vwap_diff_pct": round(vwap_diff, 2),
        "df_vwap": df
    }


def calculate_technical_signals(df: pd.DataFrame):
    """Evaluates quantitative technical indicators across Oscillators & Moving Averages."""
    if df is None or df.empty or len(df) < 20:
        return {"consensus": "THEO DÕI (NEUTRAL)", "badge_cls": "badge-yellow", "score": 50, "buy_count": 0, "neutral_count": 0, "sell_count": 0, "total_count": 0, "signals": []}

    data = df.copy().sort_values('time').reset_index(drop=True)
    c, h, l = data['close'], data['high'], data['low']
    latest_close = float(c.iloc[-1])
    signals = []

    # 1. RSI (14)
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    r_val = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50
    if r_val < 30: signals.append({"name": "RSI (14)", "value": f"{r_val:.1f}", "signal": "BUY", "desc": "Quá bán (Oversold)"})
    elif r_val > 70: signals.append({"name": "RSI (14)", "value": f"{r_val:.1f}", "signal": "SELL", "desc": "Quá mua (Overbought)"})
    elif r_val >= 50: signals.append({"name": "RSI (14)", "value": f"{r_val:.1f}", "signal": "BUY", "desc": "Động lượng tích cực"})
    else: signals.append({"name": "RSI (14)", "value": f"{r_val:.1f}", "signal": "SELL", "desc": "Động lượng yếu"})

    # 2. Stochastic
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    k_s = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    d_s = k_s.rolling(3).mean()
    k_v = float(k_s.dropna().iloc[-1]) if not k_s.dropna().empty else 50
    d_v = float(d_s.dropna().iloc[-1]) if not d_s.dropna().empty else 50
    if k_v > d_v and k_v < 80: signals.append({"name": "Stochastic %K/%D", "value": f"{k_v:.0f}/{d_v:.0f}", "signal": "BUY", "desc": "Giao cắt tăng"})
    elif k_v < d_v and k_v > 20: signals.append({"name": "Stochastic %K/%D", "value": f"{k_v:.0f}/{d_v:.0f}", "signal": "SELL", "desc": "Giao cắt giảm"})
    else: signals.append({"name": "Stochastic %K/%D", "value": f"{k_v:.0f}/{d_v:.0f}", "signal": "NEUTRAL", "desc": "Trung lập"})

    # 3. MACD
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    macd_v, sig_v, hist_v = float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])
    if macd_v > sig_v and hist_v > 0: signals.append({"name": "MACD (12,26,9)", "value": f"{hist_v:+,.0f}", "signal": "BUY", "desc": "MACD trên tín hiệu"})
    elif macd_v < sig_v and hist_v < 0: signals.append({"name": "MACD (12,26,9)", "value": f"{hist_v:+,.0f}", "signal": "SELL", "desc": "MACD dưới tín hiệu"})
    else: signals.append({"name": "MACD (12,26,9)", "value": f"{hist_v:+,.0f}", "signal": "NEUTRAL", "desc": "Trung lập"})

    # 4. Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    u_v = float((bb_mid + 2*bb_std).dropna().iloc[-1]) if not bb_mid.dropna().empty else latest_close
    l_v = float((bb_mid - 2*bb_std).dropna().iloc[-1]) if not bb_mid.dropna().empty else latest_close
    m_v = float(bb_mid.dropna().iloc[-1]) if not bb_mid.dropna().empty else latest_close
    if latest_close <= l_v: signals.append({"name": "Bollinger Bands", "value": f"{latest_close:,.0f}", "signal": "BUY", "desc": "Chạm dải dưới"})
    elif latest_close >= u_v: signals.append({"name": "Bollinger Bands", "value": f"{latest_close:,.0f}", "signal": "SELL", "desc": "Chạm dải trên"})
    elif latest_close > m_v: signals.append({"name": "Bollinger Bands", "value": f"{latest_close:,.0f}", "signal": "BUY", "desc": "Trên trục giữa BB20"})
    else: signals.append({"name": "Bollinger Bands", "value": f"{latest_close:,.0f}", "signal": "SELL", "desc": "Dưới trục giữa BB20"})

    # 5. EMA 9/21
    e9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
    e21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
    signals.append({"name": "EMA 9 / EMA 21", "value": f"{e9:,.0f} vs {e21:,.0f}", "signal": "BUY" if e9 > e21 else "SELL", "desc": "Ngắn hạn trên trung hạn" if e9 > e21 else "Ngắn hạn dưới trung hạn"})

    # 6. MA20 / MA50 / MA200
    m20 = float(c.rolling(20).mean().dropna().iloc[-1]) if len(c) >= 20 else latest_close
    signals.append({"name": "Price vs MA20", "value": f"{m20:,.0f}", "signal": "BUY" if latest_close > m20 else "SELL", "desc": "Trên hỗ trợ MA20" if latest_close > m20 else "Dưới MA20"})
    if len(c) >= 50:
        m50 = float(c.rolling(50).mean().dropna().iloc[-1])
        signals.append({"name": "Price vs MA50", "value": f"{m50:,.0f}", "signal": "BUY" if latest_close > m50 else "SELL", "desc": "Xu hướng trung hạn MA50" if latest_close > m50 else "Dưới MA50"})
    if len(c) >= 200:
        m200 = float(c.rolling(200).mean().dropna().iloc[-1])
        signals.append({"name": "Price vs MA200", "value": f"{m200:,.0f}", "signal": "BUY" if latest_close > m200 else "SELL", "desc": "Dài hạn MA200" if latest_close > m200 else "Dưới MA200"})

    buy_cnt = sum(1 for s in signals if s["signal"] == "BUY")
    sell_cnt = sum(1 for s in signals if s["signal"] == "SELL")
    neu_cnt = sum(1 for s in signals if s["signal"] == "NEUTRAL")
    tot = len(signals)
    score = round(((buy_cnt + 0.5 * neu_cnt) / tot) * 100) if tot > 0 else 50

    if score >= 70: consensus, badge_cls = "MUA MẠNH (STRONG BUY)", "badge-green"
    elif score >= 55: consensus, badge_cls = "MUA (BUY)", "badge-green"
    elif score <= 30: consensus, badge_cls = "BÁN MẠNH (STRONG SELL)", "badge-red"
    elif score <= 45: consensus, badge_cls = "BÁN (SELL)", "badge-red"
    else: consensus, badge_cls = "THEO DÕI (NEUTRAL)", "badge-yellow"

    return {"consensus": consensus, "badge_cls": badge_cls, "score": score, "buy_count": buy_cnt, "neutral_count": neu_cnt, "sell_count": sell_cnt, "total_count": tot, "signals": signals}


def calculate_trade_setup(df: pd.DataFrame):
    """Computes actionable trade setup: Entry, Targets, Stop-Loss, and R:R ratio."""
    if df is None or df.empty or len(df) < 15:
        return {}

    data = df.copy().sort_values('time').reset_index(drop=True)
    c, h, l = data['close'], data['high'], data['low']
    latest_close = float(c.iloc[-1])

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().dropna().iloc[-1]) if not tr.rolling(14).mean().dropna().empty else (latest_close * 0.02)

    entry_low = latest_close - (0.5 * atr)
    entry_high = latest_close + (0.2 * atr)
    stop_loss = max(100, latest_close - (1.5 * atr))
    target_1 = latest_close + (1.5 * atr)
    target_2 = latest_close + (3.0 * atr)

    risk_vnd = latest_close - stop_loss
    rr_t1 = ((target_1 - latest_close) / risk_vnd) if risk_vnd > 0 else 1.0
    rr_t2 = ((target_2 - latest_close) / risk_vnd) if risk_vnd > 0 else 2.0

    return {
        "current_price": latest_close,
        "atr": atr,
        "entry_range": (entry_low, entry_high),
        "stop_loss": stop_loss,
        "stop_loss_pct": ((stop_loss - latest_close) / latest_close * 100),
        "target_1": target_1,
        "target_1_pct": ((target_1 - latest_close) / latest_close * 100),
        "target_2": target_2,
        "target_2_pct": ((target_2 - latest_close) / latest_close * 100),
        "risk_reward_t1": round(rr_t1, 2),
        "risk_reward_t2": round(rr_t2, 2)
    }


def calculate_pivot_points(df: pd.DataFrame):
    """Calculates Classic and Fibonacci Pivot Points."""
    if df is None or df.empty or len(df) < 2:
        return {}

    prev = df.iloc[-2]
    h, l, c = float(prev.get('high', 0)), float(prev.get('low', 0)), float(prev.get('close', 0))
    if h == 0 or l == 0 or c == 0:
        latest = df.iloc[-1]
        h, l, c = float(latest.get('high', 10000)), float(latest.get('low', 10000)), float(latest.get('close', 10000))

    rng = h - l
    pp = (h + l + c) / 3

    return {
        "classic": {
            "R3": h + 2 * (pp - l), "R2": pp + rng, "R1": (2 * pp) - l,
            "PP": pp,
            "S1": (2 * pp) - h, "S2": pp - rng, "S3": l - 2 * (h - pp)
        },
        "fibonacci": {
            "R3": pp + (1.000 * rng), "R2": pp + (0.618 * rng), "R1": pp + (0.382 * rng),
            "PP": pp,
            "S1": pp - (0.382 * rng), "S2": pp - (0.618 * rng), "S3": pp - (1.000 * rng)
        }
    }


def generate_market_summary(indices: list, board_df: pd.DataFrame):
    """Generates an executive natural language market pulse summary."""
    vni = next((item for item in indices if item["symbol"] == "VNINDEX"), None)
    vni_pts = vni["points"] if vni else 0
    vni_chg = vni["change"] if vni else 0
    vni_pct = vni["change_pct"] if vni else 0
    vni_val = vni["value_bil"] if vni else 0

    trend = "tăng điểm bứt phá" if vni_chg > 5 else ("tăng nhẹ" if vni_chg > 0 else ("giảm điểm" if vni_chg < -5 else "điều chỉnh giằng co"))

    advances, declines, ceilings, floors, unchanged = 0, 0, 0, 0, 0
    top_gainers, top_losers, top_liquidity = [], [], []
    net_f = 0

    if board_df is not None and not board_df.empty:
        ceilings = int((board_df['status'] == 'ceiling').sum()) if 'status' in board_df.columns else 0
        floors = int((board_df['status'] == 'floor').sum()) if 'status' in board_df.columns else 0
        advances = int((board_df['change'] > 0).sum()) if 'change' in board_df.columns else 0
        declines = int((board_df['change'] < 0).sum()) if 'change' in board_df.columns else 0
        unchanged = int((board_df['change'] == 0).sum()) if 'change' in board_df.columns else 0

        if 'change_pct' in board_df.columns:
            top_gainers = board_df.sort_values('change_pct', ascending=False).head(4)[['symbol', 'change_pct', 'match_p']].to_dict('records')
            top_losers = board_df.sort_values('change_pct', ascending=True).head(4)[['symbol', 'change_pct', 'match_p']].to_dict('records')
        if 'total_vol' in board_df.columns:
            top_liquidity = board_df.sort_values('total_vol', ascending=False).head(4)[['symbol', 'total_vol', 'match_p']].to_dict('records')
        if 'f_net' in board_df.columns:
            net_f = float(board_df['f_net'].sum())

    f_stance = "mua ròng" if net_f > 0 else "bán ròng"

    text = f"""
- **VN-Index:** Đang ở mốc **{vni_pts:,.2f} điểm** ({vni_chg:+.2f} pts | {vni_pct:+.2f}%), diễn biến **{trend}** với thanh khoản ước tính **{vni_val:,.0f} Tỷ VND**.
- **Độ rộng thị trường:** Toàn sàn ghi nhận **{advances} mã tăng** ({ceilings} trần), **{declines} mã giảm** ({floors} sàn) và **{unchanged} mã tham chiếu**.
- **Khối ngoại & Dòng tiền:** Khối ngoại ghi nhận **{f_stance}** trên các mã trọng điểm. Dòng tiền tập trung sôi động ở các mã dẫn dắt: {', '.join([g['symbol'] for g in top_gainers]) if top_gainers else 'VN30'}.
    """.strip()

    return {
        "text": text,
        "advances": advances, "declines": declines,
        "ceilings": ceilings, "floors": floors, "unchanged": unchanged,
        "top_gainers": top_gainers, "top_losers": top_losers, "top_liquidity": top_liquidity
    }

