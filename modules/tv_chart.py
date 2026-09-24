"""
TradingView Lightweight Charts Component for Streamlit
Provides 60fps pan/zoom/drag candlestick chart with dynamic hover legend & separate volume scale.
"""

import json
import pandas as pd
import streamlit.components.v1 as components


def get_tv_chart_payload(df: pd.DataFrame, show_bb: bool, show_m20: bool, show_m50: bool, show_m200: bool):
    data = df.copy().sort_values('time').reset_index(drop=True)
    data['time_str'] = pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d')
    data['MA20'] = data['close'].rolling(20).mean()
    data['MA50'] = data['close'].rolling(50).mean()
    data['MA200'] = data['close'].rolling(200).mean()
    data['BB_Mid'] = data['close'].rolling(20).mean()
    data['BB_Std'] = data['close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['BB_Std'])

    c_list, v_list, m20_l, m50_l, m200_l = [], [], [], [], []
    bbu_l, bbm_l, bbl_l = [], [], []

    for _, row in data.iterrows():
        t = row['time_str']
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        v = float(row.get('volume', 0))
        c_list.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        v_list.append({"time": t, "value": v, "color": "rgba(8,153,129,0.55)" if c >= o else "rgba(242,54,69,0.55)"})
        if pd.notna(row['MA20']): m20_l.append({"time": t, "value": float(row['MA20'])})
        if pd.notna(row['MA50']): m50_l.append({"time": t, "value": float(row['MA50'])})
        if pd.notna(row['MA200']): m200_l.append({"time": t, "value": float(row['MA200'])})
        if pd.notna(row['BB_Upper']):
            bbu_l.append({"time": t, "value": float(row['BB_Upper'])})
            bbm_l.append({"time": t, "value": float(row['BB_Mid'])})
            bbl_l.append({"time": t, "value": float(row['BB_Lower'])})

    return {
        "cd": c_list, "vd": v_list, "m20": m20_l, "m50": m50_l, "m200": m200_l,
        "bbu": bbu_l, "bbm": bbm_l, "bbl": bbl_l,
        "s_bb": show_bb, "s_20": show_m20, "s_50": show_m50, "s_200": show_m200
    }


from modules.tv_html import TV_HTML_TEMPLATE


def render_tradingview_chart(df: pd.DataFrame, symbol: str, show_bb: bool = True, show_ma20: bool = True, show_ma50: bool = True, show_ma200: bool = True, height: int = 580):
    if df is None or df.empty or len(df) < 5:
        return components.html("<p style='color:#888;'>Insufficient price data.</p>", height=60)
        
    p = get_tv_chart_payload(df, show_bb, show_ma20, show_ma50, show_ma200)
    
    html = (
        TV_HTML_TEMPLATE
        .replace("__HEIGHT__", str(height))
        .replace("__SYMBOL__", str(symbol))
        .replace("__CANDLES_JSON__", json.dumps(p["cd"]))
        .replace("__VOLUME_JSON__", json.dumps(p["vd"]))
        .replace("__M20_JSON__", json.dumps(p["m20"]))
        .replace("__M50_JSON__", json.dumps(p["m50"]))
        .replace("__M200_JSON__", json.dumps(p["m200"]))
        .replace("__BBU_JSON__", json.dumps(p["bbu"]))
        .replace("__BBM_JSON__", json.dumps(p["bbm"]))
        .replace("__BBL_JSON__", json.dumps(p["bbl"]))
        .replace("__SHOW_BB__", str(p["s_bb"]).lower())
        .replace("__SHOW_M20__", str(p["s_20"]).lower())
        .replace("__SHOW_M50__", str(p["s_50"]).lower())
        .replace("__SHOW_M200__", str(p["s_200"]).lower())
    )
    components.html(html, height=height + 15)

