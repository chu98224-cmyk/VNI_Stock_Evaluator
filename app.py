"""
Vietnamese Stock Evaluator - Interactive Streamlit App
Features: Technical Analysis (Bollinger Bands, MAs), Intrinsic Valuation (DCF, Graham, Bands, DDM),
Financial Health Scoring (Piotroski F-Score, Altman Z-Score, DuPont Analysis), and Stock Screener.
"""

import datetime
import streamlit as st
import pandas as pd
import numpy as np

from modules.data_fetcher import (
    POPULAR_VN_STOCKS,
    get_all_stock_symbols,
    get_stock_price_history,
    get_company_overview,
    get_company_profile,
    get_ratio_summary,
    get_financial_statements,
    extract_latest_fundamental_metrics
)
from modules.valuation_engine import (
    calculate_pe_pb_bands,
    calculate_graham_valuation,
    calculate_dcf,
    generate_dcf_sensitivity_matrix,
    calculate_ddm,
    synthesize_valuations
)
from modules.scoring_engine import (
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dupont_analysis
)
from modules.charts import (
    create_technical_chart,
    create_valuation_bands_chart,
    create_dupont_chart
)
from modules.tv_chart import render_tradingview_chart


st.set_page_config(
    page_title="VNI Stock Evaluator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161B22; border-radius: 8px; padding: 12px; border: 1px solid #30363D; }
    .metric-card { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 16px; font-weight: bold; font-size: 0.9rem; }
    .badge-green { background-color: rgba(34, 197, 94, 0.2); color: #22C55E; border: 1px solid #22C55E; }
    .badge-yellow { background-color: rgba(234, 179, 8, 0.2); color: #EAB308; border: 1px solid #EAB308; }
    .badge-red { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🇻🇳 VNI Stock Evaluator")

# Industry Preset Selector
industry_choice = st.sidebar.selectbox("📂 Sector / Group", list(POPULAR_VN_STOCKS.keys()), index=0)
group_tickers = POPULAR_VN_STOCKS[industry_choice]

# Ticker Selector with Custom Input
ticker_input = st.sidebar.selectbox("🎯 Select Ticker", group_tickers, index=0)
custom_ticker = st.sidebar.text_input("Or enter custom symbol (e.g. VNM, SSI):", value="").strip().upper()
selected_symbol = custom_ticker if custom_ticker else ticker_input

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Chart Settings")
timeframe_choice = st.sidebar.selectbox("Timeframe", ["6 Months", "1 Year", "2 Years", "3 Years", "5 Years"], index=2)
tf_days_map = {"6 Months": 180, "1 Year": 365, "2 Years": 730, "3 Years": 1095, "5 Years": 1825}
days_to_fetch = tf_days_map.get(timeframe_choice, 1095)

show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", value=True)
show_ma20 = st.sidebar.checkbox("Moving Average 20", value=True)
show_ma50 = st.sidebar.checkbox("Moving Average 50", value=True)
show_ma200 = st.sidebar.checkbox("Moving Average 200", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=False)

# Fetch Data
price_df = get_stock_price_history(selected_symbol, days=days_to_fetch)
overview = get_company_overview(selected_symbol)
ratio_df = get_ratio_summary(selected_symbol)
metrics = extract_latest_fundamental_metrics(selected_symbol)

current_price = 0.0
if price_df is not None and not price_df.empty:
    current_price = float(price_df['close'].iloc[-1])
elif metrics.get("current_price"):
    current_price = float(metrics["current_price"])

metrics["current_price"] = current_price
if current_price > 0:
    if metrics["pe"] and metrics["pe"] > 0:
        metrics["eps"] = current_price / metrics["pe"]
    if metrics["pb"] and metrics["pb"] > 0:
        metrics["bvps"] = current_price / metrics["pb"]

# Sidebar Quick Company Info
st.sidebar.markdown("---")
st.sidebar.subheader("🏢 Company Info")
st.sidebar.write(f"**Name:** {overview.get('organ_name', selected_symbol)}")
st.sidebar.write(f"**Sector:** {overview.get('sector', 'N/A')}")
mkt_cap_vnd = overview.get('market_cap', 0)
if mkt_cap_vnd:
    st.sidebar.write(f"**Market Cap:** {mkt_cap_vnd / 1e9:,.0f} B VND")

# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(f"{selected_symbol} - {overview.get('organ_name', selected_symbol)}")
    st.caption(f"Sector: **{overview.get('sector', 'N/A')}** | Exchange: **{overview.get('com_group_code', 'HOSE')}**")
with col_h2:
    st.metric(
        label="Current Market Price",
        value=f"{current_price:,.0f} VND" if current_price > 0 else "N/A"
    )

# Top Metrics Row
f_score_res = calculate_piotroski_f_score(ratio_df)
z_score_res = calculate_altman_z_score(ratio_df)

col1, col2, col3, col4, col5 = st.columns(5)

tab_tech, tab_val, tab_health, tab_screener = st.tabs([
    "📈 Price & Technicals",
    "💰 Intrinsic Valuation",
    "🏥 Financial Health",
    "🔍 Peer Screener"
])

# =========================================================
# TAB 1: TECHNICAL ANALYSIS & BOLLINGER BANDS
# =========================================================
with tab_tech:
    st.subheader(f"📊 Technical Chart (TradingView / FireAnt Engine) - {selected_symbol}")
    
    chart_view = st.radio(
        "Chart Engine View",
        ["🔥 TradingView Interactive (Drag/Zoom/Pan)", "📊 Multi-Subplot View (RSI & MACD)"],
        horizontal=True
    )
    
    if "TradingView" in chart_view:
        render_tradingview_chart(
            df=price_df,
            symbol=selected_symbol,
            show_bb=show_bb,
            show_ma20=show_ma20,
            show_ma50=show_ma50,
            show_ma200=show_ma200,
            height=580
        )
    else:
        tech_fig = create_technical_chart(
            df=price_df,
            symbol=selected_symbol,
            show_ma20=show_ma20,
            show_ma50=show_ma50,
            show_ma200=show_ma200,
            show_bollinger=show_bb,
            show_rsi=show_rsi,
            show_macd=show_macd
        )
        st.plotly_chart(tech_fig, use_container_width=True)

    
    st.subheader("💡 Technical Signals Summary")
    if price_df is not None and not price_df.empty and len(price_df) >= 20:
        c_last = float(price_df['close'].iloc[-1])
        ma20_val = float(price_df['close'].rolling(20).mean().iloc[-1])
        ma50_val = float(price_df['close'].rolling(50).mean().iloc[-1]) if len(price_df) >= 50 else ma20_val
        ma200_val = float(price_df['close'].rolling(200).mean().iloc[-1]) if len(price_df) >= 200 else ma50_val
        
        bb_std = float(price_df['close'].rolling(20).std().iloc[-1])
        bb_up = ma20_val + 2 * bb_std
        bb_low = ma20_val - 2 * bb_std
        
        sig1, sig2, sig3 = st.columns(3)
        with sig1:
            st.markdown("**Moving Average Trend**")
            if c_last > ma50_val > ma200_val:
                st.success("🟢 Bullish Trend (Price > MA50 > MA200)")
            elif c_last < ma50_val < ma200_val:
                st.error("🔴 Bearish Trend (Price < MA50 < MA200)")
            else:
                st.warning("🟡 Neutral / Consolidation Range")
                
        with sig2:
            st.markdown("**Bollinger Band Position**")
            if c_last >= bb_up:
                st.error("⚠️ Touching Upper Band (+2SD Overbought)")
            elif c_last <= bb_low:
                st.success("🛡️ Touching Lower Band (-2SD Oversold Rebound)")
            else:
                st.info("⚖️ Trading Inside Bollinger Channel")
                

# =========================================================
# TAB 2: INTRINSIC VALUATION MODELS
# =========================================================
with tab_val:
    st.subheader(f"💰 Intrinsic Valuation Suite - {selected_symbol}")
    
    eps_val = metrics.get("eps") or 2000.0
    bvps_val = metrics.get("bvps") or 15000.0
    shares_out = metrics.get("shares_outstanding") or 1000000000
    
    bands_res = calculate_pe_pb_bands(ratio_df, eps_val, bvps_val, current_price)
    pe_fair = bands_res.get("pe_fair_value")
    pb_fair = bands_res.get("pb_fair_value")
    
    graham_res = calculate_graham_valuation(eps_val, bvps_val, growth_rate=8.0, bond_yield=3.2)
    graham_num = graham_res.get("graham_number")
    revised_graham = graham_res.get("revised_graham")
    
    st.markdown("#### ⚙️ DCF Valuation Assumptions")
    dcf_col1, dcf_col2, dcf_col3, dcf_col4 = st.columns(4)
    with dcf_col1:
        fcf_base_input = st.number_input(
            "Base FCF (Billion VND)",
            value=float(round((overview.get('market_cap', 1e13) * 0.08) / 1e9, 1)),
            step=100.0
        )
    with dcf_col2:
        dcf_growth = st.slider("5-Year FCF Growth (%)", min_value=0.0, max_value=25.0, value=10.0, step=0.5)
    with dcf_col3:
        dcf_wacc = st.slider("Discount Rate / WACC (%)", min_value=7.0, max_value=16.0, value=11.0, step=0.5)
    with dcf_col4:
        dcf_term_growth = st.slider("Terminal Growth (%)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        
    dcf_res = calculate_dcf(
        fcf_base=fcf_base_input * 1e9,
        growth_rate_y1_5=dcf_growth,
        terminal_growth_rate=dcf_term_growth,
        discount_rate=dcf_wacc,
        shares_outstanding=shares_out
    )
    dcf_fair = dcf_res["fair_value_per_share"] if dcf_res else None
    
    synth = synthesize_valuations(
        current_price=current_price,
        pe_fair=pe_fair,
        pb_fair=pb_fair,
        graham_number=graham_num,
        revised_graham=revised_graham,
        dcf_fair=dcf_fair
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Valuation Summary & Margin of Safety")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    
    blended_val = synth.get("blended_fair_value") or 0.0
    margin_val = synth.get("margin_of_safety_pct") or 0.0
    
    sum_col1.metric("Current Market Price", f"{current_price:,.0f} VND")
    sum_col2.metric("Blended Fair Value", f"{blended_val:,.0f} VND")
    sum_col3.metric("Margin of Safety", f"{margin_val:+.1f}%", delta=f"{margin_val:+.1f}%")
    
    st.info(f"**Overall Assessment:** {synth.get('status', 'N/A')}")
    
    if synth.get("models"):
        st.markdown("##### 📋 Individual Valuation Models Breakdown")
        df_models = pd.DataFrame(synth["models"])
        df_models['fair_value'] = df_models['fair_value'].apply(lambda x: f"{x:,.0f} VND")
        df_models['weight'] = df_models['weight'].apply(lambda x: f"{x*100:.0f}%")
        st.table(df_models)
        
    st.markdown("---")
    st.markdown("#### 📉 5-Year Historical Valuation Corridor")
    band_col1, band_col2 = st.columns(2)
    with band_col1:
        pe_chart = create_valuation_bands_chart(ratio_df, current_price, metric_type="pe")
        st.plotly_chart(pe_chart, use_container_width=True)
    with band_col2:
        pb_chart = create_valuation_bands_chart(ratio_df, current_price, metric_type="pb")
        st.plotly_chart(pb_chart, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 🧮 DCF Sensitivity Matrix (Growth vs WACC)")
    sens_matrix = generate_dcf_sensitivity_matrix(
        fcf_base=fcf_base_input * 1e9,
        base_growth=dcf_growth,
        base_discount=dcf_wacc,
        terminal_growth=dcf_term_growth,
        shares_outstanding=shares_out
    )
    if not sens_matrix.empty:
        st.dataframe(sens_matrix.style.format("{:,.0f} VND").background_gradient(cmap="Greens"), use_container_width=True)


# =========================================================
# TAB 3: FINANCIAL HEALTH & SCORING
# =========================================================
with tab_health:
    st.subheader(f"🏥 Financial Health & Quality Scoring - {selected_symbol}")
    
    # Piotroski F-Score
    st.markdown("### 💎 Piotroski F-Score Analysis")
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        st.metric("Total F-Score", f"{f_score_res['total_score']} / {f_score_res['max_score']}")
        st.info(f"**Health Status:** {f_score_res['health_status']}")
    with f_col2:
        if f_score_res.get("details"):
            f_table = []
            for name, d in f_score_res["details"].items():
                f_table.append({
                    "Criterion": name,
                    "Category": d["category"],
                    "Value": d["value"],
                    "Status": "✅ Passed" if d["passed"] else "❌ Flagged",
                    "Explanation": d["explanation"]
                })
            st.dataframe(pd.DataFrame(f_table), use_container_width=True, hide_index=True)
            
    st.markdown("---")
    # Altman Z-Score
    st.markdown("### 🛡️ Altman Z''-Score (Bankruptcy & Solvency Risk)")
    z_col1, z_col2 = st.columns([1, 2])
    with z_col1:
        st.metric("Z''-Score", f"{z_score_res['z_score']}")
        st.write(f"**Solvency Zone:** {z_score_res['zone']}")
    with z_col2:
        st.write("• **Safe Zone (Z > 2.6):** Healthy solvency, negligible bankruptcy risk.")
        st.write("• **Grey Zone (1.1 ≤ Z ≤ 2.6):** Moderate financial risk, monitor debt.")
        st.write("• **Distress Zone (Z < 1.1):** High solvency stress risk.")
        
    st.markdown("---")
    # DuPont Analysis
    st.markdown("### 🔬 DuPont 3-Step ROE Decomposition")
    st.caption("ROE = Net Profit Margin (%) × Asset Turnover (x) × Financial Leverage (x)")
    dupont_list = calculate_dupont_analysis(ratio_df)
    if dupont_list:
        dupont_fig = create_dupont_chart(dupont_list)
        st.plotly_chart(dupont_fig, use_container_width=True)
        st.dataframe(pd.DataFrame(dupont_list), use_container_width=True, hide_index=True)
        
    st.markdown("---")
    # Working Capital & Cash Conversion Cycle
    st.markdown("### 🔄 Working Capital & Operating Efficiency")
    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("DSO (Days Sales Out)", f"{metrics['dso']:.0f} days" if metrics['dso'] else "N/A")
    wc2.metric("DIO (Days Inventory)", f"{metrics['dio']:.0f} days" if metrics['dio'] else "N/A")
    wc3.metric("DPO (Days Payable)", f"{metrics['dpo']:.0f} days" if metrics['dpo'] else "N/A")
    cash_cycle_val = ((metrics['dso'] or 0) + (metrics['dio'] or 0) - (metrics['dpo'] or 0)) if metrics['dso'] else None
    wc4.metric("Cash Conversion Cycle", f"{cash_cycle_val:.0f} days" if cash_cycle_val else "N/A")


# =========================================================
# TAB 4: PEER SCREENER
# =========================================================
with tab_screener:
    st.subheader(f"🔍 Sector Peer Screener: {industry_choice}")
    st.caption("Real-time valuation & health comparison across industry peers.")
    
    if st.button("🔄 Scan & Compare Sector Peers", type="primary"):
        with st.spinner(f"Fetching live fundamental metrics for {len(group_tickers)} stocks..."):
            screener_rows = []
            for sym in group_tickers:
                try:
                    m = extract_latest_fundamental_metrics(sym)
                    r_df = get_ratio_summary(sym)
                    f_res = calculate_piotroski_f_score(r_df)
                    z_res = calculate_altman_z_score(r_df)
                    
                    screener_rows.append({
                        "Ticker": sym,
                        "Price (VND)": f"{m['current_price']:,.0f}" if m['current_price'] else "N/A",
                        "P/E": round(m['pe'], 1) if m['pe'] else "N/A",
                        "P/B": round(m['pb'], 2) if m['pb'] else "N/A",
                        "ROE (%)": f"{m['roe']:.1f}%" if m['roe'] else "N/A",
                        "Div Yield": f"{m['dividend_yield']:.1f}%" if m['dividend_yield'] else "0.0%",
                        "F-Score": f"{f_res['total_score']}/9",
                        "Z-Score": z_res['z_score'] if z_res['z_score'] else "N/A",
                        "Solvency": "Safe" if (z_res['z_score'] or 0) >= 2.6 else ("Grey" if (z_res['z_score'] or 0) >= 1.1 else "Distress")
                    })
                except Exception:
                    pass
                    
            if screener_rows:
                df_screen = pd.DataFrame(screener_rows)
                st.dataframe(df_screen, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not load peer comparison data at this time.")
    else:
        st.info("Click the button above to run a live scan across all stocks in this sector.")

