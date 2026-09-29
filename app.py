
import datetime
import streamlit as st
import pandas as pd
import numpy as np

from modules.data_fetcher import (
    POPULAR_VN_STOCKS,
    get_all_stock_symbols,
    get_exchange_symbols,
    get_live_price_board,
    get_intraday_ticks,
    get_intraday_history,
    get_market_indices,
    get_stock_price_history,
    get_company_overview,
    get_company_profile,
    get_ratio_summary,
    get_financial_statements,
    extract_latest_fundamental_metrics
)
from modules.live_analytics import (
    analyze_order_flow,
    calculate_technical_signals,
    calculate_trade_setup,
    calculate_pivot_points,
    generate_market_summary
)
from modules.market_board import (
    render_indices_banner,
    render_price_board_table
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
    create_dupont_chart,
    create_intraday_vwap_chart,
    create_technical_gauge_chart,
    create_order_flow_donut_chart
)
from modules.tv_chart import render_tradingview_chart


st.set_page_config(
    page_title="VNI Stock Evaluator & Live Board",
    page_icon="🇻🇳",
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
# APP NAVIGATION & SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🇻🇳 VNI Evaluator & Live")

if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "HPG"

app_view = st.sidebar.radio(
    "📍 Chế Độ Xem",
    ["📊 Bảng Giá Trực Tuyến (Live VCBS)", "🔬 Phân Tích Cổ Phiếu Chi Tiết"],
    index=0
)

market_indices = get_market_indices()


# ==============================================================================
# VIEW 1: BẢNG GIÁ TRỰC TUYẾN VCBS-STYLE (HOSE / VN30 / HNX / SECTORS)
# ==============================================================================
if app_view == "📊 Bảng Giá Trực Tuyến (Live VCBS)":
    render_indices_banner(market_indices)

    b_col1, b_col2, b_col3 = st.columns([2, 2, 1])
    with b_col1:
        board_group = st.selectbox(
            "📂 Chọn Sàn / Nhóm Cổ Phiếu",
            ["⭐ VN30 Top Picks", "🏛️ Toàn Sàn HOSE (Top 40)", "🏦 Banking (Ngân hàng)", "💻 Tech & Retail", "🏗️ Steel & Materials", "🏘️ Real Estate", "📈 Securities", "🥛 Consumer & Food", "⚡ Energy & Utilities", "🚢 Logistics & Ports", "🌾 Chemical & Fertilizer"],
            index=0
        )
    with b_col2:
        quick_inspect = st.text_input("🔍 Soi nhanh mã CK (e.g. HPG, SSI, VNM):", value="").strip().upper()
        if quick_inspect:
            st.session_state["selected_ticker"] = quick_inspect
            st.info(f"Đã chọn mã **{quick_inspect}**. Chuyển sang tab 'Phân Tích Cổ Phiếu Chi Tiết' để xem định giá & tín hiệu!")

    with b_col3:
        st.write("")
        st.write("")
        if st.button("🔄 Làm mới bảng giá", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if board_group == "⭐ VN30 Top Picks":
        symbols_to_load = POPULAR_VN_STOCKS["⭐ VN30 Top Picks"]
    elif board_group == "🏛️ Toàn Sàn HOSE (Top 40)":
        hose_all = get_exchange_symbols("HSX")
        symbols_to_load = hose_all[:40] if hose_all else POPULAR_VN_STOCKS["⭐ VN30 Top Picks"]
    else:
        for k, v in POPULAR_VN_STOCKS.items():
            if board_group.split()[-1] in k:
                symbols_to_load = v
                break
        else:
            symbols_to_load = POPULAR_VN_STOCKS["⭐ VN30 Top Picks"]

    with st.spinner("Đang tải dữ liệu bảng giá trực tuyến..."):
        board_df = get_live_price_board(symbols_to_load)

    summary_wrap = generate_market_summary(market_indices, board_df)
    with st.expander("📢 BẢN TIN TỔNG HỢP THỊ TRƯỜNG HÔM NAY (MARKET PULSE WRAP)", expanded=True):
        st.markdown(summary_wrap["text"])
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.markdown(f"**Độ Rộng:** 🟢 {summary_wrap['advances']} | 🟡 {summary_wrap['unchanged']} | 🔴 {summary_wrap['declines']}")
        with m_c2:
            gainers_str = ", ".join([f"{g['symbol']} (+{g['change_pct']:.1f}%)" for g in summary_wrap['top_gainers'][:2]])
            st.markdown(f"🔥 **Top Tăng:** {gainers_str if gainers_str else 'N/A'}")
        with m_c3:
            losers_str = ", ".join([f"{l['symbol']} ({l['change_pct']:.1f}%)" for l in summary_wrap['top_losers'][:2]])
            st.markdown(f"❄️ **Top Giảm:** {losers_str if losers_str else 'N/A'}")
        with m_c4:
            vol_str = ", ".join([f"{v['symbol']}" for v in summary_wrap['top_liquidity'][:2]])
            st.markdown(f"💧 **Top KL:** {vol_str if vol_str else 'N/A'}")

    st.markdown("### 🖥️ Bảng Điện Tử Trực Tuyến")
    render_price_board_table(board_df)


# ==============================================================================
# VIEW 2: PHÂN TÍCH CỔ PHIẾU CHI TIẾT (DEEP DIVE EVALUATOR)
# ==============================================================================
else:
    # Sidebar Controls for Stock Analysis
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Chọn Cổ Phiếu Phân Tích")
    
    industry_choice = st.sidebar.selectbox("📂 Nhóm Ngành", list(POPULAR_VN_STOCKS.keys()), index=0)
    group_tickers = POPULAR_VN_STOCKS[industry_choice]
    
    default_idx = group_tickers.index(st.session_state["selected_ticker"]) if st.session_state["selected_ticker"] in group_tickers else 0
    ticker_input = st.sidebar.selectbox("🎯 Mã Cổ Phiếu", group_tickers, index=default_idx)
    custom_ticker = st.sidebar.text_input("Hoặc nhập mã khác (e.g. VNM, SSI):", value="").strip().upper()
    
    selected_symbol = custom_ticker if custom_ticker else ticker_input
    st.session_state["selected_ticker"] = selected_symbol
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Cài Đặt Biểu Đồ")
    tf_options = ["6 Tháng", "1 Năm", "2 Năm", "3 Năm", "4 Năm", "5 Năm", "10 Năm", "Toàn Bộ Lịch Sử (Max)"]
    timeframe_choice = st.sidebar.selectbox("Khung thời gian", tf_options, index=4)
    tf_days_map = {
        "6 Tháng": 180,
        "1 Năm": 365,
        "2 Năm": 730,
        "3 Năm": 1095,
        "4 Năm": 1460,
        "5 Năm": 1825,
        "10 Năm": 3650,
        "Toàn Bộ Lịch Sử (Max)": 7300
    }
    days_to_fetch = tf_days_map.get(timeframe_choice, 1460)
    
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", value=True)
    show_ma20 = st.sidebar.checkbox("Moving Average 20", value=True)
    show_ma50 = st.sidebar.checkbox("Moving Average 50", value=True)
    show_ma200 = st.sidebar.checkbox("Moving Average 200", value=True)
    show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
    show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=False)
    show_pivots = st.sidebar.checkbox("Điểm Xoay Pivot (Kháng Cự & Hỗ Trợ)", value=False)
    pivot_mode = "Classic"
    if show_pivots:
        pivot_mode = st.sidebar.selectbox("Phương pháp Pivot", ["Classic (Chuẩn)", "Fibonacci"], index=0)
    
    # Fetch Data
    price_df = get_stock_price_history(selected_symbol, days=days_to_fetch)
    overview = get_company_overview(selected_symbol)
    ratio_df = get_ratio_summary(selected_symbol)
    metrics = extract_latest_fundamental_metrics(selected_symbol)
    intraday_df = get_intraday_ticks(selected_symbol)
    
    pivots_all = calculate_pivot_points(price_df)
    pivot_key = "classic" if "Classic" in pivot_mode else "fibonacci"
    pivots_active = pivots_all.get(pivot_key, {}) if pivots_all else {}
    
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
    
    # Header Info
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(f"{selected_symbol} - {overview.get('organ_name', selected_symbol)}")
        st.caption(f"Ngành: **{overview.get('sector', 'N/A')}** | Sàn: **{overview.get('com_group_code', 'HOSE')}**")
    with col_h2:
        st.metric(
            label="Giá Thị Trường Hiện Tại",
            value=f"{current_price:,.0f} VND" if current_price > 0 else "N/A"
        )
    
    tab_live, tab_signals, tab_tech, tab_val, tab_health, tab_screener = st.tabs([
        "⚡ Khớp Lệnh & Khối Ngoại (Live)",
        "🎯 Tín Hiệu Mua/Bán & Kế Hoạch GD",
        "📈 Biểu Đồ Kỹ Thuật",
        "💰 Định Giá Doanh Nghiệp",
        "🏥 Sức Khỏe Tài Chính",
        "🔍 Lọc Cổ Phiếu Ngành"
    ])

    # =========================================================
    # TAB 1: LIVE INTRADAY & ORDER FLOW
    # =========================================================
    with tab_live:
        st.subheader(f"⚡ Diễn Biến Khớp Lệnh & Khối Ngoại Trong Phiên - {selected_symbol}")
        of_res = analyze_order_flow(intraday_df, ref_price=current_price)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Mua Chủ Động", f"{of_res['buy_vol']:,.0f} CP" if of_res['has_data'] else "N/A", f"{of_res['buy_pct']}% tổng khớp")
        c2.metric("🔴 Bán Chủ Động", f"{of_res['sell_vol']:,.0f} CP" if of_res['has_data'] else "N/A", f"{of_res['sell_pct']}% tổng khớp")
        c3.metric("⚖️ Đường VWAP", f"{of_res['vwap_latest']:,.0f} VND" if of_res['has_data'] else "N/A", f"{of_res['vwap_diff_pct']:+.2f}% vs giá")
        c4.markdown(f"**Áp lực Khớp lệnh:**<br><span class='status-badge {of_res.get('badge_type', 'badge-yellow')}'>{of_res['pressure']}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Intraday Timeframe Selection for 60fps TradingView Chart
        itv_map = {"⚡ 1 Phút (1m)": "1m", "📊 5 Phút (5m)": "5m", "⏰ 15 Phút (15m)": "15m", "🕒 1 Giờ (1H)": "1H"}
        itv_label = st.radio("Khung Thời Gian Trong Phiên (Intraday)", list(itv_map.keys()), horizontal=True, index=0)
        itv_code = itv_map[itv_label]
        
        intraday_hist_df = get_intraday_history(selected_symbol, interval=itv_code, days=5)
        
        st.markdown(f"#### 📈 Biểu Đồ Kỹ Thuật Intraday (TradingView 60fps Drag/Zoom/Pan) - {selected_symbol} • {itv_label}")
        if intraday_hist_df is not None and not intraday_hist_df.empty and len(intraday_hist_df) >= 5:
            render_tradingview_chart(
                df=intraday_hist_df,
                symbol=selected_symbol,
                show_bb=show_bb,
                show_ma20=show_ma20,
                show_ma50=show_ma50,
                show_ma200=False,
                show_pivots=show_pivots,
                pivots_data=pivots_active,
                height=520,
                timeframe_label=itv_label
            )
        else:
            render_tradingview_chart(
                df=price_df,
                symbol=selected_symbol,
                show_bb=show_bb,
                show_ma20=show_ma20,
                show_ma50=show_ma50,
                show_ma200=show_ma200,
                show_pivots=show_pivots,
                pivots_data=pivots_active,
                height=520,
                timeframe_label="Daily Fallback"
            )

        st.markdown("---")
        chart_col1, chart_col2 = st.columns([1, 1])
        with chart_col1:
            if of_res['has_data']:
                donut_fig = create_order_flow_donut_chart(of_res['buy_vol'], of_res['sell_vol'], of_res['neutral_vol'])
                st.plotly_chart(donut_fig, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu phân bổ lệnh.")
        with chart_col2:
            st.markdown("#### 💡 Ý nghĩa VWAP & Khớp Lệnh Trong Phiên:")
            st.write("• **Giá > VWAP:** Phe Mua chiếm ưu thế, dòng tiền sẵn sàng đẩy giá lên cao hơn mức trung bình.")
            st.write("• **Giá < VWAP:** Phe Bán áp đảo, cảnh giác áp lực hạ giá của bên bán.")
            st.write("• **Tỷ lệ Mua chủ động > 60%:** Tín hiệu gom hàng rõ nét của dòng tiền lớn (Smart Money).")
            st.write("• **Kéo/Thả & Phóng to:** Bạn có thể tự do dùng chuột kéo thả, phóng to thu nhỏ từng phút/giờ giao dịch mượt mà 60fps.")

    # =========================================================
    # TAB 2: ACTIONABLE BUY/SELL SIGNALS & TRADE SETUP
    # =========================================================
    with tab_signals:
        st.subheader(f"🎯 Tín Hiệu Định Lượng & Kế Hoạch Giao Dịch - {selected_symbol}")
        sig_data = calculate_technical_signals(price_df)
        trade_setup = calculate_trade_setup(price_df)
        pivots = calculate_pivot_points(price_df)
        
        sg_col1, sg_col2 = st.columns([1, 1])
        with sg_col1:
            gauge_fig = create_technical_gauge_chart(sig_data['score'], sig_data['consensus'], sig_data['buy_count'], sig_data['neutral_count'], sig_data['sell_count'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        with sg_col2:
            st.markdown("### 📋 Kế Hoạch Giao Dịch Đề Xuất (Trade Setup)")
            if trade_setup:
                st.write(f"• **Vùng Mua Khuyến Nghị:** `{trade_setup['entry_range'][0]:,.0f} - {trade_setup['entry_range'][1]:,.0f} VND`")
                st.write(f"• **Mục Tiêu 1 (TP1):** `{trade_setup['target_1']:,.0f} VND` ({trade_setup['target_1_pct']:+.1f}%) | R:R = **1 : {trade_setup['risk_reward_t1']}**")
                st.write(f"• **Mục Tiêu 2 (TP2):** `{trade_setup['target_2']:,.0f} VND` ({trade_setup['target_2_pct']:+.1f}%) | R:R = **1 : {trade_setup['risk_reward_t2']}**")
                st.write(f"• **Cắt Lỗ (Stop-Loss):** `{trade_setup['stop_loss']:,.0f} VND` ({trade_setup['stop_loss_pct']:.1f}%) [1.5x ATR Volatility]")
            else:
                st.info("Chưa đủ dữ liệu để tính toán Kế hoạch giao dịch.")
                
        st.markdown("---")
        st.markdown("### 📊 Chi Tiết Các Tín Hiệu Kỹ Thuật Định Lượng")
        if sig_data['signals']:
            df_sig = pd.DataFrame(sig_data['signals'])
            st.dataframe(df_sig, use_container_width=True, hide_index=True)
            
        if pivots:
            st.markdown("---")
            st.markdown("### 📍 Điểm Xoay Hỗ Trợ & Kháng Cự (Pivot Points Matrix)")
            pv_col1, pv_col2 = st.columns(2)
            with pv_col1:
                st.markdown("**Classic Pivot Points:**")
                st.write(f"• **Kháng Cự R3 / R2 / R1:** `{pivots['classic']['R3']:,.0f}` | `{pivots['classic']['R2']:,.0f}` | `{pivots['classic']['R1']:,.0f}`")
                st.write(f"• **Điểm Xoay Trục (Pivot PP):** `{pivots['classic']['PP']:,.0f}`")
                st.write(f"• **Hỗ Trợ S1 / S2 / S3:** `{pivots['classic']['S1']:,.0f}` | `{pivots['classic']['S2']:,.0f}` | `{pivots['classic']['S3']:,.0f}`")
            with pv_col2:
                st.markdown("**Fibonacci Pivot Points:**")
                st.write(f"• **Fib R3 / R2 / R1:** `{pivots['fibonacci']['R3']:,.0f}` | `{pivots['fibonacci']['R2']:,.0f}` | `{pivots['fibonacci']['R1']:,.0f}`")
                st.write(f"• **Fib Trục (PP):** `{pivots['fibonacci']['PP']:,.0f}`")
                st.write(f"• **Fib S1 / S2 / S3:** `{pivots['fibonacci']['S1']:,.0f}` | `{pivots['fibonacci']['S2']:,.0f}` | `{pivots['fibonacci']['S3']:,.0f}`")

    # =========================================================
    # TAB 3: TECHNICAL ANALYSIS & BOLLINGER BANDS
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
                show_pivots=show_pivots,
                pivots_data=pivots_active,
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
                show_macd=show_macd,
                show_pivots=show_pivots,
                pivots_data=pivots_active
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
    # TAB 4: INTRINSIC VALUATION MODELS
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
    # TAB 5: FINANCIAL HEALTH & SCORING
    # =========================================================
    with tab_health:
        st.subheader(f"🏥 Financial Health & Quality Scoring - {selected_symbol}")
        f_score_res = calculate_piotroski_f_score(ratio_df)
        z_score_res = calculate_altman_z_score(ratio_df)
        
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
    # TAB 6: PEER SCREENER
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


