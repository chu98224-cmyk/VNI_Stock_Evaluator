"""
Market Board Module (VCBS-Style Live Price Board Component)
Renders real-time exchange board with 3-level order depth, indices overview, and fast stock selector.
"""

import datetime
import streamlit as st
import pandas as pd
from modules.data_fetcher import get_symbol_live_depth_and_foreign
from modules.charts import create_foreign_flow_chart


def render_indices_banner(indices: list):
    """
    Renders top 4 real-time index cards (VN INDEX, VN 30, HN INDEX, UPCOM INDEX) like VCBS header.
    """
    if not indices:
        return

    cols = st.columns(len(indices))
    for i, idx in enumerate(indices):
        with cols[i]:
            sym = idx.get("name", idx.get("symbol", ""))
            pts = idx.get("points", 0)
            chg = idx.get("change", 0)
            chg_pct = idx.get("change_pct", 0)
            vol = idx.get("volume", 0)
            val = idx.get("value_bil", 0)

            color = "#00C087" if chg > 0 else ("#FF3B30" if chg < 0 else "#EAB308")
            arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "■")
            vol_str = f"{vol/1e6:,.1f}Tr" if vol >= 1e6 else f"{vol:,.0f}"
            val_str = f"{val/1000:,.1f}K Tỷ" if val >= 1000 else f"{val:,.0f} Tỷ"

            st.markdown(f"""
            <div style="background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #E6EDF3;">{sym}</span>
                    <span style="font-weight: 700; font-size: 1.05rem; color: {color};">{pts:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; font-size: 0.85rem;">
                    <span style="color: {color}; font-weight: 600;">{arrow} {chg:+.2f} ({chg_pct:+.2f}%)</span>
                    <span style="color: #8B949E;">{vol_str} | {val_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def format_price(val, status="ref"):
    """Format price with VCBS colors."""
    if val is None or pd.isna(val) or val == 0:
        return "-"
    val_k = val / 1000 if val > 1000 else val
    return f"{val_k:,.2f}" if (val_k % 1 != 0) else f"{val_k:,.1f}"


def format_vol(val):
    """Format volume in hundreds/thousands."""
    if val is None or pd.isna(val) or val == 0:
        return "-"
    if val >= 1e6:
        return f"{val/1e6:.2f}M"
    elif val >= 1e3:
        return f"{val/1e3:.1f}k"
    return f"{val:,.0f}"


def render_price_board_table(board_df: pd.DataFrame, on_select_symbol_callback=None):
    """
    Renders interactive VCBS-style live price board with full 3-level bids, asks, and matched colors.
    """
    if board_df is None or board_df.empty:
        st.info("Chưa có dữ liệu bảng giá cho nhóm cổ phiếu này.")
        return

    display_rows = []
    for _, r in board_df.iterrows():
        sym = r['symbol']
        display_rows.append({
            "Mã CK": sym,
            "TC": r['ref'] / 1000 if r['ref'] > 1000 else r['ref'],
            "Trần": r['ceil'] / 1000 if r['ceil'] > 1000 else r['ceil'],
            "Sàn": r['floor'] / 1000 if r['floor'] > 1000 else r['floor'],
            "G3 (Mua)": r['b3_p'] / 1000 if r['b3_p'] > 1000 else r['b3_p'],
            "KL3 (Mua)": r['b3_v'],
            "G2 (Mua)": r['b2_p'] / 1000 if r['b2_p'] > 1000 else r['b2_p'],
            "KL2 (Mua)": r['b2_v'],
            "G1 (Mua)": r['b1_p'] / 1000 if r['b1_p'] > 1000 else r['b1_p'],
            "KL1 (Mua)": r['b1_v'],
            "Khớp Giá": r['match_p'] / 1000 if r['match_p'] > 1000 else r['match_p'],
            "+/-": r['change'] / 1000 if abs(r['change']) > 1000 else r['change'],
            "%": r['change_pct'],
            "KL Khớp": r['match_v'],
            "G1 (Bán)": r['a1_p'] / 1000 if r['a1_p'] > 1000 else r['a1_p'],
            "KL1 (Bán)": r['a1_v'],
            "G2 (Bán)": r['a2_p'] / 1000 if r['a2_p'] > 1000 else r['a2_p'],
            "KL2 (Bán)": r['a2_v'],
            "G3 (Bán)": r['a3_p'] / 1000 if r['a3_p'] > 1000 else r['a3_p'],
            "KL3 (Bán)": r['a3_v'],
            "Tổng KL": r['total_vol'],
            "Cao": r['high'] / 1000 if r['high'] > 1000 else r['high'],
            "Thấp": r['low'] / 1000 if r['low'] > 1000 else r['low'],
            "TB": r['avg'] / 1000 if r['avg'] > 1000 else r['avg'],
            "NN Mua": r['f_buy'],
            "NN Bán": r['f_sell']
        })

    df_view = pd.DataFrame(display_rows)

    st.dataframe(
        df_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mã CK": st.column_config.TextColumn("Mã CK", width="small"),
            "TC": st.column_config.NumberColumn("TC", format="%.2f"),
            "Trần": st.column_config.NumberColumn("Trần", format="%.2f"),
            "Sàn": st.column_config.NumberColumn("Sàn", format="%.2f"),
            "Khớp Giá": st.column_config.NumberColumn("Khớp", format="%.2f"),
            "+/-": st.column_config.NumberColumn("+/-", format="%+.2f"),
            "%": st.column_config.NumberColumn("%", format="%+.2f%%"),
            "KL Khớp": st.column_config.NumberColumn("KL Khớp", format="%d"),
            "Tổng KL": st.column_config.NumberColumn("Tổng KL", format="%d"),
            "G1 (Mua)": st.column_config.NumberColumn("G1 Mua", format="%.2f"),
            "KL1 (Mua)": st.column_config.NumberColumn("KL1 Mua", format="%d"),
            "G1 (Bán)": st.column_config.NumberColumn("G1 Bán", format="%.2f"),
            "KL1 (Bán)": st.column_config.NumberColumn("KL1 Bán", format="%d"),
            "NN Mua": st.column_config.NumberColumn("NN Mua", format="%d"),
            "NN Bán": st.column_config.NumberColumn("NN Bán", format="%d")
        },
        height=540
    )



def render_fireant_live_tab(selected_symbol: str, current_price: float, intraday_df: pd.DataFrame):
    """
    Renders FireAnt-style live stock modal:
    Tab 1: 3-level Order Book Depth & Live Trade Log + Buy/Sell Volume Summary.
    Tab 2: Foreign Trading breakdown (Volume, Value, Net) & Foreign Room stats.
    """
    st.subheader(f"⚡ Sổ Lệnh & Giao Dịch Khối Ngoại - {selected_symbol}")
    
    subtab_orderbook, subtab_foreign = st.tabs([
        "📋 Sổ Lệnh & Nhật Ký Khớp Lệnh",
        "🌍 Giao Dịch NĐTNN & Room Ngoại"
    ])
    
    depth_info = get_symbol_live_depth_and_foreign(selected_symbol)
    
    with subtab_orderbook:
        ctrl_col1, ctrl_col2 = st.columns([3, 1])
        with ctrl_col1:
            st.caption(f"🕒 Dữ liệu cập nhật lúc: **{datetime.datetime.now().strftime('%H:%M:%S')}** (Bộ nhớ đệm 15 giây)")
        with ctrl_col2:
            if st.button("🔄 Làm mới sổ lệnh", key="btn_refresh_orderbook", use_container_width=True):
                st.rerun()

        # 1. 3-Level Order Depth Table (FireAnt Style)
        st.markdown("##### 📊 Sổ Lệnh Khớp Giá (3 Bước Giá Mua / Bán)")
        if depth_info and depth_info.get("bids") and depth_info.get("asks"):
            bids = depth_info["bids"]
            asks = depth_info["asks"]
            ref_p = depth_info.get("ref", 0)
            
            depth_rows = ""
            for i in range(3):
                b_p = bids[i]["price"]
                b_v = bids[i]["vol"]
                a_p = asks[i]["price"]
                a_v = asks[i]["vol"]
                
                b_p_str = f"{b_p/1000:,.2f}" if b_p >= 1000 else f"{b_p:,.2f}"
                a_p_str = f"{a_p/1000:,.2f}" if a_p >= 1000 else f"{a_p:,.2f}"
                b_v_str = f"{int(b_v):,}" if b_v > 0 else "-"
                a_v_str = f"{int(a_v):,}" if a_v > 0 else "-"
                
                b_col = "#00C087" if b_p > ref_p else ("#FF3B30" if b_p < ref_p else "#EAB308")
                a_col = "#00C087" if a_p > ref_p else ("#FF3B30" if a_p < ref_p else "#EAB308")
                
                depth_rows += f'<tr style="border-bottom: 1px solid #21262D; height: 34px;"><td style="color: #E6EDF3; font-weight: 600; padding: 6px;">{b_v_str}</td><td style="color: {b_col}; font-weight: 700; padding: 6px; border-right: 1px solid #30363D; background: rgba(0,192,135,0.05);">{b_p_str}</td><td style="color: {a_col}; font-weight: 700; padding: 6px; background: rgba(255,59,48,0.05);">{a_p_str}</td><td style="color: #E6EDF3; font-weight: 600; padding: 6px;">{a_v_str}</td></tr>'
            
            depth_html = f'<div style="background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 10px; margin-bottom: 16px;"><table style="width: 100%; border-collapse: collapse; text-align: center; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, monospace; font-size: 14px;"><thead><tr style="border-bottom: 1px solid #30363D; color: #8B949E; font-size: 13px;"><th colspan="2" style="padding: 6px; border-right: 1px solid #30363D; color: #00C087; font-weight: 700;">ĐẶT MUA</th><th colspan="2" style="padding: 6px; color: #FF3B30; font-weight: 700;">ĐẶT BÁN</th></tr><tr style="border-bottom: 1px solid #21262D; color: #8B949E; font-size: 12px;"><th style="padding: 6px; width: 25%;">KL Mua</th><th style="padding: 6px; width: 25%; border-right: 1px solid #30363D;">Giá Mua</th><th style="padding: 6px; width: 25%;">Giá Bán</th><th style="padding: 6px; width: 25%;">KL Bán</th></tr></thead><tbody>{depth_rows}</tbody></table></div>'
            st.markdown(depth_html, unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu sổ lệnh 3 bước giá.")
        # 2. Live Trade Log (Nhật ký khớp lệnh)
        st.markdown("##### 📜 Nhật Ký Khớp Lệnh Trực Tuyến")
        if intraday_df is not None and not intraday_df.empty:
            df_trades = intraday_df.copy()
            ref_val = depth_info.get("ref", current_price)
            if 'time' in df_trades.columns:
                df_trades['Khớp'] = pd.to_datetime(df_trades['time']).dt.strftime('%H:%M:%S')
            else:
                df_trades['Khớp'] = '-'
            
            p_sample = df_trades['price'].iloc[0] if not df_trades.empty else 0
            if p_sample >= 1000:
                df_trades['Giá_num'] = df_trades['price'] / 1000
                ref_scaled = ref_val / 1000 if ref_val >= 1000 else ref_val
            else:
                df_trades['Giá_num'] = df_trades['price']
                ref_scaled = ref_val / 1000 if ref_val >= 1000 else ref_val

            df_trades['Giá'] = df_trades['Giá_num'].apply(lambda x: f"{x:,.2f}")
            df_trades['diff'] = df_trades['Giá_num'] - ref_scaled
            df_trades['+/-'] = df_trades['diff'].apply(lambda x: f"{x:+.2f}")
            df_trades['KL'] = df_trades['volume'].apply(lambda x: f"{int(x):,}")
            
            match_col = 'match_type' if 'match_type' in df_trades.columns else None
            if match_col:
                df_trades['M/B'] = df_trades[match_col].apply(
                    lambda x: '🟢 M' if 'buy' in str(x).lower() or 'b' == str(x).lower() else ('🔴 B' if 'sell' in str(x).lower() or 's' == str(x).lower() else '⚪ -')
                )
            else:
                df_trades['M/B'] = '⚪ -'
            
            display_df = df_trades[['Khớp', 'Giá', '+/-', 'KL', 'M/B']].iloc[::-1].reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, height=360, hide_index=True)
            
            # 3. FireAnt-style Official Summary (KL Mua / Bán Chủ Động & Tổng Khớp)
            st.markdown("##### 📊 Thống Kê Khớp Lệnh Chủ Động & Toàn Phiên (Dữ Liệu Sàn)")
            tot_vol = depth_info.get('total_volume', 0)
            tot_val_bil = depth_info.get('total_value_bil', 0)
            buy_v = depth_info.get('buy_vol', 0)
            sell_v = depth_info.get('sell_vol', 0)
            buy_p = depth_info.get('buy_pct', 0)
            sell_p = depth_info.get('sell_pct', 0)
            buy_val = depth_info.get('buy_val_bil', 0)
            sell_val = depth_info.get('sell_val_bil', 0)

            sum_c1, sum_c2, sum_c3 = st.columns(3)
            sum_c1.metric(
                "📦 Tổng KL Khớp",
                f"{tot_vol:,.0f} CP",
                f"GT: {tot_val_bil:,.2f} Tỷ VND" if tot_val_bil > 0 else None
            )
            sum_c2.metric(
                "🟢 KL MUA Chủ Động",
                f"{buy_v:,.0f} CP" if buy_v > 0 else "-",
                f"{buy_p:.1f}% tổng khớp | {buy_val:,.2f} Tỷ" if buy_v > 0 else None
            )
            sum_c3.metric(
                "🔴 KL BÁN Chủ Động",
                f"{sell_v:,.0f} CP" if sell_v > 0 else "-",
                f"{sell_p:.1f}% tổng khớp | {sell_val:,.2f} Tỷ" if sell_v > 0 else None
            )

            st.markdown("---")
            # 4. Detailed Official Trading Metrics (Giá TB, Biên Độ, ATO/ATC)
            avg_p = depth_info.get('avg_price', 0)
            high_p = depth_info.get('highest', 0)
            low_p = depth_info.get('lowest', 0)
            ref_p = depth_info.get('ref', 0)
            ato_v = depth_info.get('match_vol_ato', 0)
            atc_v = depth_info.get('match_vol_atc', 0)

            c1, c2, c3 = st.columns(3)
            avg_display = f"{avg_p/1000:,.2f}" if avg_p >= 1000 else (f"{avg_p:,.2f}" if avg_p > 0 else "-")
            ref_display = f"{ref_p/1000:,.2f}" if ref_p >= 1000 else (f"{ref_p:,.2f}" if ref_p > 0 else "-")
            diff_avg = (avg_p - ref_p) / 1000 if (avg_p >= 1000 and ref_p >= 1000) else (avg_p - ref_p)
            c1.metric(
                "📊 Giá Khớp TB",
                f"{avg_display}",
                f"{diff_avg:+.2f} vs TC ({ref_display})" if ref_p > 0 and avg_p > 0 else None
            )
            high_display = f"{high_p/1000:,.2f}" if high_p >= 1000 else f"{high_p:,.2f}"
            low_display = f"{low_p/1000:,.2f}" if low_p >= 1000 else f"{low_p:,.2f}"
            spread = (high_p - low_p) / 1000 if high_p >= 1000 else (high_p - low_p)
            c2.metric(
                "🎯 Biên Độ (Thấp - Cao)",
                f"{low_display} - {high_display}" if (high_p > 0 or low_p > 0) else "-",
                f"Biên độ: {spread:,.2f}" if (high_p > 0 and low_p > 0) else None
            )
            atc_display = f"{atc_v:,.0f}" if atc_v > 0 else "-"
            ato_display = f"{ato_v:,.0f}" if ato_v > 0 else "-"
            c3.metric(
                "⏱️ Khớp Lệnh ATO / ATC",
                f"ATC: {atc_display} CP",
                f"ATO: {ato_display} CP" if ato_v > 0 else None
            )
        else:
            st.info("Chưa có dữ liệu nhật ký khớp lệnh phiên hôm nay.")

    with subtab_foreign:
        st.markdown("##### 🌍 Giao Dịch Nhà Đầu Tư Nước Ngoài (NĐTNN)")
        
        f_buy_v = depth_info.get("foreign_buy_vol", 0)
        f_sell_v = depth_info.get("foreign_sell_vol", 0)
        f_net_v = depth_info.get("foreign_net_vol", 0)
        
        f_buy_val = depth_info.get("foreign_buy_val_bil", 0)
        f_sell_val = depth_info.get("foreign_sell_val_bil", 0)
        f_net_val = depth_info.get("foreign_net_val_bil", 0)
        
        fv1, fv2, fv3 = st.columns(3)
        fv1.metric("🟢 KL Mua Ngoại", f"{f_buy_v:,.0f} CP")
        fv2.metric("🔴 KL Bán Ngoại", f"{f_sell_v:,.0f} CP")
        fv3.metric("⚖️ KL Mua - Bán Ròng", f"{f_net_v:+,.0f} CP", f"{'Mua ròng' if f_net_v >= 0 else 'Bán ròng'}")
        
        gv1, gv2, gv3 = st.columns(3)
        gv1.metric("🟢 GT Mua Ngoại", f"{f_buy_val:,.2f} Tỷ VND")
        gv2.metric("🔴 GT Bán Ngoại", f"{f_sell_val:,.2f} Tỷ VND")
        gv3.metric("⚖️ GT Mua - Bán Ròng", f"{f_net_val:+,.2f} Tỷ VND", f"{'Dương (Mua ròng)' if f_net_val >= 0 else 'Âm (Bán ròng)'}")
        
        st.markdown("---")
        st.markdown("##### 🚪 Tình Trạng Room Khối Ngoại")
        
        cur_r = depth_info.get("current_room", 0)
        tot_r = depth_info.get("total_room", 0)
        own_r = depth_info.get("owned_room", 0)
        own_pct = depth_info.get("ownership_pct", 0)
        
        r1, r2, r3 = st.columns(3)
        r1.metric("🏛️ Tổng Room Cho Phép", f"{tot_r:,.0f} CP")
        r2.metric("🔓 Room Khả Dụng (Còn lại)", f"{cur_r:,.0f} CP")
        r3.metric("🔒 NĐTNN Đang Nắm Giữ", f"{own_r:,.0f} CP", f"{own_pct:.2f}% Tổng Room")
        
        st.progress(min(1.0, max(0.0, own_pct / 100.0)), text=f"Tỷ lệ sở hữu nước ngoài: {own_pct:.2f}%")
        
        st.markdown("---")
        flow_chart = create_foreign_flow_chart(selected_symbol, current_net_val_bil=f_net_val)
        st.plotly_chart(flow_chart, use_container_width=True)

