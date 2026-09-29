"""
Market Board Module (VCBS-Style Live Price Board Component)
Renders real-time exchange board with 3-level order depth, indices overview, and fast stock selector.
"""

import streamlit as st
import pandas as pd


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
