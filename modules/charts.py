"""
Interactive Charting Module using Plotly
Supports Candlesticks, Bollinger Bands, Moving Averages (MA20/50/200), Volume, RSI, MACD, and Valuation Bands.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_technical_chart(
    df: pd.DataFrame,
    symbol: str,
    show_ma20: bool = True,
    show_ma50: bool = True,
    show_ma200: bool = True,
    show_bollinger: bool = True,
    show_rsi: bool = True,
    show_macd: bool = False,
    fair_value_ref: float = None
):
    """Constructs technical chart with Candlesticks, Bollinger Bands, MAs, Volume, RSI, MACD."""
    if df is None or df.empty or len(df) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient price data", showarrow=False, font=dict(size=16))
        return fig

    data = df.copy().sort_values('time').reset_index(drop=True)
    data['MA20'] = data['close'].rolling(20).mean()
    data['MA50'] = data['close'].rolling(50).mean()
    data['MA200'] = data['close'].rolling(200).mean()
    data['BB_Mid'] = data['close'].rolling(20).mean()
    data['BB_Std'] = data['close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['BB_Std'])
    data['Vol_MA20'] = data['volume'].rolling(20).mean()
    
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    rows, row_heights, titles = 2, [0.70, 0.30], [f"{symbol} Price & Overlays", "Volume"]
    if show_rsi and not show_macd:
        rows, row_heights, titles = 3, [0.60, 0.20, 0.20], [f"{symbol} Price & Overlays", "Volume", "RSI (14)"]
    elif show_macd and not show_rsi:
        rows, row_heights, titles = 3, [0.60, 0.20, 0.20], [f"{symbol} Price & Overlays", "Volume", "MACD"]
    elif show_rsi and show_macd:
        rows, row_heights, titles = 4, [0.50, 0.16, 0.17, 0.17], [f"{symbol} Price & Overlays", "Volume", "RSI (14)", "MACD"]
        
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights, subplot_titles=titles)
    
    fig.add_trace(go.Candlestick(
        x=data['time'], open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name="OHLC", increasing_line_color='#00C087', decreasing_line_color='#FF3B30',
        increasing_fillcolor='#00C087', decreasing_fillcolor='#FF3B30'
    ), row=1, col=1)
    
    if show_bollinger:
        fig.add_trace(go.Scatter(x=data['time'], y=data['BB_Upper'], name='BB Upper', line=dict(color='rgba(56, 189, 248, 0.6)', width=1.2, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['time'], y=data['BB_Lower'], name='BB Lower', line=dict(color='rgba(56, 189, 248, 0.6)', width=1.2, dash='dot'), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.06)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['time'], y=data['BB_Mid'], name='BB Middle', line=dict(color='rgba(56, 189, 248, 0.9)', width=1.4)), row=1, col=1)
        
    if show_ma20 and not show_bollinger:
        fig.add_trace(go.Scatter(x=data['time'], y=data['MA20'], name='MA 20', line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    if show_ma50:
        fig.add_trace(go.Scatter(x=data['time'], y=data['MA50'], name='MA 50', line=dict(color='#3B82F6', width=1.8)), row=1, col=1)
    if show_ma200:
        fig.add_trace(go.Scatter(x=data['time'], y=data['MA200'], name='MA 200', line=dict(color='#A855F7', width=2.2)), row=1, col=1)
        
    if fair_value_ref and fair_value_ref > 0:
        fig.add_hline(y=fair_value_ref, line_dash="dash", line_color="#EAB308", annotation_text=f"Fair: {fair_value_ref:,.0f} VND", row=1, col=1)
        
    vol_colors = ['#00C087' if c >= o else '#FF3B30' for o, c in zip(data['open'], data['close'])]
    fig.add_trace(go.Bar(x=data['time'], y=data['volume'], name='Volume', marker_color=vol_colors, opacity=0.8), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['time'], y=data['Vol_MA20'], name='Vol MA20', line=dict(color='#F59E0B', width=1.2)), row=2, col=1)
    
    curr_row = 3
    if show_rsi:
        fig.add_trace(go.Scatter(x=data['time'], y=data['RSI'], name='RSI (14)', line=dict(color='#EC4899', width=1.8)), row=curr_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.7)", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(34, 197, 94, 0.7)", row=curr_row, col=1)
        curr_row += 1
        
    if show_macd:
        fig.add_trace(go.Scatter(x=data['time'], y=data['MACD'], name='MACD', line=dict(color='#38BDF8', width=1.5)), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=data['time'], y=data['MACD_Signal'], name='Signal', line=dict(color='#F97316', width=1.5)), row=curr_row, col=1)
        hist_colors = ['#00C087' if v >= 0 else '#FF3B30' for v in data['MACD_Hist']]
        fig.add_trace(go.Bar(x=data['time'], y=data['MACD_Hist'], name='Histogram', marker_color=hist_colors), row=curr_row, col=1)
        
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
        xaxis_rangeslider_visible=False, height=720 if rows >= 3 else 520,
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#21262D")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#21262D")
    return fig



def create_valuation_bands_chart(ratio_df: pd.DataFrame, current_price: float, metric_type: str = "pe"):
    """
    Renders historical P/E or P/B percentile corridor chart.
    """
    if ratio_df is None or ratio_df.empty or metric_type not in ratio_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No valuation multiple history", showarrow=False)
        return fig
        
    df = ratio_df.dropna(subset=[metric_type]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid multiple data", showarrow=False)
        return fig
        
    df['label'] = df.apply(lambda r: f"{int(r['year'])} Q{int(r['quarter'])}" if pd.notna(r.get('quarter')) and r.get('quarter') != 5 else str(int(r['year'])), axis=1)
    
    val_series = df[metric_type]
    avg = val_series.mean()
    sd = val_series.std() if len(val_series) > 1 else 0
    med = val_series.median()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['label'], y=[avg + 2*sd]*len(df), name='+2 SD (Overvalued)', line=dict(color='#EF4444', dash='dash')))
    fig.add_trace(go.Scatter(x=df['label'], y=[avg + sd]*len(df), name='+1 SD', line=dict(color='#F97316', dash='dot')))
    fig.add_trace(go.Scatter(x=df['label'], y=[med]*len(df), name='Median Fair Value', line=dict(color='#EAB308', width=2)))
    fig.add_trace(go.Scatter(x=df['label'], y=[max(0.1, avg - sd)]*len(df), name='-1 SD', line=dict(color='#38BDF8', dash='dot')))
    fig.add_trace(go.Scatter(x=df['label'], y=[max(0.1, avg - 2*sd)]*len(df), name='-2 SD (Deep Value)', line=dict(color='#22C55E', dash='dash')))
    
    fig.add_trace(go.Scatter(
        x=df['label'], y=val_series, name=f"Historical {metric_type.upper()}",
        mode='lines+markers', line=dict(color='#A855F7', width=2.5),
        marker=dict(size=6, color='#D8B4FE')
    ))
    
    title_text = "5-Year Historical P/E Multiple Corridor" if metric_type == "pe" else "5-Year Historical P/B Multiple Corridor"
    fig.update_layout(
        title=title_text, template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
        height=380, margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_dupont_chart(dupont_data: list):
    """
    Renders 3-Step DuPont decomposition bar & line chart.
    """
    if not dupont_data:
        fig = go.Figure()
        fig.add_annotation(text="No DuPont data available", showarrow=False)
        return fig
        
    df = pd.DataFrame(dupont_data)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=df['period'], y=df['net_margin_pct'], name="Net Margin (%)", marker_color='#38BDF8'), secondary_y=False)
    fig.add_trace(go.Bar(x=df['period'], y=df['asset_turnover'], name="Asset Turnover (x)", marker_color='#F59E0B'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['period'], y=df['roe'], name="ROE (%)", mode='lines+markers', line=dict(color='#00C087', width=3), marker=dict(size=8)), secondary_y=True)
    
    fig.update_layout(
        title="DuPont ROE Drivers Decomposition", template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
        height=380, margin=dict(l=30, r=30, t=50, b=30), barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Margin (%) / Turnover (x)", secondary_y=False, showgrid=True, gridcolor="#21262D")
    fig.update_yaxes(title_text="ROE (%)", secondary_y=True, showgrid=False)
    return fig
