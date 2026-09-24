"""
Data Fetcher Module for Vietnamese Stock Market
Integrates with vnstock (VCI, KBS data sources) with caching and error handling.
"""

import os
import datetime
import pandas as pd
import numpy as np
import streamlit as st

os.environ['VNSTOCK_TELEMETRY'] = 'off'

try:
    from vnstock import Quote, Finance, Company, Listing
except ImportError:
    Quote = None
    Finance = None
    Company = None
    Listing = None


POPULAR_VN_STOCKS = {
    "⭐ VN30 Top Picks": ["HPG", "FPT", "VNM", "VCB", "MBB", "TCB", "MWG", "VIC", "VHM", "GAS", "MSN", "SSI", "STB", "ACB", "VRE"],
    "🏦 Banking (Ngân hàng)": ["VCB", "MBB", "TCB", "ACB", "BID", "CTG", "STB", "HDB", "VPB", "LPB", "VIB", "TPB"],
    "💻 Tech & Retail (Công nghệ & Bán lẻ)": ["FPT", "MWG", "PNJ", "DGW", "FRT", "CTR", "ELC", "CMG"],
    "🏗️ Steel & Materials (Thép & Vật liệu)": ["HPG", "NKG", "HSG", "VGS", "HT1", "BCC"],
    "🏘️ Real Estate (Bất động sản)": ["VHM", "VIC", "KDH", "NLG", "DXG", "PDR", "DIG", "NVL", "VRE", "KBC", "IDC"],
    "📈 Securities (Chứng khoán)": ["SSI", "VND", "VCI", "HCM", "SHS", "MBS", "FTS", "BSI", "CTS"],
    "🥛 Consumer & Food (Tiêu dùng & Thực phẩm)": ["VNM", "MSN", "SAB", "KDC", "QNS", "MCH", "DBC", "BAF"],
    "⚡ Energy & Utilities (Năng lượng & Tiện ích)": ["GAS", "POW", "REE", "PVD", "PVS", "PLX", "BSR", "NT2", "GEG", "HDG"],
    "🚢 Logistics & Ports (Cảng biển & Vận tải)": ["GMD", "HAH", "VSC", "PVT", "VOS", "PHP"],
    "🌾 Chemical & Fertilizer (Hóa chất & Phân bón)": ["DGC", "DPM", "DCM", "CSV", "BFC"]
}


@st.cache_data(ttl=1800, show_spinner=False)
def get_all_stock_symbols():
    """Retrieve all available symbols from the exchange or fallback to defaults."""
    try:
        if Listing is not None:
            l = Listing(source='VCI')
            df = l.all_symbols()
            if df is not None and not df.empty and 'symbol' in df.columns:
                return df[['symbol', 'organ_name']].drop_duplicates(subset=['symbol']).to_dict('records')
    except Exception:
        pass
    
    all_syms = []
    for grp, syms in POPULAR_VN_STOCKS.items():
        for s in syms:
            all_syms.append({"symbol": s, "organ_name": f"{s} Corporation"})
    return all_syms


@st.cache_data(ttl=600, show_spinner=False)
def get_stock_price_history(symbol: str, start_date: str = None, end_date: str = None, days: int = 365 * 3):
    """Fetch daily OHLCV price history for a given ticker."""
    symbol = symbol.upper().strip()
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
    try:
        q = Quote(source='VCI', symbol=symbol)
        df = q.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.copy()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if df['close'].median() < 1000:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = df[col] * 1000
            return df
    except Exception as e:
        print(f"Error fetching history from VCI for {symbol}: {e}")
        
    try:
        q = Quote(source='KBS', symbol=symbol)
        df = q.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.copy()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').reset_index(drop=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if df['close'].median() < 1000:
                for col in ['open', 'high', 'low', 'close']:
                    if col in df.columns:
                        df[col] = df[col] * 1000
            return df
    except Exception as e:
        print(f"Error fetching history from KBS for {symbol}: {e}")
        
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_company_overview(symbol: str):
    """Fetch company overview, sector, market cap, and trading summary."""
    symbol = symbol.upper().strip()
    try:
        c = Company(source='VCI', symbol=symbol)
        df = c.overview()
        if df is not None and not df.empty:
            return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Error fetching overview for {symbol}: {e}")
    return {}




@st.cache_data(ttl=1800, show_spinner=False)
def get_ratio_summary(symbol: str):
    """Fetch comprehensive historical quarterly/yearly financial ratios."""
    symbol = symbol.upper().strip()
    try:
        c = Company(source='VCI', symbol=symbol)
        df = c.ratio_summary()
        if df is not None and not df.empty:
            df = df.copy()
            numeric_cols = [
                'pe', 'pb', 'ps', 'ev_to_ebitda', 'dividend_yield', 'roe', 'roa', 'roic',
                'current_ratio', 'quick_ratio', 'cash_ratio', 'debt_to_equity',
                'gross_margin', 'ebit_margin', 'pre_tax_profit_margin', 'after_tax_profit_margin',
                'asset_turnover', 'fixed_asset_turnover', 'day_sale_outstanding',
                'days_inventory_outstanding', 'days_payable_outstanding', 'cash_cycle',
                'financial_leverage', 'market_cap', 'owners_equity', 'net_interest_margin', 'npl', 'casa_ratio'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
            if 'quarter' in df.columns:
                df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')
                
            df = df.sort_values(['year', 'quarter']).reset_index(drop=True)
            return df
    except Exception as e:
        print(f"Error fetching ratio summary for {symbol}: {e}")
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_financial_statements(symbol: str, period: str = 'year'):
    """Fetch Income Statement, Balance Sheet, and Cash Flow statements."""
    symbol = symbol.upper().strip()
    res = {"is": pd.DataFrame(), "bs": pd.DataFrame(), "cf": pd.DataFrame()}
    try:
        f = Finance(source='VCI', symbol=symbol)
        try:
            is_df = f.income_statement(period=period)
            if is_df is not None and not is_df.empty:
                res["is"] = is_df
        except Exception:
            pass
            
        try:
            bs_df = f.balance_sheet(period=period)
            if bs_df is not None and not bs_df.empty:
                res["bs"] = bs_df
        except Exception:
            pass
            
        try:
            cf_df = f.cash_flow(period=period)
            if cf_df is not None and not cf_df.empty:
                res["cf"] = cf_df
        except Exception:
            pass
    except Exception as e:
        print(f"Error fetching financial statements for {symbol}: {e}")
        
    return res


def extract_latest_fundamental_metrics(symbol: str, current_price: float = None):
    """Aggregate latest fundamental metrics required for valuation & scoring."""
    overview = get_company_overview(symbol)
    ratio_df = get_ratio_summary(symbol)
    
    price = current_price or overview.get("current_price", 0)
    if price and price > 0 and price < 1000:
        price = price * 1000
        
    metrics = {
        "symbol": symbol,
        "company_name": overview.get("organ_name", symbol),
        "sector": overview.get("sector", "N/A"),
        "market_cap": overview.get("market_cap", 0),
        "shares_outstanding": overview.get("issue_share", 0) or overview.get("number_of_shares_mkt_cap", 0),
        "current_price": price,
        "pe": None,
        "pb": None,
        "ps": None,
        "roe": None,
        "roa": None,
        "roic": None,
        "dividend_yield": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "quick_ratio": None,
        "gross_margin": None,
        "net_margin": None,
        "eps": None,
        "bvps": None,
        "dso": None,
        "dio": None,
        "dpo": None,
        "historical_pe": [],
        "historical_pb": []
    }
    
    if ratio_df is not None and not ratio_df.empty:
        latest = ratio_df.dropna(subset=['pe', 'pb'], how='all').iloc[-1] if not ratio_df.empty else None
        if latest is not None:
            metrics["pe"] = latest.get("pe", None)
            metrics["pb"] = latest.get("pb", None)
            metrics["ps"] = latest.get("ps", None)
            metrics["roe"] = latest.get("roe", None)
            metrics["roa"] = latest.get("roa", None)
            metrics["roic"] = latest.get("roic", None)
            metrics["dividend_yield"] = latest.get("dividend_yield", None)
            metrics["debt_to_equity"] = latest.get("debt_to_equity", None)
            metrics["current_ratio"] = latest.get("current_ratio", None)
            metrics["quick_ratio"] = latest.get("quick_ratio", None)
            metrics["gross_margin"] = latest.get("gross_margin", None)
            metrics["net_margin"] = latest.get("after_tax_profit_margin", None)
            metrics["dso"] = latest.get("day_sale_outstanding", None)
            metrics["dio"] = latest.get("days_inventory_outstanding", None)
            metrics["dpo"] = latest.get("days_payable_outstanding", None)
            
            pe_clean = ratio_df['pe'].dropna().tolist()
            pb_clean = ratio_df['pb'].dropna().tolist()
            metrics["historical_pe"] = [p for p in pe_clean if 0 < p < 100]
            metrics["historical_pb"] = [b for b in pb_clean if 0 < b < 20]
            
            p = metrics["current_price"]
            if p and p > 0:
                if metrics["pe"] and metrics["pe"] > 0:
                    metrics["eps"] = p / metrics["pe"]
                if metrics["pb"] and metrics["pb"] > 0:
                    metrics["bvps"] = p / metrics["pb"]
                    
    return metrics

@st.cache_data(ttl=1800, show_spinner=False)
def get_company_profile(symbol: str):
    """Fetch company profile, business description, history."""
    symbol = symbol.upper().strip()
    try:
        c = Company(source='VCI', symbol=symbol)
        df = c.profile()
        if df is not None and not df.empty:
            return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Error fetching profile for {symbol}: {e}")
    return {}
