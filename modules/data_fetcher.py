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
    from vnstock import Quote, Finance, Company, Listing, Trading
except ImportError:
    Quote = None
    Finance = None
    Company = None
    Listing = None
    Trading = None


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
@st.cache_data(ttl=3600, show_spinner=False)
def get_exchange_symbols(exchange: str = "HSX"):
    """Fetch all ticker symbols belonging to an exchange (HSX/HOSE, VN30, HNX, UPCOM)."""
    exchange = exchange.upper().strip()
    if exchange == "HOSE":
        exchange = "HSX"
        
    try:
        if Listing is not None:
            l = Listing(source='vci')
            if exchange == "VN30":
                df_vn30 = l.symbols_by_group('VN30')
                if df_vn30 is not None and not df_vn30.empty:
                    if isinstance(df_vn30, pd.Series):
                        return df_vn30.tolist()
                    elif 'symbol' in df_vn30.columns:
                        return df_vn30['symbol'].tolist()
            else:
                df = l.symbols_by_exchange()
                if df is not None and not df.empty and 'exchange' in df.columns:
                    filtered = df[df['exchange'] == exchange]['symbol'].dropna().tolist()
                    if filtered:
                        return filtered
    except Exception as e:
        print(f"Error fetching symbols for exchange {exchange}: {e}")
        
    if exchange in ["HSX", "HOSE"]:
        hose_defaults = []
        for syms in POPULAR_VN_STOCKS.values():
            hose_defaults.extend(syms)
        return list(dict.fromkeys(hose_defaults))
    elif exchange == "VN30":
        return POPULAR_VN_STOCKS["⭐ VN30 Top Picks"]
    return ["HPG", "FPT", "SSI", "VNM", "VCB", "MBB", "TCB", "MWG", "VIC", "VHM"]


@st.cache_data(ttl=30, show_spinner=False)
def get_live_price_board(symbols: list):
    """
    Fetch real-time price board for a list of symbols with bids, asks, match, volume, and foreign trade.
    """
    if not symbols:
        return pd.DataFrame()
        
    symbols = [str(s).upper().strip() for s in symbols if isinstance(s, str) and str(s).strip()]
    if not symbols:
        return pd.DataFrame()
        
    try:
        if Trading is not None:
            t = Trading(source='vci')
            all_batches = [symbols[i:i + 35] for i in range(0, len(symbols), 35)]
            records = []
            
            for batch in all_batches:
                pb = t.price_board(batch)
                if pb is not None and not pb.empty:
                    for _, row in pb.iterrows():
                        sym = row.get(('listing', 'symbol'))
                        if not sym or sym != sym or not isinstance(sym, str):
                            continue
                        sym = str(sym).strip().upper()
                        
                        ref = float(row.get(('listing', 'ref_price'), 0) or 0)
                        ceil = float(row.get(('listing', 'ceiling'), 0) or 0)
                        flr = float(row.get(('listing', 'floor'), 0) or 0)
                        match_p = float(row.get(('match', 'match_price'), 0) or 0)
                        match_v = float(row.get(('match', 'match_vol'), 0) or 0)
                        tot_vol = float(row.get(('match', 'accumulated_volume'), 0) or 0)
                        tot_val = float(row.get(('match', 'accumulated_value'), 0) or 0)
                        high = float(row.get(('match', 'highest'), 0) or 0)
                        low = float(row.get(('match', 'lowest'), 0) or 0)
                        avg_p = float(row.get(('match', 'avg_match_price'), 0) or 0)
                        f_buy = float(row.get(('match', 'foreign_buy_volume'), 0) or 0)
                        f_sell = float(row.get(('match', 'foreign_sell_volume'), 0) or 0)
                        f_buy_val = float(row.get(('match', 'foreign_buy_value'), 0) or 0)
                        f_sell_val = float(row.get(('match', 'foreign_sell_value'), 0) or 0)
                        
                        b1_p = float(row.get(('bid_ask', 'bid_1_price'), 0) or 0)
                        b1_v = float(row.get(('bid_ask', 'bid_1_volume'), 0) or 0)
                        b2_p = float(row.get(('bid_ask', 'bid_2_price'), 0) or 0)
                        b2_v = float(row.get(('bid_ask', 'bid_2_volume'), 0) or 0)
                        b3_p = float(row.get(('bid_ask', 'bid_3_price'), 0) or 0)
                        b3_v = float(row.get(('bid_ask', 'bid_3_volume'), 0) or 0)
                        
                        a1_p = float(row.get(('bid_ask', 'ask_1_price'), 0) or 0)
                        a1_v = float(row.get(('bid_ask', 'ask_1_volume'), 0) or 0)
                        a2_p = float(row.get(('bid_ask', 'ask_2_price'), 0) or 0)
                        a2_v = float(row.get(('bid_ask', 'ask_2_volume'), 0) or 0)
                        a3_p = float(row.get(('bid_ask', 'ask_3_price'), 0) or 0)
                        a3_v = float(row.get(('bid_ask', 'ask_3_volume'), 0) or 0)
                        
                        change = (match_p - ref) if (ref > 0 and match_p > 0) else 0.0
                        change_pct = ((match_p - ref) / ref * 100) if (ref > 0 and match_p > 0) else 0.0
                        
                        if match_p == 0:
                            status = "ref"
                        elif ceil > 0 and match_p >= ceil:
                            status = "ceiling"
                        elif flr > 0 and match_p <= flr:
                            status = "floor"
                        elif match_p > ref:
                            status = "up"
                        elif match_p < ref:
                            status = "down"
                        else:
                            status = "ref"
                            
                        records.append({
                            "symbol": sym, "ref": ref, "ceil": ceil, "floor": flr,
                            "b3_p": b3_p, "b3_v": b3_v, "b2_p": b2_p, "b2_v": b2_v, "b1_p": b1_p, "b1_v": b1_v,
                            "match_p": match_p, "change": change, "change_pct": change_pct, "match_v": match_v,
                            "a1_p": a1_p, "a1_v": a1_v, "a2_p": a2_p, "a2_v": a2_v, "a3_p": a3_p, "a3_v": a3_v,
                            "total_vol": tot_vol, "total_val": tot_val, "high": high, "low": low, "avg": avg_p,
                            "f_buy": f_buy, "f_sell": f_sell, "f_net": f_buy - f_sell,
                            "f_buy_val": f_buy_val, "f_sell_val": f_sell_val, "status": status
                        })
            if records:
                return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching live price board: {e}")
    return pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def get_intraday_ticks(symbol: str):
    """Fetch live intraday tick-by-tick trades (time, price, volume, match_type)."""
    symbol = symbol.upper().strip()
    try:
        if Quote is not None:
            q = Quote(symbol=symbol, source='vci')
            df = q.intraday()
            if df is not None and not df.empty:
                df = df.copy()
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                if 'price' in df.columns:
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    if df['price'].median() < 1000:
                        df['price'] = df['price'] * 1000
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                return df
    except Exception as e:
        print(f"Error fetching intraday ticks for {symbol}: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_intraday_history(symbol: str, interval: str = '1m', days: int = 5):
    """
    Fetch comprehensive minute-by-minute (1m, 5m, 15m, 1H) intraday OHLCV bars with VWAP.
    """
    symbol = symbol.upper().strip()
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    
    try:
        if Quote is not None:
            q = Quote(source='vci', symbol=symbol)
            df = q.history(start=start_date, end=end_date, interval=interval)
            if df is not None and not df.empty:
                df = df.copy()
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.sort_values('time').reset_index(drop=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['close']).reset_index(drop=True)
                if not df.empty and df['close'].median() < 1000:
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = df[col] * 1000
                
                # Compute cumulative VWAP
                df['cum_val'] = (df['close'] * df['volume']).cumsum()
                df['cum_vol'] = df['volume'].cumsum()
                df['vwap'] = df['cum_val'] / df['cum_vol'].replace(0, np.nan)
                return df
    except Exception as e:
        print(f"Error fetching intraday history for {symbol}: {e}")
        
    return pd.DataFrame()




@st.cache_data(ttl=30, show_spinner=False)
def get_market_indices():
    """Fetch key Vietnamese market indices (VN-INDEX, VN30, HNX-INDEX, UPCOM)."""
    indices = [
        {"symbol": "VNINDEX", "name": "VN INDEX", "exchange": "HOSE"},
        {"symbol": "VN30", "name": "VN 30", "exchange": "HOSE"},
        {"symbol": "HNXINDEX", "name": "HN INDEX", "exchange": "HNX"},
        {"symbol": "UPCOMINDEX", "name": "UPCOM INDEX", "exchange": "UPCOM"}
    ]
    
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
    
    res = []
    for item in indices:
        sym = item["symbol"]
        try:
            if Quote is not None:
                q = Quote(symbol=sym, source='vci')
                df = q.history(start=start_date, end=end_date)
                if df is not None and not df.empty:
                    df = df.sort_values('time').reset_index(drop=True)
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    close = float(latest.get('close', 0))
                    prev_close = float(prev.get('close', close))
                    change = close - prev_close
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                    vol = float(latest.get('volume', 0))
                    val_est = (close * vol / 1e6) if close < 5000 else (vol * 15000 / 1e9)
                    
                    res.append({
                        "symbol": sym,
                        "name": item["name"],
                        "exchange": item["exchange"],
                        "points": close,
                        "change": change,
                        "change_pct": change_pct,
                        "volume": vol,
                        "value_bil": val_est,
                        "history": df['close'].tolist()[-10:]
                    })
                    continue
        except Exception as e:
            print(f"Error fetching index {sym}: {e}")
            
        res.append({
            "symbol": sym,
            "name": item["name"],
            "exchange": item["exchange"],
            "points": 1785.11 if sym == "VNINDEX" else (1938.5 if sym == "VN30" else (272.21 if sym == "HNXINDEX" else 125.21)),
            "change": 10.02 if sym == "VNINDEX" else (6.35 if sym == "VN30" else -0.56),
            "change_pct": 0.56 if sym == "VNINDEX" else (0.33 if sym == "VN30" else -0.21),
            "volume": 467000000,
            "value_bil": 11600.0,
            "history": [1760, 1770, 1765, 1775, 1780, 1785.11]
        })
        
    return res



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
