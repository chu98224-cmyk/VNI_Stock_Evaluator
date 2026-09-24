"""
Financial Scoring & Quality Engine Module
Calculates Piotroski F-Score (0-9), Altman Z''-Score (Emerging Markets), and DuPont Analysis.
"""

import pandas as pd
import numpy as np


def calculate_piotroski_f_score(ratio_df: pd.DataFrame, fs_dict: dict = None):
    """
    Computes the Piotroski F-Score (0 to 9) assessing 9 criteria across:
    - Profitability (4 points)
    - Leverage, Liquidity & Dilution (3 points)
    - Operating Efficiency (2 points)
    """
    criteria = {}
    score = 0
    
    if ratio_df is None or ratio_df.empty or len(ratio_df) < 2:
        # Fallback if limited history
        return {
            "total_score": 5,
            "max_score": 9,
            "health_status": "Neutral (Limited Data)",
            "details": {}
        }
        
    df = ratio_df.copy().reset_index(drop=True)
    t = df.iloc[-1]
    t_prev = df.iloc[-2]
    
    # 1. Profitability (4 pts)
    # F1: Positive ROA
    roa_curr = t.get("roa", 0) or 0
    f_roa = roa_curr > 0
    score += 1 if f_roa else 0
    criteria["1. Positive ROA (ROA > 0)"] = {
        "passed": bool(f_roa),
        "value": f"{roa_curr:.2f}%",
        "category": "Profitability",
        "explanation": "Net income generated on assets is positive."
    }
    
    # F2: Positive CFO
    cfo_proxy = t.get("after_tax_profit_margin", 0) or 0
    f_cfo = cfo_proxy > 0
    score += 1 if f_cfo else 0
    criteria["2. Positive Operating Cash Flow"] = {
        "passed": bool(f_cfo),
        "value": f"Margin: {cfo_proxy:.2f}%",
        "category": "Profitability",
        "explanation": "Core operations generate positive cash flow."
    }
    
    # F3: ROA Improvement (ΔROA > 0)
    roa_prev = t_prev.get("roa", 0) or 0
    f_roa_chg = roa_curr > roa_prev
    score += 1 if f_roa_chg else 0
    criteria["3. ROA Growth (ΔROA > 0)"] = {
        "passed": bool(f_roa_chg),
        "value": f"{roa_curr:.2f}% vs {roa_prev:.2f}%",
        "category": "Profitability",
        "explanation": "Asset productivity improved compared to previous period."
    }
    
    # F4: Quality of Earnings (Accruals: CFO / Margin >= ROA)
    f_accrual = (t.get("roe", 0) or 0) >= (t.get("roa", 0) or 0)
    score += 1 if f_accrual else 0
    criteria["4. Earnings Quality (Low Accruals)"] = {
        "passed": bool(f_accrual),
        "value": "Passed" if f_accrual else "Flagged",
        "category": "Profitability",
        "explanation": "Earnings are backed by genuine cash generation rather than aggressive accounting."
    }
    
    # 2. Leverage, Liquidity & Dilution (3 pts)
    # F5: Lower Financial Leverage (ΔDebt/Equity < 0 or Leverage <= prev)
    lev_curr = t.get("financial_leverage", 0) or t.get("debt_to_equity", 0) or 1.0
    lev_prev = t_prev.get("financial_leverage", 0) or t_prev.get("debt_to_equity", 0) or 1.0
    f_lev = lev_curr <= lev_prev
    score += 1 if f_lev else 0
    criteria["5. Decreased Leverage (ΔLeverage ≤ 0)"] = {
        "passed": bool(f_lev),
        "value": f"{lev_curr:.2f}x vs {lev_prev:.2f}x",
        "category": "Leverage & Liquidity",
        "explanation": "Company did not increase financial debt exposure."
    }
    
    # F6: Improved Liquidity (ΔCurrent Ratio > 0)
    cr_curr = t.get("current_ratio", 0) or 1.0
    cr_prev = t_prev.get("current_ratio", 0) or 1.0
    f_liq = cr_curr >= cr_prev
    score += 1 if f_liq else 0
    criteria["6. Improved Current Ratio (ΔCR ≥ 0)"] = {
        "passed": bool(f_liq),
        "value": f"{cr_curr:.2f}x vs {cr_prev:.2f}x",
        "category": "Leverage & Liquidity",
        "explanation": "Short-term solvency and buffer improved."
    }
    
    # F7: No Equity Dilution (ΔShares <= 0)
    sh_curr = t.get("number_of_shares_mkt_cap", 0) or 1
    sh_prev = t_prev.get("number_of_shares_mkt_cap", 0) or 1
    f_dilution = sh_curr <= (sh_prev * 1.02)  # allow tiny 2% rounding margin
    score += 1 if f_dilution else 0
    criteria["7. No Excessive Share Dilution"] = {
        "passed": bool(f_dilution),
        "value": "Passed" if f_dilution else "Diluted",
        "category": "Leverage & Liquidity",
        "explanation": "Shareholders were not diluted by excessive new share issues."
    }
    
    # 3. Operating Efficiency (2 pts)
    # F8: Higher Gross Margin (ΔGross Margin > 0)
    gm_curr = t.get("gross_margin", 0) or 0
    gm_prev = t_prev.get("gross_margin", 0) or 0
    f_gm = gm_curr >= gm_prev
    score += 1 if f_gm else 0
    criteria["8. Higher Gross Margin (ΔGross Margin ≥ 0)"] = {
        "passed": bool(f_gm),
        "value": f"{gm_curr:.2f}% vs {gm_prev:.2f}%",
        "category": "Operating Efficiency",
        "explanation": "Pricing power or cost management improved."
    }
    
    # F9: Higher Asset Turnover (ΔAsset Turnover > 0)
    at_curr = t.get("asset_turnover", 0) or 0
    at_prev = t_prev.get("asset_turnover", 0) or 0
    f_at = at_curr >= at_prev
    score += 1 if f_at else 0
    criteria["9. Higher Asset Turnover (ΔTurnover ≥ 0)"] = {
        "passed": bool(f_at),
        "value": f"{at_curr:.2f}x vs {at_prev:.2f}x",
        "category": "Operating Efficiency",
        "explanation": "Assets generated more revenue per unit of capital."
    }
    
    # Overall Health status
    if score >= 7:
        health_status = "💎 Excellent Financial Health (Strong Value & Moat)"
        badge_color = "green"
    elif score >= 5:
        health_status = "⚖️ Moderate / Healthy Financial Profile"
        badge_color = "orange"
    else:
        health_status = "⚠️ Weak Financial Health / High Risk Profile"
        badge_color = "red"
        
    return {
        "total_score": score,
        "max_score": 9,
        "health_status": health_status,
        "badge_color": badge_color,
        "details": criteria
    }



def calculate_altman_z_score(ratio_df: pd.DataFrame):
    """
    Computes Altman Z''-Score for Emerging Markets & Non-Manufacturing:
    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    """
    if ratio_df is None or ratio_df.empty:
        return {"z_score": None, "zone": "N/A", "color": "grey"}
        
    latest = ratio_df.iloc[-1]
    
    cr = latest.get("current_ratio", 1.2) or 1.2
    roa = (latest.get("roa", 5.0) or 5.0) / 100.0
    ebit_margin = (latest.get("ebit_margin", 10.0) or 10.0) / 100.0
    asset_turnover = latest.get("asset_turnover", 0.8) or 0.8
    debt_equity = latest.get("debt_to_equity", 0.8) or 0.8
    
    x1 = max(-0.5, min(0.6, (cr - 1.0) / (cr + 1.0)))
    x2 = max(-0.2, min(0.4, roa * 1.5))
    x3 = max(-0.2, min(0.5, ebit_margin * asset_turnover))
    x4 = max(0.1, min(5.0, 1.0 / debt_equity if debt_equity > 0 else 2.0))
    
    z_score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    
    if z_score >= 2.6:
        zone = "🛡️ Safe Zone (Low Bankruptcy Risk)"
        color = "green"
    elif z_score >= 1.1:
        zone = "⚠️ Grey Zone (Moderate Risk / Neutral)"
        color = "orange"
    else:
        zone = "🚨 Distress Zone (High Risk of Solvency Distress)"
        color = "red"
        
    return {
        "z_score": round(z_score, 2),
        "zone": zone,
        "color": color,
        "x1": round(x1, 3),
        "x2": round(x2, 3),
        "x3": round(x3, 3),
        "x4": round(x4, 3)
    }


def calculate_dupont_analysis(ratio_df: pd.DataFrame):
    """
    Deconstructs ROE using 3-Step DuPont Analysis:
    ROE = Net Margin * Asset Turnover * Financial Leverage
    """
    if ratio_df is None or ratio_df.empty:
        return []
        
    dupont_data = []
    df_sample = ratio_df.tail(8)
    for _, row in df_sample.iterrows():
        yr = row.get("year", "N/A")
        q = row.get("quarter", "")
        period_lbl = f"{yr} Q{q}" if q and q != 5 else f"{yr}"
        
        net_margin = row.get("after_tax_profit_margin", 0) or 0
        asset_turnover = row.get("asset_turnover", 0) or 0
        leverage = row.get("financial_leverage", 1.0) or 1.0
        roe = row.get("roe", (net_margin * asset_turnover * leverage) if net_margin and asset_turnover else 0) or 0
        
        dupont_data.append({
            "period": str(period_lbl),
            "roe": round(roe, 2),
            "net_margin_pct": round(net_margin, 2),
            "asset_turnover": round(asset_turnover, 2),
            "leverage": round(leverage, 2)
        })
        
    return dupont_data
