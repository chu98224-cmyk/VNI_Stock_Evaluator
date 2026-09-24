"""
Valuation Engine Module
Calculates intrinsic value using DCF, Graham Formula, Historical P/E & P/B Bands, and DDM.
"""

import math
import numpy as np
import pandas as pd


def calculate_pe_pb_bands(ratio_df: pd.DataFrame, current_eps: float, current_bvps: float, current_price: float):
    """
    Compute 3-5 Year Historical P/E and P/B statistics (Mean, Median, ±1 Std, ±2 Std)
    and estimate Fair Values based on historical multiples.
    """
    res = {
        "pe_median": None,
        "pe_mean": None,
        "pe_std": None,
        "pe_fair_value": None,
        "pb_median": None,
        "pb_mean": None,
        "pb_std": None,
        "pb_fair_value": None,
        "pe_bands": {},
        "pb_bands": {}
    }
    
    if ratio_df is None or ratio_df.empty:
        return res
        
    # Process P/E
    if 'pe' in ratio_df.columns:
        pe_series = ratio_df['pe'].dropna()
        pe_valid = pe_series[(pe_series > 0) & (pe_series < 80)]
        if len(pe_valid) >= 3:
            pe_med = float(pe_valid.median())
            pe_avg = float(pe_valid.mean())
            pe_sd = float(pe_valid.std()) if len(pe_valid) > 1 else 0.0
            
            res["pe_median"] = pe_med
            res["pe_mean"] = pe_avg
            res["pe_std"] = pe_sd
            res["pe_bands"] = {
                "-2SD": max(1.0, pe_avg - 2 * pe_sd),
                "-1SD": max(1.0, pe_avg - pe_sd),
                "Mean": pe_avg,
                "Median": pe_med,
                "+1SD": pe_avg + pe_sd,
                "+2SD": pe_avg + 2 * pe_sd
            }
            if current_eps and current_eps > 0:
                res["pe_fair_value"] = pe_med * current_eps

    # Process P/B
    if 'pb' in ratio_df.columns:
        pb_series = ratio_df['pb'].dropna()
        pb_valid = pb_series[(pb_series > 0) & (pb_series < 15)]
        if len(pb_valid) >= 3:
            pb_med = float(pb_valid.median())
            pb_avg = float(pb_valid.mean())
            pb_sd = float(pb_valid.std()) if len(pb_valid) > 1 else 0.0
            
            res["pb_median"] = pb_med
            res["pb_mean"] = pb_avg
            res["pb_std"] = pb_sd
            res["pb_bands"] = {
                "-2SD": max(0.2, pb_avg - 2 * pb_sd),
                "-1SD": max(0.4, pb_avg - pb_sd),
                "Mean": pb_avg,
                "Median": pb_med,
                "+1SD": pb_avg + pb_sd,
                "+2SD": pb_avg + 2 * pb_sd
            }
            if current_bvps and current_bvps > 0:
                res["pb_fair_value"] = pb_med * current_bvps
                
    return res


def calculate_graham_valuation(eps: float, bvps: float, growth_rate: float = 8.0, bond_yield: float = 3.0):
    """
    Calculate Benjamin Graham Valuations:
    1. Graham Number: sqrt(22.5 * EPS * BVPS)
    2. Revised Graham Formula: EPS * (8.5 + 2g) * 4.4 / Y
    """
    res = {
        "graham_number": None,
        "revised_graham": None,
        "growth_rate_used": growth_rate,
        "bond_yield_used": bond_yield
    }
    
    if eps and bvps and eps > 0 and bvps > 0:
        res["graham_number"] = math.sqrt(22.5 * eps * bvps)
        
    if eps and eps > 0 and bond_yield > 0:
        multiplier = (8.5 + 2 * growth_rate) * (4.4 / bond_yield)
        res["revised_graham"] = eps * multiplier

        
    return res


def calculate_dcf(
    fcf_base: float,
    growth_rate_y1_5: float,
    terminal_growth_rate: float,
    discount_rate: float,
    shares_outstanding: float,
    net_debt: float = 0.0
):
    """
    Discounted Cash Flow (DCF / FCFE) Model
    """
    if not fcf_base or fcf_base <= 0 or not shares_outstanding or shares_outstanding <= 0:
        return None
        
    r = discount_rate / 100.0
    g = growth_rate_y1_5 / 100.0
    g_term = terminal_growth_rate / 100.0
    
    if r <= g_term:
        r = g_term + 0.01
        
    projections = []
    current_fcf = fcf_base
    pv_sum = 0.0
    
    for year in range(1, 6):
        current_fcf = current_fcf * (1 + g)
        discount_factor = (1 + r) ** year
        pv = current_fcf / discount_factor
        pv_sum += pv
        projections.append({
            "year": f"Year {year}",
            "fcf": current_fcf,
            "pv": pv,
            "discount_factor": discount_factor
        })
        
    terminal_fcf = current_fcf * (1 + g_term)
    terminal_value = terminal_fcf / (r - g_term)
    pv_terminal_value = terminal_value / ((1 + r) ** 5)
    
    enterprise_value = pv_sum + pv_terminal_value
    equity_value = enterprise_value - net_debt
    fair_value_per_share = equity_value / shares_outstanding
    
    return {
        "fair_value_per_share": fair_value_per_share,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "pv_explicit_fcf": pv_sum,
        "terminal_value": terminal_value,
        "pv_terminal_value": pv_terminal_value,
        "projections": projections
    }


def generate_dcf_sensitivity_matrix(
    fcf_base: float,
    base_growth: float,
    base_discount: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float = 0.0
):
    """
    Generates a 2D sensitivity matrix comparing varying Discount Rates vs Growth Rates.
    """
    if not fcf_base or fcf_base <= 0 or not shares_outstanding or shares_outstanding <= 0:
        return pd.DataFrame()
        
    growth_shifts = [-4.0, -2.0, 0.0, 2.0, 4.0]
    discount_shifts = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    growths = [round(base_growth + s, 1) for s in growth_shifts]
    discounts = [round(base_discount + s, 1) for s in discount_shifts]
    
    grid = {}
    for d in discounts:
        row = {}
        for g in growths:
            dcf_res = calculate_dcf(
                fcf_base=fcf_base,
                growth_rate_y1_5=g,
                terminal_growth_rate=terminal_growth,
                discount_rate=d,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt
            )
            val = dcf_res["fair_value_per_share"] if dcf_res else 0
            row[f"g = {g}%"] = round(val, 0)
        grid[f"WACC = {d}%"] = row
        
    return pd.DataFrame(grid).T


def calculate_ddm(last_dividend: float, dividend_growth: float = 5.0, required_return: float = 11.0):
    """
    Dividend Discount Model (Gordon Growth Model)
    Fair Value = D1 / (r - g)
    """
    if not last_dividend or last_dividend <= 0:
        return None
        
    r = required_return / 100.0
    g = dividend_growth / 100.0
    
    if r <= g:
        return None
        
    next_dividend = last_dividend * (1 + g)
    fair_value = next_dividend / (r - g)
    return {
        "fair_value": fair_value,
        "next_dividend": next_dividend,
        "dividend_growth": dividend_growth,
        "required_return": required_return
    }


def synthesize_valuations(
    current_price: float,
    pe_fair: float = None,
    pb_fair: float = None,
    graham_number: float = None,
    revised_graham: float = None,
    dcf_fair: float = None,
    ddm_fair: float = None
):
    """
    Aggregate all valuation models into a single weighted fair value and margin of safety.
    """
    models = []
    if pe_fair and pe_fair > 0:
        models.append({"method": "5-Year Median P/E Band", "fair_value": pe_fair, "weight": 0.25})
    if pb_fair and pb_fair > 0:
        models.append({"method": "5-Year Median P/B Band", "fair_value": pb_fair, "weight": 0.20})
    if graham_number and graham_number > 0:
        models.append({"method": "Benjamin Graham Number", "fair_value": graham_number, "weight": 0.15})
    if revised_graham and revised_graham > 0:
        models.append({"method": "Revised Graham Formula", "fair_value": revised_graham, "weight": 0.15})
    if dcf_fair and dcf_fair > 0:
        models.append({"method": "Discounted Cash Flow (DCF)", "fair_value": dcf_fair, "weight": 0.25})
    if ddm_fair and ddm_fair > 0:
        models.append({"method": "Dividend Discount Model (DDM)", "fair_value": ddm_fair, "weight": 0.15})
        
    if not models:
        return {"models": [], "blended_fair_value": None, "margin_of_safety_pct": None, "status": "N/A"}
        
    total_weight = sum(m["weight"] for m in models)
    blended_fair = sum(m["fair_value"] * (m["weight"] / total_weight) for m in models)
    
    margin_pct = 0.0
    status = "Fair Value"
    if current_price and current_price > 0 and blended_fair > 0:
        margin_pct = ((blended_fair - current_price) / current_price) * 100.0
        if margin_pct >= 20.0:
            status = "🔥 Strong Undervalued (High Margin of Safety)"
        elif margin_pct >= 5.0:
            status = "✅ Undervalued (Moderate Margin of Safety)"
        elif margin_pct >= -10.0:
            status = "⚖️ Fairly Valued"
        elif margin_pct >= -25.0:
            status = "⚠️ Overvalued (Caution)"
        else:
            status = "❌ Heavily Overvalued (High Risk)"
            
    return {
        "models": models,
        "blended_fair_value": blended_fair,
        "margin_of_safety_pct": margin_pct,
        "status": status
    }

        
    return res
