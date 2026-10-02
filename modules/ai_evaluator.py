"""
AI Evaluator & Advisory Engine for Vietnamese Equities.
Provides Quantitative Rule-Based Analysis and Multi-LLM Chat capabilities (Gemini, OpenAI, DeepSeek, Ollama).
"""

import re
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from modules.data_fetcher import (
    get_stock_price_history,
    get_company_overview,
    get_company_profile,
    get_ratio_summary,
    extract_latest_fundamental_metrics,
    get_symbol_live_depth_and_foreign
)
from modules.live_analytics import (
    calculate_technical_signals,
    calculate_trade_setup,
    calculate_pivot_points
)
from modules.valuation_engine import (
    calculate_pe_pb_bands,
    calculate_graham_valuation,
    calculate_dcf,
    calculate_ddm,
    synthesize_valuations
)
from modules.scoring_engine import (
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dupont_analysis
)


def extract_stock_symbol_from_prompt(prompt: str, default_symbol: str = "HPG") -> str:
    """
    Extracts Vietnam 3-letter stock ticker from user prompt.
    E.g., 'Định giá HPG ở vùng 28k thế nào' -> 'HPG'
    """
    if not prompt:
        return default_symbol
        
    cleaned = prompt.upper()
    prefix_match = re.search(r'(?:MÃ|CỔ PHIẾU|CP|CON|TICKER|STOCK)\s+([A-Z0-9]{3})\b', cleaned)
    if prefix_match:
        return prefix_match.group(1)
        
    stopwords = {"CHO", "NÊN", "GIA", "GIÁ", "MUA", "BÁN", "BAN", "XEM", "HOI", "HỎI", "TÔI", "TOI", "THE", "THẾ", "NAO", "NÀO", "VỚI", "VOI", "TẠI", "TAI", "KHI", "CÓ", "VÀ", "LA", "LÀ", "Ở", "O"}
    candidates = re.findall(r'\b[A-Z]{3}\b', cleaned)
    
    for c in candidates:
        if c not in stopwords:
            return c
            
    return default_symbol


def extract_target_price_from_prompt(prompt: str) -> Optional[float]:
    """
    Extracts explicit price mentioned in user query.
    E.g. 'giá 28k', 'ở 28.5k', 'giá 28,500', 'vùng 110k'
    """
    if not prompt:
        return None
        
    k_match = re.search(r'(\d+(?:[.,]\d+)?)\s*k\b', prompt, re.IGNORECASE)
    if k_match:
        val_str = k_match.group(1).replace(',', '.')
        try:
            return float(val_str) * 1000.0
        except ValueError:
            pass
            
    full_match = re.search(r'giá\s+(\d{2,3}(?:[.,]\d{3})+)', prompt, re.IGNORECASE)
    if full_match:
        val_str = full_match.group(1).replace('.', '').replace(',', '')
        try:
            return float(val_str)
        except ValueError:
            pass
            
    return None


def build_stock_dossier(symbol: str, target_eval_price: Optional[float] = None) -> Dict[str, Any]:
    """
    Assembles a comprehensive quantitative dossier of the stock
    including technicals, valuation models, financial health, and live statistics.
    """
    symbol = symbol.upper().strip()
    
    # 1. Price history & live metrics
    price_df = get_stock_price_history(symbol, days=730)
    overview = get_company_overview(symbol)
    profile = get_company_profile(symbol)
    ratio_df = get_ratio_summary(symbol)
    metrics = extract_latest_fundamental_metrics(symbol)
    
    current_price = 0.0
    if price_df is not None and not price_df.empty:
        current_price = float(price_df['close'].iloc[-1])
    elif metrics.get("current_price"):
        current_price = float(metrics["current_price"])
        
    eval_price = target_eval_price if (target_eval_price and target_eval_price > 0) else current_price
    
    # Calculate EPS & BVPS
    eps = metrics.get("eps")
    bvps = metrics.get("bvps")
    if (not eps or eps <= 0) and metrics.get("pe") and metrics["pe"] > 0 and current_price > 0:
        eps = current_price / metrics["pe"]
    if (not bvps or bvps <= 0) and metrics.get("pb") and metrics["pb"] > 0 and current_price > 0:
        bvps = current_price / metrics["pb"]
        
    # 2. Valuation Models
    pe_pb_bands = calculate_pe_pb_bands(ratio_df, current_eps=eps or 0.0, current_bvps=bvps or 0.0, current_price=eval_price)
    graham = calculate_graham_valuation(eps=eps or 0.0, bvps=bvps or 0.0)
    
    # Estimate FCF for DCF
    fcf_est = (metrics.get("market_cap", 1e12) * 0.06) if metrics.get("market_cap") else 500e9
    shares = metrics.get("shares_outstanding") or 100_000_000
    dcf = calculate_dcf(
        fcf_base=fcf_est,
        growth_rate_y1_5=10.0,
        terminal_growth_rate=3.0,
        discount_rate=12.0,
        shares_outstanding=shares
    )
    div_val = (metrics.get("dividend_yield", 0) * eval_price / 100.0) if metrics.get("dividend_yield") else None
    ddm = calculate_ddm(last_dividend=div_val)
    
    synth_val = synthesize_valuations(
        current_price=eval_price,
        pe_fair=pe_pb_bands.get("pe_fair_value"),
        pb_fair=pe_pb_bands.get("pb_fair_value"),
        graham_number=graham.get("graham_number"),
        revised_graham=graham.get("revised_graham"),
        dcf_fair=dcf.get("fair_value_per_share") if dcf else None,
        ddm_fair=ddm.get("fair_value") if ddm else None
    )
    
    # 3. Financial Health & Scoring
    piotroski = calculate_piotroski_f_score(ratio_df)
    altman = calculate_altman_z_score(ratio_df)
    dupont = calculate_dupont_analysis(ratio_df)
    
    # 4. Technical Analysis & Trade Setup
    tech_signals = calculate_technical_signals(price_df) or {}
    trade_setup = calculate_trade_setup(price_df) or {}
    pivots = calculate_pivot_points(price_df) or {}
    
    rsi_val = 50.0
    macd_val = 0.0
    for s in tech_signals.get("signals", []):
        if s.get("name") == "RSI (14)":
            try:
                rsi_val = float(s.get("value", 50.0))
            except (ValueError, TypeError):
                pass
        elif s.get("name") == "MACD (12,26,9)":
            try:
                val_clean = str(s.get("value", 0)).replace(',', '').replace('+', '')
                macd_val = float(val_clean)
            except (ValueError, TypeError):
                pass
    
    # 5. Live Depth & Foreign Flow
    depth_flow = get_symbol_live_depth_and_foreign(symbol) or {}
    
    return {
        "symbol": symbol,
        "company_name": overview.get("organ_name", symbol),
        "sector": overview.get("sector", "N/A"),
        "exchange": overview.get("com_group_code", "HOSE"),
        "current_market_price": current_price,
        "evaluated_price": eval_price,
        "is_custom_eval_price": (target_eval_price is not None and target_eval_price > 0),
        "fundamental_metrics": {
            "pe": metrics.get("pe"),
            "pb": metrics.get("pb"),
            "ps": metrics.get("ps"),
            "roe": metrics.get("roe"),
            "roa": metrics.get("roa"),
            "roic": metrics.get("roic"),
            "dividend_yield": metrics.get("dividend_yield"),
            "debt_to_equity": metrics.get("debt_to_equity"),
            "gross_margin": metrics.get("gross_margin"),
            "net_margin": metrics.get("net_margin"),
            "eps": eps,
            "bvps": bvps,
        },
        "valuation": {
            "blended_fair_value": synth_val.get("blended_fair_value"),
            "margin_of_safety_pct": synth_val.get("margin_of_safety_pct"),
            "status": synth_val.get("status"),
            "pe_median_fair": pe_pb_bands.get("pe_fair_value"),
            "pb_median_fair": pe_pb_bands.get("pb_fair_value"),
            "graham_number": graham.get("graham_number"),
            "revised_graham": graham.get("revised_graham"),
            "dcf_fair": dcf.get("fair_value_per_share") if dcf else None,
            "ddm_fair": ddm.get("fair_value") if ddm else None,
        },
        "financial_health": {
            "piotroski_score": piotroski.get("total_score") if piotroski else None,
            "piotroski_max": piotroski.get("max_score", 9) if piotroski else 9,
            "health_status": piotroski.get("health_status", "N/A") if piotroski else "N/A",
            "altman_z_score": altman.get("z_score") if altman else None,
            "altman_zone": altman.get("zone", "N/A") if altman else "N/A",
            "dupont_latest_roe": dupont[0].get("roe") if (dupont and len(dupont) > 0) else None,
        },
        "technicals": {
            "score": tech_signals.get("score", 50),
            "consensus": tech_signals.get("consensus", "THEO DÕI (NEUTRAL)"),
            "buy_count": tech_signals.get("buy_count", 0),
            "neutral_count": tech_signals.get("neutral_count", 0),
            "sell_count": tech_signals.get("sell_count", 0),
            "rsi": rsi_val,
            "macd": macd_val,
            "trade_setup": trade_setup,
            "pivots": pivots.get("classic", {}) if pivots else {},
        },
        "live_flow": {
            "foreign_net_vol": depth_flow.get("foreign_flow", {}).get("net_volume", 0) if depth_flow.get("foreign_flow") else 0,
            "foreign_net_val_bil": depth_flow.get("foreign_flow", {}).get("net_value_billion", 0) if depth_flow.get("foreign_flow") else 0,
            "bid_total_vol": depth_flow.get("bid_total_vol", 0),
            "ask_total_vol": depth_flow.get("ask_total_vol", 0)
        }
    }


def evaluate_stock_quantitative(dossier: Dict[str, Any], user_prompt: str = "") -> str:
    """
    Built-in Deterministic Financial Advisory Engine.
    Produces comprehensive, multi-pillar evaluation reports in Vietnamese.
    Fully null-safe across all metrics and indicators.
    """
    if not dossier or not isinstance(dossier, dict):
        return "⚠️ Không thể tải dữ liệu định lượng cho mã cổ phiếu này."

    sym = dossier.get("symbol") or "N/A"
    name = dossier.get("company_name") or sym
    sector = dossier.get("sector") or "N/A"
    eval_p = dossier.get("evaluated_price")
    is_custom = dossier.get("is_custom_eval_price", False)
    
    val = dossier.get("valuation") or {}
    fair_val = val.get("blended_fair_value")
    margin_pct = val.get("margin_of_safety_pct")
    val_status = val.get("status") or "N/A"
    
    health = dossier.get("financial_health") or {}
    f_score = health.get("piotroski_score")
    z_score = health.get("altman_z_score")
    z_zone = health.get("altman_zone") or "N/A"
    
    tech = dossier.get("technicals") or {}
    consensus = tech.get("consensus") or "THEO DÕI (NEUTRAL)"
    tech_score = tech.get("score") if tech.get("score") is not None else 50
    setup = tech.get("trade_setup") or {}
    pivots = tech.get("pivots") or {}
    
    fund = dossier.get("fundamental_metrics") or {}
    pe = fund.get("pe")
    pb = fund.get("pb")
    roe = fund.get("roe")
    dy = fund.get("dividend_yield")
    de = fund.get("debt_to_equity")
    
    val_points = 50
    if margin_pct is not None and isinstance(margin_pct, (int, float)):
        if margin_pct >= 25.0:
            val_points = 95
        elif margin_pct >= 10.0:
            val_points = 80
        elif margin_pct >= -10.0:
            val_points = 55
        elif margin_pct >= -25.0:
            val_points = 35
        else:
            val_points = 15

    health_points = 50
    if f_score is not None and isinstance(f_score, (int, float)):
        health_points = (f_score / 9.0) * 100
        
    tech_points = tech_score if (tech_score is not None and isinstance(tech_score, (int, float))) else 50
    composite_score = (val_points * 0.40) + (health_points * 0.25) + (tech_points * 0.35)
    
    if composite_score >= 75:
        overall_action = "🟢 KHUYẾN NGHỊ: MUA / TÍCH LŨY MẠNH (STRONG BUY)"
        action_summary = "Cổ phiếu hội tụ đầy đủ yếu tố: Định giá rẻ có biên an toàn cao, sức khỏe tài chính vững chắc và xu hướng kỹ thuật thuận lợi."
    elif composite_score >= 60:
        overall_action = "🟢 KHUYẾN NGHỊ: MUA TÍCH LŨY TỪNG PHẦN (ACCUMULATE)"
        action_summary = "Cổ phiếu có nền tảng cơ bản tốt, định giá hợp lý. Có thể giải ngân thăm dò theo từng nhịp điều chỉnh."
    elif composite_score >= 45:
        overall_action = "🟡 KHUYẾN NGHỊ: THEO DÕI & NẮM GIỮ (HOLD / NEUTRAL)"
        action_summary = "Cổ phiếu đang ở vùng cân bằng hoặc tín hiệu kỹ thuật/định giá chưa có sự bứt phá rõ rệt. Nên quan sát thêm."
    elif composite_score >= 30:
        overall_action = "🟠 KHUYẾN NGHỊ: HẠ TỶ TRỌNG / CHỐT LỜI DẦN (REDUCE)"
        action_summary = "Định giá đã tiến sát hoặc vượt giá trị thực, hoặc chỉ báo kỹ thuật xuất hiện tín hiệu suy yếu/quá mua."
    else:
        overall_action = "🔴 KHUYẾN NGHỊ: BÁN / ĐỨNG NGOÀI (SELL / AVOID)"
        action_summary = "Cổ phiếu chịu áp lực bán lớn, định giá quá đắt so với nội tại hoặc sức khỏe tài chính có dấu hiệu rủi ro cao."

    # Format helpers
    def _fmt_vnd(v):
        if v is not None and isinstance(v, (int, float)) and v > 0:
            return f"{v:,.0f} VND"
        return "N/A"

    pe_str = f"{pe:.1f}x" if (pe is not None and isinstance(pe, (int, float))) else "N/A"
    pb_str = f"{pb:.2f}x" if (pb is not None and isinstance(pb, (int, float))) else "N/A"
    roe_str = f"{roe:.1f}%" if (roe is not None and isinstance(roe, (int, float))) else "N/A"
    dy_str = f"{dy:.1f}%" if (dy is not None and isinstance(dy, (int, float))) else "0.0%"
    de_str = f"{de:.2f}x" if (de is not None and isinstance(de, (int, float))) else "N/A"
    
    fair_str = _fmt_vnd(fair_val)
    margin_str = f"{margin_pct:+.1f}%" if (margin_pct is not None and isinstance(margin_pct, (int, float))) else "N/A"
    f_str = f"{f_score}/9" if f_score is not None else "N/A"
    z_str = f"{z_score:.2f}" if (z_score is not None and isinstance(z_score, (int, float))) else (str(z_score) if z_score else "N/A")
    tech_str = f"{tech_score:.0f}/100" if isinstance(tech_score, (int, float)) else "50/100"
    
    eval_p_str = _fmt_vnd(eval_p)
    eval_note = "*(Giá tham chiếu nhập vào)*" if is_custom else "*(Giá thị trường)*"
    safe_de_note = "An toàn, đòn bẩy thấp" if (de is not None and isinstance(de, (int, float)) and de < 1.0) else "Đòn bẩy trung bình / cao"

    rsi_v = tech.get("rsi")
    rsi_str = f"{rsi_v:.1f}" if (rsi_v is not None and isinstance(rsi_v, (int, float))) else "50.0"
    
    piv_r1 = _fmt_vnd(pivots.get("R1"))
    piv_r2 = _fmt_vnd(pivots.get("R2"))
    piv_pp = _fmt_vnd(pivots.get("PP"))
    piv_s1 = _fmt_vnd(pivots.get("S1"))
    piv_s2 = _fmt_vnd(pivots.get("S2"))

    pe_fair_line = _fmt_vnd(val.get('pe_median_fair'))
    pb_fair_line = _fmt_vnd(val.get('pb_median_fair'))
    graham_line = _fmt_vnd(val.get('graham_number'))
    dcf_line = _fmt_vnd(val.get('dcf_fair'))
    ddm_line = _fmt_vnd(val.get('ddm_fair'))
    
    report = f"""### 🤖 BÁO CÁO ĐÁNH GIÁ ĐẦU TƯ AI - MÃ {sym}
**Doanh nghiệp:** {name} | **Ngành:** {sector}  
**Mức giá đánh giá:** `{eval_p_str}` {eval_note}

---

#### 🏆 KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG
### **{overall_action}**
> **Điểm Tổng Hợp AI:** `{composite_score:.1f} / 100`  
> **Nhận định:** {action_summary}

---

#### 1. 💰 ĐỊNH GIÁ & BIÊN AN TOÀN (VALUATION)
- **Giá trị Hợp lý Tổng hợp (Synthesized Fair Value):** `{fair_str}`
- **Biên an toàn (Margin of Safety):** `{margin_str}` ({val_status})
- **Chi tiết các mô hình định giá:**
  - 📐 **P/E Median Band (5 năm):** `{pe_fair_line}` (P/E hiện tại: `{pe_str}`)
  - 📚 **P/B Median Band (5 năm):** `{pb_fair_line}` (P/B hiện tại: `{pb_str}`)
  - 🏛️ **Công thức Benjamin Graham:** `{graham_line}`
  - ⚡ **Chiết khấu Dòng tiền (DCF):** `{dcf_line}`
  - 💵 **Chiết khấu Cổ tức (DDM):** `{ddm_line}` (Tỷ suất cổ tức: `{dy_str}`)

---

#### 2. 🏥 SỨC KHỎE TÀI CHÍNH & QUẢN TRỊ RỦI RO (FINANCIAL QUALITY)
- **Điểm Piotroski F-Score:** `{f_str}` ➔ **{health.get('health_status', 'N/A')}**
- **Chỉ số Altman Z''-Score:** `{z_str}` ➔ **{z_zone}** *(Rủi ro kiệt quệ tài chính thấp nếu Z > 2.6)*
- **Khả năng sinh lời & Đòn bẩy:**
  - **ROE:** `{roe_str}` | **Đòn bẩy Nợ/Vốn CSH (D/E):** `{de_str}`
  - Cấu trúc tài chính: {safe_de_note}.

---

#### 3. 📈 PHÂN TÍCH KỸ THUẬT & DÒNG TIỀN (TECHNICAL & FLOW)
- **Đồng thuận Kỹ thuật:** **{consensus}** (Điểm kỹ thuật: `{tech_str}`)
- **Chỉ báo RSI (14):** `{rsi_str}`
- **Điểm xoay Pivot (Classic):**
  - Kháng cự R1: `{piv_r1}` | Kháng cự R2: `{piv_r2}`
  - Trục xoay PP: `{piv_pp}`
  - Hỗ trợ S1: `{piv_s1}` | Hỗ trợ S2: `{piv_s2}`

---

#### 4. 📋 KẾ HOẠCH GIAO DỊCH ĐỀ XUẤT (TRADE SETUP)
"""
    if setup and isinstance(setup, dict) and setup.get('entry_range') and setup.get('target_1') and setup.get('stop_loss'):
        try:
            e0, e1 = setup['entry_range']
            t1 = setup.get('target_1', 0)
            t1_pct = setup.get('target_1_pct', 0)
            t2 = setup.get('target_2', 0)
            t2_pct = setup.get('target_2_pct', 0)
            sl = setup.get('stop_loss', 0)
            sl_pct = setup.get('stop_loss_pct', 0)
            rr1 = setup.get('risk_reward_t1', 1.5)
            rr2 = setup.get('risk_reward_t2', 2.5)
            
            report += f"""- **Vùng Mua Khuyến Nghị:** `{e0:,.0f} - {e1:,.0f} VND`
- **Mục Tiêu 1 (TP1):** `{t1:,.0f} VND` ({t1_pct:+.1f}%) | R:R = **1 : {rr1}**
- **Mục Tiêu 2 (TP2):** `{t2:,.0f} VND` ({t2_pct:+.1f}%) | R:R = **1 : {rr2}**
- **Cắt Lỗ (Stop-Loss):** `{sl:,.0f} VND` ({sl_pct:.1f}%)
- **Quản lý Tỷ trọng đề xuất:** Tối đa `15% - 25%` tổng NAV danh mục đối với vị thế này.
"""
        except Exception:
            report += "- *Chưa có thiết lập giao dịch rõ ràng do biên độ nén giá hoặc thiếu dữ liệu nến gần nhất.*\n"
    else:
        report += "- *Chưa có thiết lập giao dịch rõ ràng do biên độ nén giá hoặc thiếu dữ liệu nến gần nhất.*\n"
        
    report += "\n*Lưu ý: Báo cáo phân tích định lượng AI mang tính chất tham khảo, nhà đầu tư nên cân nhắc quản trị rủi ro và khẩu vị cá nhân trước khi giải ngân.*"
    return report



def call_gemini_api(api_key: str, model: str, prompt: str, system_instruction: str) -> str:
    """Invokes Google Gemini via REST API."""
    model_name = model if model else "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_instruction}\n\nUser Question:\n{prompt}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048
        }
    }
    
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts:
            return parts[0].get("text", "")
    return "Không nhận được phản hồi từ Gemini API."


def call_openai_compatible_api(api_url: str, api_key: str, model: str, messages: list) -> str:
    """Invokes OpenAI, DeepSeek, or OpenAI-compatible endpoints."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048
    }
    resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_ollama_api(base_url: str, model: str, prompt: str, system_prompt: str) -> str:
    """Invokes local Ollama instance."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model or "llama3",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    resp = requests.post(url, json=payload, timeout=40)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def evaluate_stock_ai(
    dossier: Dict[str, Any],
    user_prompt: str,
    provider: str = "builtin",
    api_key: str = "",
    model_name: str = "",
    custom_endpoint: str = ""
) -> str:
    """
    Main evaluation dispatcher.
    Uses LLM (Gemini, OpenAI, DeepSeek, Ollama) if configured, or falls back seamlessly to Built-in Quant Engine.
    """
    if provider == "builtin" or not api_key:
        return evaluate_stock_quantitative(dossier, user_prompt)
        
    system_instruction = (
        "Bạn là Chuyên gia Tư Vấn Đầu Tư Chứng Khoán Việt Nam & Chuyên Gia Định Giá Cao Cấp (Senior Equity Research Analyst & Fund Manager). "
        "Dưới đây là BỘ HỒ SƠ TÀI CHÍNH & ĐỊNH GIÁ ĐỊNH LƯỢNG CHÍNH XÁC vừa được trích xuất từ dữ liệu thời gian thực của thị trường chứng khoán Việt Nam (HOSE/HNX/UPCOM).\n"
        "NHIỆM VỤ CỦA BẠN:\n"
        "1. Trả lời câu hỏi của người dùng một cách trực diện, chuyên nghiệp, logic và sâu sắc.\n"
        "2. Đưa ra đánh giá rõ ràng: Ở mức giá này CÓ ĐÁNG ĐỂ MUA, TÍCH LŨY, NẮM GIỮ hay CẦN BÁN/HẠ TỶ TRỌNG?\n"
        "3. Dẫn chứng các số liệu cụ thể từ hồ sơ (Định giá P/E band, Graham, DCF, F-Score, Altman Z, Hỗ trợ/Kháng cự Pivot, Điểm vào/cắt lỗ/chốt lời).\n"
        "4. Đưa ra tỷ trọng giải ngân khuyến nghị (% NAV) và cảnh báo rủi ro cụ thể.\n"
        "5. Dùng tiếng Việt chuẩn mực ngành tài chính chứng khoán, định dạng Markdown đẹp, rõ ràng."
    )
    
    dossier_json = json.dumps(dossier, ensure_ascii=False, indent=2)
    full_prompt = (
        f"HỒ SƠ ĐỊNH LƯỢNG CỔ PHIẾU ({dossier.get('symbol')}):\n"
        f"```json\n{dossier_json}\n```\n\n"
        f"CÂU HỎI / YÊU CẦU CỦA NHÀ ĐẦU TƯ:\n{user_prompt}"
    )
    
    try:
        if provider == "gemini":
            m = model_name or "gemini-1.5-flash"
            return call_gemini_api(api_key, m, full_prompt, system_instruction)
            
        elif provider == "openai":
            m = model_name or "gpt-4o-mini"
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt}
            ]
            return call_openai_compatible_api("https://api.openai.com/v1/chat/completions", api_key, m, messages)
            
        elif provider == "deepseek":
            m = model_name or "deepseek-chat"
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt}
            ]
            return call_openai_compatible_api("https://api.deepseek.com/chat/completions", api_key, m, messages)
            
        elif provider == "ollama":
            base_url = custom_endpoint or "http://localhost:11434"
            m = model_name or "llama3"
            return call_ollama_api(base_url, m, full_prompt, system_instruction)
            
        else:
            return evaluate_stock_quantitative(dossier, user_prompt)
            
    except Exception as e:
        quant_result = evaluate_stock_quantitative(dossier, user_prompt)
        return (
            f"> ⚠️ *Lưu ý kết nối AI API ({provider}): {str(e)}. Hệ thống đã tự động chuyển sang Chế độ Định giá & Phân tích Định lượng AI Cục bộ (Offline Quant Engine).* \n\n"
            + quant_result
        )




def render_ai_chat_interface(
    selected_symbol: str,
    current_price: float = 0.0,
    ai_provider: str = "builtin",
    ai_api_key: str = "",
    ai_model_name: str = "",
    ai_endpoint: str = "",
    key_suffix: str = ""
):
    """
    Renders an interactive conversational AI Box Chat for stock evaluation and analysis.
    """
    import streamlit as st

    st.subheader(f"🤖 Trợ Lý AI Định Giá & Tư Vấn Đầu Tư - {selected_symbol}")
    st.caption("AI tự động tổng hợp Định giá (DCF, Graham, P/E Band), Sức khỏe tài chính (F-Score, Altman Z), Tín hiệu Kỹ thuật & Kế hoạch giải ngân.")

    top_c1, top_c2 = st.columns([4, 1])
    with top_c1:
        st.markdown(f"🧠 **Động cơ AI:** `{ai_provider.upper()}` ({ai_model_name if ai_model_name else 'Quantitative Expert Engine'}) | 🎯 **Mã hiện tại:** `{selected_symbol}`")
    with top_c2:
        if st.button("🗑️ Xóa Lịch Sử Chat", key=f"btn_clear_chat_{key_suffix}", use_container_width=True):
            st.session_state["ai_chat_history"] = []
            st.rerun()

    # Quick Action Suggestion Chips
    st.markdown("##### ⚡ Câu hỏi mẫu nhanh:")
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_query = None
    if qc1.button(f"🎯 Có nên MUA/BÁN {selected_symbol}?", key=f"btn_q1_{key_suffix}", use_container_width=True):
        quick_query = f"Đánh giá toàn diện mã {selected_symbol} ở mức giá hiện tại: có nên MUA, NẮM GIỮ hay BÁN?"
    if qc2.button(f"💰 Định giá & Biên an toàn?", key=f"btn_q2_{key_suffix}", use_container_width=True):
        quick_query = f"Định giá hợp lý của {selected_symbol} (DCF, Graham, P/E Band) là bao nhiêu và Margin of Safety thế nào?"
    if qc3.button(f"🛡️ Sức khỏe tài chính & Nợ?", key=f"btn_q3_{key_suffix}", use_container_width=True):
        quick_query = f"Phân tích sức khỏe tài chính, điểm F-Score, rủi ro nợ vay và phá sản Altman Z của {selected_symbol}."
    if qc4.button(f"📈 Điểm mua & Cắt lỗ SL?", key=f"btn_q4_{key_suffix}", use_container_width=True):
        quick_query = f"Lập kế hoạch giao dịch ngắn hạn cho {selected_symbol}: Vùng mua, Mục tiêu chốt lời TP1/TP2 và Cắt lỗ SL ở đâu?"

    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = [
            {
                "role": "assistant",
                "content": f"Xin chào! Tôi là Trợ Lý Định Giá & Phân Tích Đầu Tư AI cho cổ phiếu **{selected_symbol}**.\n\nBạn có thể hỏi tôi về định giá (DCF, Graham, P/E Band), sức khỏe tài chính (Piotroski, Altman Z), xu hướng kỹ thuật, hoặc gõ bất kỳ mã nào (ví dụ: *'HPG giá 28k có mua được không?'*, *'Định giá SSI'*) để nhận báo cáo khuyến nghị chi tiết!"
            }
        ]

    for msg in st.session_state["ai_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(f"Hỏi AI về {selected_symbol} (e.g. 'HPG 28k mua được không?', 'Định giá SSI')...", key=f"ai_chat_input_{key_suffix}")
    prompt_to_process = quick_query or user_input

    if prompt_to_process:
        st.session_state["ai_chat_history"].append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("AI đang tổng hợp dữ liệu báo cáo tài chính, định giá và tín hiệu kỹ thuật..."):
                target_symbol = extract_stock_symbol_from_prompt(prompt_to_process, default_symbol=selected_symbol)
                target_price = extract_target_price_from_prompt(prompt_to_process)

                dossier = build_stock_dossier(target_symbol, target_eval_price=target_price)
                ai_response = evaluate_stock_ai(
                    dossier=dossier,
                    user_prompt=prompt_to_process,
                    provider=ai_provider,
                    api_key=ai_api_key,
                    model_name=ai_model_name,
                    custom_endpoint=ai_endpoint
                )
                st.markdown(ai_response)
                st.session_state["ai_chat_history"].append({"role": "assistant", "content": ai_response})

