import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
from agno.tools import tool
from data.yfinance_utils import (
    safe_div, 
    last_before_or_equal, 
    last_val, 
    calculate_yoy_growth
)


@tool
def yf_fundamental_snapshot(ticker: str, as_of: Optional[str] = None) -> dict:
    """
    Coleta dados fundamentais completos via yfinance.
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA)
        as_of: Data de referência no formato YYYY-MM-DD (evita look-ahead)
    
    Returns:
        Dict com todos os campos do FundamentalSnapshot
    """
    as_of_dt = pd.to_datetime(as_of).to_pydatetime() if as_of else datetime.utcnow()
    tk = yf.Ticker(ticker)
    
    snap: Dict[str, Any] = {
        "ticker": ticker,
        "as_of": as_of or datetime.utcnow().strftime("%Y-%m-%d"),
        "evidence": [],
    }
    
    # ============ PREÇO E MARKET CAP ============
    # CORREÇÃO CRÍTICA: .info é mais confiável que fast_info
    try:
        info = tk.info
        snap["price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        snap["market_cap"] = info.get("marketCap")
        snap["shares_out"] = info.get("sharesOutstanding")
        
        # Valuation direto do info (quando disponível)
        snap["pe"] = info.get("trailingPE") or info.get("forwardPE")
        snap["pb"] = info.get("priceToBook")
        
        snap["evidence"].append("info(price, market_cap, shares, pe, pb)")
    except Exception as e:
        snap["evidence"].append(f"info_error:{str(e)[:50]}")
        return snap  # early return - sem dados básicos não dá pra continuar
    
    # ============ STATEMENTS ============
    try:
        inc = last_before_or_equal(tk.income_stmt, as_of_dt)
    except Exception:
        inc = None
    
    try:
        bs = last_before_or_equal(tk.balance_sheet, as_of_dt)
    except Exception:
        bs = None
    
    # ============ INCOME STATEMENT ============
    revenue = last_val(inc, "Total Revenue")
    gross_profit = last_val(inc, "Gross Profit")
    operating_income = last_val(inc, "Operating Income")
    net_income = last_val(inc, "Net Income")
    
    # ============ BALANCE SHEET ============
    total_assets = last_val(bs, "Total Assets")
    total_equity = last_val(bs, "Stockholders Equity")
    total_debt = last_val(bs, "Total Debt")
    current_assets = last_val(bs, "Current Assets")
    current_liab = last_val(bs, "Current Liabilities")
    
    # ============ MÉTRICAS CALCULADAS ============
    
    # Margens
    snap["gross_margin"] = safe_div(gross_profit, revenue)
    snap["op_margin"] = safe_div(operating_income, revenue)
    snap["net_margin"] = safe_div(net_income, revenue)
    
    # Rentabilidade
    snap["roe"] = safe_div(net_income, total_equity)
    snap["roa"] = safe_div(net_income, total_assets)
    
    # Risco/Alavancagem
    snap["debt_to_equity"] = safe_div(total_debt, total_equity)
    snap["current_ratio"] = safe_div(current_assets, current_liab)
    snap["total_debt"] = total_debt
    snap["total_equity"] = total_equity
    
    # P/S (se não veio do info)
    if snap.get("ps") is None:
        revenue_per_share = safe_div(revenue, snap["shares_out"])
        snap["ps"] = safe_div(snap["price"], revenue_per_share)
    
    # ============ CRESCIMENTO YoY ============
    snap["revenue_growth_yoy"] = calculate_yoy_growth(inc, "Total Revenue")
    snap["net_income_growth_yoy"] = calculate_yoy_growth(inc, "Net Income")
    
    # ============ EVIDENCE ============
    if inc is not None:
        snap["evidence"].append("income_stmt")
    if bs is not None:
        snap["evidence"].append("balance_sheet")
    
    return snap


@tool
def fundamental_score(snapshot: dict) -> dict:
    """
    Calcula score fundamental 0-100 com subscores.
    
    Args:
        snapshot: Dict do yf_fundamental_snapshot
    
    Returns:
        {
            "score": float ou None,
            "confidence": float (0-1),
            "subscores": {"valuation": float, "quality": float, "risk": float},
            "warning": str (se houver)
        }
    """
    s = snapshot
    
    # ============ CONFIANÇA ============
    total_fields = 12
    available = sum(
        1 for k in [
            "pe", "pb", "ps", 
            "gross_margin", "op_margin", "net_margin",
            "roe", "roa", 
            "debt_to_equity", "current_ratio",
            "revenue_growth_yoy", "net_income_growth_yoy"
        ]
        if s.get(k) is not None
    )
    confidence = available / total_fields
    
    # Se < 50% dos dados, score é inválido
    if confidence < 0.5:
        return {
            "score": None,
            "confidence": round(confidence, 2),
            "subscores": {"valuation": None, "quality": None, "risk": None},
            "warning": f"Dados insuficientes ({available}/{total_fields} campos)"
        }
    
    # ============ FUNÇÕES DE NORMALIZAÇÃO ============
    
    def inv(x):
        """Para valuation: menor é melhor (P/E baixo é bom)"""
        if x is None or x <= 0 or x > 200:
            return 0.0
        return min(1.0, 30.0 / float(x))  # P/E <= 30 score máximo
    
    def clip01(x):
        """Clip entre 0 e 1"""
        if x is None or np.isnan(x):
            return 0.0
        return float(np.clip(x, 0.0, 1.0))
    
    def de_penalty(x):
        """Debt/Equity: 0.0 → score 1.0, >= 2.0 → score 0.0"""
        if x is None or x < 0:
            return 0.5  # neutro se não tiver dado
        return float(np.clip(1.0 - (x / 2.0), 0.0, 1.0))
    
    # ============ VALUATION (40% do score) ============
    v_pe = inv(s.get("pe"))
    v_pb = inv(s.get("pb"))
    v_ps = inv(s.get("ps"))
    
    valuation = 0.40 * (0.5 * v_pe + 0.3 * v_pb + 0.2 * v_ps)
    
    # ============ QUALITY (40% do score) ============
    q_gm = clip01(s.get("gross_margin"))
    q_om = clip01(s.get("op_margin"))
    q_nm = clip01(s.get("net_margin"))
    q_roe = clip01(s.get("roe"))
    q_roa = clip01(s.get("roa"))
    
    quality = 0.40 * (
        0.25 * q_gm + 
        0.25 * q_om + 
        0.20 * q_nm + 
        0.20 * q_roe + 
        0.10 * q_roa
    )
    
    # ============ RISK (20% do score) ============
    r_de = de_penalty(s.get("debt_to_equity"))
    r_liq = clip01(min(1.0, (s.get("current_ratio") or 0) / 2.0))
    
    risk = 0.20 * (0.70 * r_de + 0.30 * r_liq)
    
    # ============ SCORE FINAL ============
    score = (valuation + quality + risk) * 100.0
    
    return {
        "score": round(float(score), 2),
        "confidence": round(confidence, 2),
        "subscores": {
            "valuation": round(float(valuation * 100), 2),
            "quality": round(float(quality * 100), 2),
            "risk": round(float(risk * 100), 2),
        }
    }
