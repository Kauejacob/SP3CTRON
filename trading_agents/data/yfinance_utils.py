# data/yfinance_utils.py
"""
Utilitários para coleta de dados via yfinance.
"""

import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Converte para float de forma segura.
    Trata None, '', 'None', NaN, Infinity.
    """
    if value is None or value == 'None' or value == '':
        return default
    try:
        result = float(value)
        # Verifica NaN ou Infinity
        if not (result == result and abs(result) != float('inf')):
            return default
        return result
    except (ValueError, TypeError):
        return default


def get_fundamental_snapshot(ticker: str, as_of: Optional[str] = None) -> Dict[str, Any]:
    """
    Coleta snapshot de dados fundamentalistas de um ticker.
    ROBUSTO: Trata valores None/ausentes.
    
    Args:
        ticker: Ticker da ação (ex: PETR4.SA)
        as_of: Data de referência YYYY-MM-DD (None = hoje)
    
    Returns:
        Dict com dados fundamentalistas
    """
    if as_of is None:
        as_of = datetime.now().strftime("%Y-%m-%d")
    
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        
        # ============ COLETA COM TRATAMENTO ============
        
        # Preço
        price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
        
        # Market Cap
        market_cap = safe_float(info.get('marketCap'))
        
        # Shares Outstanding
        shares = safe_float(info.get('sharesOutstanding'))
        
        # Múltiplos de Valuation
        pe = safe_float(info.get('trailingPE') or info.get('forwardPE'))
        pb = safe_float(info.get('priceToBook'))
        ps = safe_float(info.get('priceToSalesTrailing12Months'))
        
        # Margens
        gross_margin = safe_float(info.get('grossMargins'))
        op_margin = safe_float(info.get('operatingMargins'))
        net_margin = safe_float(info.get('profitMargins'))
        
        # Retornos
        roe = safe_float(info.get('returnOnEquity'))
        roa = safe_float(info.get('returnOnAssets'))
        
        # Crescimento
        revenue_growth = safe_float(info.get('revenueGrowth'))
        earnings_growth = safe_float(info.get('earningsGrowth'))
        
        # Dívida e Liquidez
        total_debt = safe_float(info.get('totalDebt'))
        total_equity = safe_float(info.get('totalStockholderEquity'))
        
        # Debt to Equity
        if total_debt and total_equity and total_equity > 0:
            debt_to_equity = total_debt / total_equity
        else:
            debt_to_equity = None
        
        # Current Ratio
        current_ratio = safe_float(info.get('currentRatio'))
        
        # Dividend Yield
        dividend_yield = safe_float(info.get('dividendYield'))
        
        # ============ EVIDÊNCIAS ============
        evidence = []
        
        if price:
            evidence.append(f"Preço atual: R$ {price:.2f}")
        if market_cap:
            evidence.append(f"Market Cap: R$ {market_cap/1e9:.2f}B")
        if pe:
            evidence.append(f"P/E: {pe:.2f}")
        if net_margin:
            evidence.append(f"Margem Líquida: {net_margin*100:.1f}%")
        if roe:
            evidence.append(f"ROE: {roe*100:.1f}%")
        if debt_to_equity:
            evidence.append(f"D/E: {debt_to_equity:.2f}")
        
        if not evidence:
            evidence.append("Dados fundamentalistas limitados para esta empresa")
        
        # ============ SNAPSHOT ============
        snapshot = {
            'ticker': ticker,
            'as_of': as_of,
            'price': price,
            'market_cap': market_cap,
            'shares_out': shares,
            'pe': pe,
            'pb': pb,
            'ps': ps,
            'gross_margin': gross_margin,
            'op_margin': op_margin,
            'net_margin': net_margin,
            'roe': roe,
            'roa': roa,
            'revenue_growth_yoy': revenue_growth,
            'net_income_growth_yoy': earnings_growth,
            'total_debt': total_debt,
            'total_equity': total_equity,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'dividend_yield': dividend_yield,
            'evidence': evidence
        }
        
        return snapshot
        
    except Exception as e:
        # Fallback: retorna snapshot mínimo
        return {
            'ticker': ticker,
            'as_of': as_of,
            'price': None,
            'market_cap': None,
            'shares_out': None,
            'pe': None,
            'pb': None,
            'ps': None,
            'gross_margin': None,
            'op_margin': None,
            'net_margin': None,
            'roe': None,
            'roa': None,
            'revenue_growth_yoy': None,
            'net_income_growth_yoy': None,
            'total_debt': None,
            'total_equity': None,
            'debt_to_equity': None,
            'current_ratio': None,
            'dividend_yield': None,
            'evidence': [f"Erro ao coletar dados: {str(e)[:100]}"]
        }


# Alias para compatibilidade com código antigo
get_snapshot = get_fundamental_snapshot


# ============ TESTE ============

if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "PETR4.SA"
    
    print(f"🧪 Testando coleta de dados para {ticker}")
    print("="*70)
    
    snapshot = get_fundamental_snapshot(ticker)
    
    print(f"\n📊 Dados coletados:")
    for key, value in snapshot.items():
        if key == 'evidence':
            continue
        if value is not None:
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        else:
            print(f"   {key}: N/A")
    
    print(f"\n📝 Evidências:")
    for ev in snapshot['evidence']:
        print(f"   • {ev}")