"""
Cálculo de métricas de performance e benchmark (CDI).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import requests
from io import StringIO


def get_cdi_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Baixa dados do CDI do BCB via API.
    
    Args:
        start_date: Data início YYYY-MM-DD
        end_date: Data fim YYYY-MM-DD
    
    Returns:
        DataFrame com índice date e coluna 'cdi_daily' (taxa diária)
    """
    print(f"\n📊 Baixando dados do CDI (BCB)...")
    print(f"   Período: {start_date} a {end_date}")
    
    try:
        # API do BCB - série 12 (CDI)
        # URL: https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Formata datas para API (dd/mm/yyyy)
        start_str = start_dt.strftime('%d/%m/%Y')
        end_str = end_dt.strftime('%d/%m/%Y')
        
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={start_str}&dataFinal={end_str}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("   ⚠️ API retornou vazio, usando CDI fixo de 0.035% ao dia")
            return _create_dummy_cdi(start_date, end_date)
        
        # Converte para DataFrame
        df = pd.DataFrame(data)
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
        df['valor'] = df['valor'].astype(float)
        
        # Taxa diária (BCB já fornece % ao dia)
        df['cdi_daily'] = df['valor'] / 100  # Converte % para decimal
        
        df = df.set_index('data')[['cdi_daily']]
        df.index.name = 'date'
        
        print(f"   ✅ CDI carregado: {len(df)} dias")
        print(f"      Taxa média: {df['cdi_daily'].mean()*100:.4f}% ao dia")
        
        return df
        
    except Exception as e:
        print(f"   ⚠️ Erro ao baixar CDI: {e}")
        print(f"   Usando CDI fixo de 0.035% ao dia (~13.5% ao ano)")
        return _create_dummy_cdi(start_date, end_date)


def _create_dummy_cdi(start_date: str, end_date: str, annual_rate: float = 0.135) -> pd.DataFrame:
    """
    Cria série de CDI dummy com taxa fixa.
    
    Args:
        start_date: Data início
        end_date: Data fim
        annual_rate: Taxa anual (ex: 0.135 = 13.5% ao ano)
    
    Returns:
        DataFrame com CDI diário
    """
    # Gera datas (dias úteis)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # Taxa diária equivalente
    daily_rate = (1 + annual_rate) ** (1/252) - 1
    
    df = pd.DataFrame({
        'cdi_daily': daily_rate
    }, index=dates)
    
    df.index.name = 'date'
    
    return df


def align_cdi_to_portfolio(portfolio_dates: pd.DatetimeIndex, cdi_df: pd.DataFrame) -> pd.Series:
    """
    Alinha CDI com as datas do portfólio.
    
    Args:
        portfolio_dates: Datas do portfólio
        cdi_df: DataFrame com CDI
    
    Returns:
        Series com CDI alinhado às datas do portfólio
    """
    # Reindex para datas do portfólio
    cdi_aligned = cdi_df['cdi_daily'].reindex(portfolio_dates)
    
    # Forward fill para preencher feriados
    cdi_aligned = cdi_aligned.fillna(method='ffill')
    
    # Se ainda tiver NaN no início, preenche com média
    if cdi_aligned.isna().any():
        mean_cdi = cdi_df['cdi_daily'].mean()
        cdi_aligned = cdi_aligned.fillna(mean_cdi)
    
    return cdi_aligned


def calculate_metrics(
    portfolio_history: pd.DataFrame,
    cdi_series: Optional[pd.Series] = None,
    risk_free_rate: float = 0.135  # 13.5% ao ano
) -> Dict:
    """
    Calcula métricas de performance do portfólio.
    
    Args:
        portfolio_history: DataFrame com histórico (colunas: total_value, returns)
        cdi_series: Series com retornos diários do CDI
        risk_free_rate: Taxa livre de risco anual (se não tiver CDI)
    
    Returns:
        Dict com métricas
    """
    if len(portfolio_history) < 2:
        return {}
    
    # Retornos
    returns = portfolio_history['returns'] / 100  # Converte % para decimal
    
    # Remove primeiro dia (retorno = 0)
    returns = returns[1:]
    
    if len(returns) == 0:
        return {}
    
    # ============ RETORNO TOTAL ============
    initial_value = portfolio_history['total_value'].iloc[0]
    final_value = portfolio_history['total_value'].iloc[-1]
    total_return = ((final_value / initial_value) - 1) * 100
    
    # ============ RETORNO ANUALIZADO ============
    num_days = len(portfolio_history)
    years = num_days / 252  # Trading days
    
    if years > 0:
        annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100
    else:
        annualized_return = 0.0
    
    # ============ VOLATILIDADE ============
    volatility_daily = returns.std()
    volatility_annual = volatility_daily * np.sqrt(252) * 100
    
    # ============ SHARPE RATIO ============
    if cdi_series is not None:
        # Usa CDI real
        cdi_returns = cdi_series[1:]  # Alinha com returns
        excess_returns = returns - cdi_returns
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    else:
        # Usa taxa fixa
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = returns - daily_rf
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    
    # ============ MAX DRAWDOWN ============
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # ============ CALMAR RATIO ============
    calmar = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0.0
    
    # ============ WIN RATE ============
    positive_days = (returns > 0).sum()
    win_rate = (positive_days / len(returns)) * 100
    
    # ============ BEST/WORST DAY ============
    best_day = returns.max() * 100
    worst_day = returns.min() * 100
    
    # ============ CDI COMPARISON ============
    if cdi_series is not None:
        cdi_cumulative = (1 + cdi_series[1:]).cumprod()
        cdi_total_return = (cdi_cumulative.iloc[-1] - 1) * 100
        outperformance = total_return - cdi_total_return
    else:
        cdi_total_return = ((1 + risk_free_rate) ** years - 1) * 100
        outperformance = total_return - cdi_total_return
    
    return {
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'volatility_annual_pct': volatility_annual,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate_pct': win_rate,
        'best_day_pct': best_day,
        'worst_day_pct': worst_day,
        'cdi_total_return_pct': cdi_total_return,
        'outperformance_pct': outperformance,
        'num_days': num_days,
        'num_years': years,
    }


def print_metrics(metrics: Dict):
    """
    Imprime métricas formatadas.
    """
    print("\n" + "="*70)
    print("📊 MÉTRICAS DE PERFORMANCE")
    print("="*70)
    
    print(f"\n📈 Retornos:")
    print(f"   Total: {metrics['total_return_pct']:.2f}%")
    print(f"   Anualizado: {metrics['annualized_return_pct']:.2f}%")
    print(f"   CDI (período): {metrics['cdi_total_return_pct']:.2f}%")
    print(f"   Outperformance: {metrics['outperformance_pct']:+.2f}%")
    
    print(f"\n📊 Risco:")
    print(f"   Volatilidade anual: {metrics['volatility_annual_pct']:.2f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    
    print(f"\n🎯 Ratios:")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"   Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    print(f"\n📆 Estatísticas:")
    print(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"   Melhor dia: {metrics['best_day_pct']:+.2f}%")
    print(f"   Pior dia: {metrics['worst_day_pct']:.2f}%")
    print(f"   Período: {metrics['num_years']:.2f} anos ({metrics['num_days']} dias)")
    
    print("\n" + "="*70)


# ============ TESTE ============

if __name__ == "__main__":
    print("🧪 TESTE DO MÓDULO METRICS")
    print("="*70)
    
    # Testa download CDI
    cdi = get_cdi_data("2023-01-01", "2024-01-01")
    print(f"\n📊 CDI (primeiros 5 dias):")
    print(cdi.head())
    
    # Simula histórico de portfólio
    print(f"\n🧪 Simulando histórico de portfólio...")
    
    dates = pd.bdate_range(start="2023-01-01", end="2024-01-01")
    
    # Simula retornos aleatórios
    np.random.seed(42)
    returns = np.random.normal(0.05, 1.0, len(dates))  # 0.05% média,

if __name__ == "__main__":
    print("🧪 TESTE DO MÓDULO METRICS")
    print("="*70)
    
    # Testa download CDI
    cdi = get_cdi_data("2023-01-01", "2024-01-01")
    print(f"\n📊 CDI (primeiros 5 dias):")
    print(cdi.head())
    
    # Simula histórico de portfólio
    print(f"\n🧪 Simulando histórico de portfólio...")
    
    dates = pd.bdate_range(start="2023-01-01", end="2024-01-01")
    
    # Simula retornos aleatórios
    np.random.seed(42)
    returns = np.random.normal(0.05, 1.0, len(dates))  # 0.05% média, 1% vol diária
    
    # Calcula valores do portfólio
    initial_value = 50_000_000
    cumulative_returns = (1 + returns/100).cumprod()
    total_values = initial_value * cumulative_returns
    
    # Cria DataFrame
    portfolio_history = pd.DataFrame({
        'total_value': total_values,
        'returns': returns
    }, index=dates)
    
    # Alinha CDI
    cdi_aligned = align_cdi_to_portfolio(dates, cdi)
    
    # Calcula métricas
    metrics = calculate_metrics(portfolio_history, cdi_aligned)
    
    # Imprime

    print_metrics(metrics)
