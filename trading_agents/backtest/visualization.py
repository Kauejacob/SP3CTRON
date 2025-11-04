# backtest/visualization.py
"""
Visualizações para resultados do backtest.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional
import seaborn as sns

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_portfolio_vs_cdi(
    history_df: pd.DataFrame,
    cdi_series: pd.Series,
    initial_capital: float,
    save_path: Optional[str] = None
):
    """
    Plota evolução do portfólio vs CDI.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # SUBPLOT 1: VALOR ACUMULADO
    ax1 = axes[0]
    
    portfolio_values = history_df['total_value']
    ax1.plot(
        portfolio_values.index,
        portfolio_values,
        label='Portfólio Multi-Agente',
        linewidth=2.5,
        color='#2E86AB'
    )
    
    cdi_cumulative = initial_capital * (1 + cdi_series).cumprod()
    ax1.plot(
        cdi_cumulative.index,
        cdi_cumulative,
        label='CDI',
        linewidth=2,
        linestyle='--',
        color='#A23B72'
    )
    
    ax1.set_title('Evolução do Portfólio vs CDI', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Valor (R$)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'R$ {x/1e6:.1f}M'))
    
    # SUBPLOT 2: RETORNO ACUMULADO
    ax2 = axes[1]
    
    portfolio_returns = ((portfolio_values / initial_capital) - 1) * 100
    ax2.plot(portfolio_returns.index, portfolio_returns, label='Portfólio', linewidth=2.5, color='#2E86AB')
    
    cdi_returns_cum = ((cdi_cumulative / initial_capital) - 1) * 100
    ax2.plot(cdi_returns_cum.index, cdi_returns_cum, label='CDI', linewidth=2, linestyle='--', color='#A23B72')
    
    ax2.fill_between(
        portfolio_returns.index,
        portfolio_returns,
        cdi_returns_cum,
        where=(portfolio_returns >= cdi_returns_cum),
        alpha=0.3,
        color='green',
        label='Outperformance'
    )
    
    ax2.fill_between(
        portfolio_returns.index,
        portfolio_returns,
        cdi_returns_cum,
        where=(portfolio_returns < cdi_returns_cum),
        alpha=0.3,
        color='red',
        label='Underperformance'
    )
    
    ax2.set_title('Retorno Acumulado (%)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_ylabel('Retorno (%)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✅ Salvo: {os.path.basename(save_path)}")
    
    plt.close()


def plot_drawdown(history_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plota drawdown do portfólio.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    returns = history_df['returns'] / 100
    cumulative = (1 + returns[1:]).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max) * 100
    
    ax.fill_between(drawdown.index, 0, drawdown, alpha=0.7, color='red', label='Drawdown')
    ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5)
    
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    
    ax.scatter([max_dd_idx], [max_dd_val], color='black', s=100, zorder=5, label=f'Max DD: {max_dd_val:.2f}%')
    
    ax.set_title('Drawdown ao Longo do Tempo', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✅ Salvo: {os.path.basename(save_path)}")
    
    plt.close()


def plot_monthly_returns(history_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plota heatmap de retornos mensais.
    """
    returns = history_df['returns'] / 100
    monthly_returns = returns.resample('M').apply(lambda x: ((1 + x).prod() - 1) * 100)
    
    monthly_returns_pivot = monthly_returns.to_frame('returns')
    monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
    monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
    
    pivot_table = monthly_returns_pivot.pivot(index='year', columns='month', values='returns')
    
    month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    pivot_table.columns = [month_names[m-1] for m in pivot_table.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Retorno (%)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Retornos Mensais (%)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mês', fontsize=12)
    ax.set_ylabel('Ano', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✅ Salvo: {os.path.basename(save_path)}")
    
    plt.close()


def create_performance_report(results: Dict, save_dir: Optional[str] = None):
    """
    Cria relatório visual completo.
    
    Args:
        results: Dict retornado por BacktestEngine.get_results()
        save_dir: Diretório para salvar figuras (None = apenas mostra)
    """
    print("\n📊 Gerando relatório visual...")
    
    history_df = results['history']
    cdi_series = results['cdi']
    
    if history_df.empty or len(history_df) < 2:
        print("   ⚠️ Histórico vazio ou muito curto, pulando gráficos")
        return
    
    # ✅ CRIA DIRETÓRIO SE NÃO EXISTIR
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"   📁 Diretório: {save_dir}")
    
    initial_capital = history_df['total_value'].iloc[0]
    
    try:
        # 1. Portfólio vs CDI
        print("   1️⃣ Plotando evolução do portfólio...")
        save_path = f"{save_dir}/portfolio_vs_cdi.png" if save_dir else None
        plot_portfolio_vs_cdi(history_df, cdi_series, initial_capital, save_path)
        
        # 2. Drawdown
        print("   2️⃣ Plotando drawdown...")
        save_path = f"{save_dir}/drawdown.png" if save_dir else None
        plot_drawdown(history_df, save_path)
        
        # 3. Retornos mensais
        print("   3️⃣ Plotando retornos mensais...")
        save_path = f"{save_dir}/monthly_returns.png" if save_dir else None
        plot_monthly_returns(history_df, save_path)
        
        print("   ✅ Relatório visual concluído!")
        
    except Exception as e:
        print(f"   ⚠️ Erro ao gerar gráficos: {e}")
        import traceback
        traceback.print_exc()


# ============ TESTE ============

if __name__ == "__main__":
    print("🧪 TESTE DO MÓDULO VISUALIZATION")
    print("="*70)
    
    # Simula dados
    dates = pd.bdate_range(start="2023-01-01", end="2024-01-01")
    
    np.random.seed(42)
    returns = np.random.normal(0.05, 0.8, len(dates))
    
    initial_value = 50_000_000
    cumulative_returns = (1 + returns/100).cumprod()
    total_values = initial_value * cumulative_returns
    
    history_df = pd.DataFrame({
        'total_value': total_values,
        'returns': returns
    }, index=dates)
    
    # CDI simulado
    cdi_daily = 0.00035
    cdi_series = pd.Series(cdi_daily, index=dates)
    
    # Testa create_performance_report
    results = {
        'history': history_df,
        'cdi': cdi_series
    }
    
    print("\nTestando create_performance_report...")
    create_performance_report(results, save_dir="test_plots")
    
    print("\n✅ Teste concluído!")
    print(f"📁 Gráficos salvos em: test_plots/")