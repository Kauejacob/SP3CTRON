# orchestrator.py
"""
Orquestrador do pipeline multi-agente.
Gerencia o fluxo: Analyst → Bear + Bull → Senior → Decisão Final
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from typing import Optional

from models.schemas import TradingState
from agents.analyst import run_analyst
from agents.bear import run_bear
from agents.bull import run_bull
from agents.senior import run_senior 


def run_trading_pipeline(
    ticker: str,
    as_of: Optional[str] = None,
    min_confidence: float = 0.7,
    verbose: bool = True
) -> TradingState:
    """
    Executa o pipeline completo de análise multi-agente.
    
    Args:
        ticker: Símbolo da ação (ex: PETR4.SA)
        as_of: Data de referência YYYY-MM-DD (None = hoje)
        min_confidence: Confiança mínima para prosseguir (0-1)
        verbose: Se True, imprime progresso
    
    Returns:
        TradingState com todos os outputs dos agentes
    """
    
    start_time = time.time()
    
    # Inicializa estado
    state = TradingState(
        ticker=ticker,
        as_of=as_of or time.strftime("%Y-%m-%d"),
        pipeline_status="initialized"
    )
    
    if verbose:
        print("\n" + "="*70)
        print(f"🚀 INICIANDO PIPELINE MULTI-AGENTE: {ticker}")
        print("="*70)
    
    try:
        # ============ STEP 1: ANALISTA FUNDAMENTAL ============
        if verbose:
            print(f"\n[STEP 1/3] 📊 Analista Fundamental")
            print("-" * 70)
        
        analyst_result = run_analyst(ticker, as_of)
        
        if analyst_result["status"] != "success":
            state.errors.append(f"Analista falhou: {analyst_result.get('message')}")
            state.pipeline_status = "failed"
            return state
        
        # Verifica confiança
        if analyst_result.get("confidence", 0) < min_confidence:
            warning = f"Confiança baixa do analista ({analyst_result['confidence']:.0%})"
            state.warnings.append(warning)
            if verbose:
                print(f"   ⚠️ {warning}")
        
        state.analyst_report = analyst_result["report"]
        state.pipeline_status = "analyst_done"
        
        if verbose:
            print(f"\n   ✅ Análise do Analista completa")
            print(f"      Veredito: {state.analyst_report.verdict.value.upper()}")
            print(f"      Score: {state.analyst_report.score:.1f}/100")
            print(f"      Confiança: {state.analyst_report.confidence:.0%}")
        
        # ============ STEP 2: ESPECIALISTAS (BEAR + BULL) ============
        if verbose:
            print(f"\n[STEP 2/3] 🐻🐂 Especialistas Debatendo")
            print("-" * 70)
        
        # Bear (Pessimista)
        try:
            bear_perspective = run_bear(state.analyst_report, verbose=verbose)
            state.bear_perspective = bear_perspective
        except Exception as e:
            error_msg = f"Agente Bear falhou: {e}"
            state.errors.append(error_msg)
            if verbose:
                print(f"   ❌ {error_msg}")
        
        # Bull (Otimista)
        try:
            bull_perspective = run_bull(state.analyst_report, verbose=verbose)
            state.bull_perspective = bull_perspective
        except Exception as e:
            error_msg = f"Agente Bull falhou: {e}"
            state.errors.append(error_msg)
            if verbose:
                print(f"   ❌ {error_msg}")
        
        # Verifica se pelo menos um especialista funcionou
        if not state.bear_perspective and not state.bull_perspective:
            state.errors.append("Ambos especialistas falharam")
            state.pipeline_status = "failed"
            return state
        
        state.pipeline_status = "specialists_done"
        
        if verbose:
            print(f"\n   ✅ Especialistas concluíram análise")
            if state.bear_perspective:
                print(f"      🐻 Bear: {state.bear_perspective.recommended_action.value.upper()} "
                      f"(downside: {state.bear_perspective.estimated_downside:.1f}%)")
            if state.bull_perspective:
                print(f"      🐂 Bull: {state.bull_perspective.recommended_action.value.upper()} "
                      f"(upside: +{state.bull_perspective.estimated_upside:.1f}%)")
        
        # ============ STEP 3: SENIOR (DECISÃO FINAL) ============
        if verbose:
            print(f"\n[STEP 3/3] 👔 Decisão do Senior")
            print("-" * 70)
        
        try:
            senior_decision = run_senior(
                analyst_report=state.analyst_report,
                bear_perspective=state.bear_perspective,
                bull_perspective=state.bull_perspective,
                verbose=verbose
            )
            state.senior_decision = senior_decision
            state.pipeline_status = "completed"
            
            if verbose:
                print(f"\n   ✅ Decisão Final do Senior")
                print(f"      Veredito: {senior_decision.final_verdict.value.upper()}")
                print(f"      Tamanho da Posição: {senior_decision.position_size:.1f}%")
                print(f"      Horizonte: {senior_decision.holding_period}")
                
        except Exception as e:
            error_msg = f"Agente Senior falhou: {e}"
            state.errors.append(error_msg)
            state.pipeline_status = "failed"
            if verbose:
                print(f"   ❌ {error_msg}")
        
    except Exception as e:
        state.errors.append(f"Erro crítico no pipeline: {e}")
        state.pipeline_status = "failed"
        if verbose:
            print(f"\n❌ Erro crítico: {e}")
    
    finally:
        state.execution_time_seconds = time.time() - start_time
        
        if verbose:
            print("\n" + "="*70)
            print(f"⏱️ Pipeline concluído em {state.execution_time_seconds:.2f}s")
            print(f"📊 Status final: {state.pipeline_status}")
            if state.warnings:
                print(f"⚠️ Avisos: {len(state.warnings)}")
            if state.errors:
                print(f"❌ Erros: {len(state.errors)}")
            print("="*70)
    
    return state


# ============ TESTE ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Multi-Agente de Trading")
    parser.add_argument("--ticker", type=str, default="PETR4.SA", help="Ticker da ação")
    parser.add_argument("--as-of", type=str, help="Data de referência (YYYY-MM-DD)")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso")
    
    args = parser.parse_args()
    
    # Executa pipeline
    state = run_trading_pipeline(
        ticker=args.ticker,
        as_of=args.as_of,
        verbose=not args.quiet
    )
    
    # Summary
    print("\n" + "="*70)
    print("📋 RESUMO DA ANÁLISE")
    print("="*70)
    
    if state.analyst_report:
        print(f"\n📊 Analista: {state.analyst_report.verdict.value.upper()} "
              f"(score: {state.analyst_report.score:.1f})")
    
    if state.bear_perspective:
        print(f"🐻 Bear: {state.bear_perspective.recommended_action.value.upper()} "
              f"(downside: {state.bear_perspective.estimated_downside:.1f}%, "
              f"prob: {state.bear_perspective.downside_probability:.0%})")
    
    if state.bull_perspective:
        print(f"🐂 Bull: {state.bull_perspective.recommended_action.value.upper()} "
              f"(upside: +{state.bull_perspective.estimated_upside:.1f}%, "
              f"prob: {state.bull_perspective.upside_probability:.0%})")
    
    if state.senior_decision:
        print(f"\n👔 Decisão Final: {state.senior_decision.final_verdict.value.upper()}")
        print(f"   Tamanho da Posição: {state.senior_decision.position_size:.1f}%")
        print(f"   Horizonte: {state.senior_decision.holding_period}")
        print(f"   Confiança: {state.senior_decision.confidence:.0%}")
    
    print(f"\n⏱️ Tempo total: {state.execution_time_seconds:.2f}s")
    print("\n" + "="*70)