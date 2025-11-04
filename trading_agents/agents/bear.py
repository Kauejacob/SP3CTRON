# agents/bear.py
"""
Agente Pessimista (Bear) - Analisa riscos e cenários negativos.
"""

# ============ IMPORTS E CONFIGURAÇÃO DE PATH ============
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ CARREGA VARIÁVEIS DE AMBIENTE ============
# Carrega .env da raiz do projeto
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
# Carrega o .env de forma robusta
env_path = find_dotenv(usecwd=True)  # procura a partir do CWD do processo
if not env_path:  # se não encontrou, force o caminho relativo ao arquivo atual
    env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Valida se a API key foi carregada
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada no .env!")

# Remove espaços
OPENAI_API_KEY = OPENAI_API_KEY.strip()

# ============ IMPORTS DO PROJETO ============
import json
from typing import Optional
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from models.schemas import BearPerspective, FundamentalReport, Verdict


# ============ PROMPT DO AGENTE BEAR ============

BEAR_INSTRUCTIONS = """
Você é um **Analista Bearish (Pessimista) Sênior** com 20 anos de experiência em identificar riscos e problemas em empresas.

## SUA MISSÃO:
Analisar CRITICAMENTE o relatório do analista fundamental e os dados da empresa, focando em:
- Riscos estruturais e conjunturais
- Vulnerabilidades financeiras
- Ameaças competitivas e de mercado
- Sinais de deterioração
- Fatores que podem levar a perdas

## PROTOCOLO DE ANÁLISE:

### 1. Analise os Dados Fornecidos
Você receberá:
- Relatório completo do Analista Fundamental
- Snapshot com dados financeiros brutos
- Score e subscores de valuation/quality/risk

### 2. Identifique Preocupações Específicas (concerns)
Liste 5-7 preocupações CONCRETAS baseadas nos dados:
- Se P/E alto: "P/E de X está Y% acima da média, indicando sobrevalorização"
- Se dívida alta: "D/E de X indica alto risco de alavancagem"
- Se margens caindo: "Margem líquida caiu Z% YoY, sinalizando pressão competitiva"

**REGRAS:**
- Cite NÚMEROS EXATOS dos dados
- Cada concern deve ter evidência quantitativa
- Evite generalidades ("mercado pode cair" ❌) → seja específico ("ROE de 8% vs 15% do setor indica ineficiência" ✅)

### 3. Cenário Pessimista (worst_case_scenario)
Construa uma narrativa do PIOR CENÁRIO plausível (2-3 parágrafos):
- O que pode dar errado?
- Encadeamento de eventos negativos
- Impacto estimado no preço/fundamentals
- Baseie-se nos dados reais fornecidos

### 4. Probabilidades e Estimativas
- **downside_probability** (0-1): Quão provável é o cenário negativo?
  * 0.7-1.0: Altamente provável, dados críticos
  * 0.4-0.7: Moderadamente provável, alguns red flags
  * 0.0-0.4: Pouco provável, mas riscos existem

- **estimated_downside** (% negativo): Queda estimada no pior caso
  * Ex: -15.5 significa queda de 15.5%
  * Base em múltiplos setoriais, histórico de stress

### 5. Recomendação
- **recommended_action**: SELL (se riscos críticos) | HOLD (se moderados) | BUY (só se upside compensar riscos)
- **confidence** (0-1): Sua confiança na análise

### 6. Evidências do Analista
Liste 3-5 pontos ESPECÍFICOS do relatório do analista que suportam sua visão bearish.
Cite textualmente se possível.

### 7. Métricas-Chave Analisadas
Destaque as métricas que mais pesaram na análise:
```json
{
  "pe_ratio": 45.2,
  "debt_to_equity": 2.1,
  "roe": 0.08,
  "current_ratio": 0.7
}
```

## FORMATO DE SAÍDA:
Retorne JSON seguindo EXATAMENTE o schema BearPerspective.

## EXEMPLO:
```json
{
  "ticker": "XPTO4.SA",
  "as_of": "2024-03-29",
  "concerns": [
    "P/E de 45x está 80% acima da média do setor de 25x, indicando sobrevalorização extrema",
    "D/E de 2.1x sugere alto risco de insolvência em cenário de alta de juros",
    "Margem líquida de 8% caiu 30% YoY, sinalizando pressão competitiva intensa",
    "Current ratio de 0.7 indica problemas de liquidez de curto prazo",
    "ROE de 8% está 47% abaixo da média setorial de 15%, indicando baixa eficiência"
  ],
  "worst_case_scenario": "No pior cenário, a empresa enfrenta aperto de liquidez devido ao current ratio baixo (0.7), forçando renegociação de dívidas a taxas mais altas. Com D/E de 2.1x e juros subindo, o custo de capital pode dobrar, comprimindo ainda mais as margens já pressionadas (queda de 30% YoY). A sobrevalorização (P/E 80% acima do setor) torna o papel vulnerável a correção abrupta. Em cenário de recessão, combinando deterioração de margens, problemas de liquidez e múltiplos insustentáveis, o papel pode cair 40-50% até atingir P/E de 25x (média setorial).",
  "downside_probability": 0.65,
  "estimated_downside": -45.0,
  "recommended_action": "sell",
  "confidence": 0.75,
  "evidence_from_analyst": [
    "Analista identificou 'D/E de 2.1x indica alto risco financeiro'",
    "Score de valuation foi 0.0, indicando empresa cara",
    "Analista alertou: 'Pressão competitiva pode comprimir margens'"
  ],
  "key_metrics_analyzed": {
    "pe_ratio": 45.2,
    "debt_to_equity": 2.1,
    "net_margin": 0.08,
    "roe": 0.08,
    "current_ratio": 0.7
  }
}
```

## REGRAS CRÍTICAS:
- Use APENAS dados fornecidos
- Cite números EXATOS
- Seja PESSIMISTA mas REALISTA
- Cada afirmação deve ter evidência quantitativa
- JSON puro, sem markdown
"""


# ============ AGENTE ============

bear_agent = Agent(
    name="BearAnalyst",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=BEAR_INSTRUCTIONS,
)


# ============ ORCHESTRATOR ============

def run_bear(
    analyst_report: FundamentalReport,
    verbose: bool = True
) -> BearPerspective:
    """
    Executa análise pessimista baseada no relatório do analista.
    
    Args:
        analyst_report: Relatório do analista fundamental
        verbose: Se True, imprime progresso
    
    Returns:
        BearPerspective com análise pessimista
    """
    
    if verbose:
        print(f"\n🐻 Analisando perspectiva BEARISH para {analyst_report.ticker}...")
    
    # Prepara contexto para o agente
    prompt = f"""
Analise os dados abaixo sob uma perspectiva PESSIMISTA e identifique todos os riscos e vulnerabilidades.

# RELATÓRIO DO ANALISTA FUNDAMENTAL

**Ticker:** {analyst_report.ticker}
**Data:** {analyst_report.as_of}
**Veredito do Analista:** {analyst_report.verdict.value.upper()}
**Score:** {analyst_report.score:.1f}/100 (confiança: {analyst_report.confidence:.0%})

**Summary:**
{analyst_report.summary}

**Rationale:**
{chr(10).join(f"  • {r}" for r in analyst_report.rationale)}

**Risks identificados:**
{chr(10).join(f"  • {r}" for r in analyst_report.risks)}

# DADOS FINANCEIROS BRUTOS

{json.dumps(analyst_report.snapshot, indent=2, ensure_ascii=False)}

---

Gere a análise bearish em JSON seguindo o schema BearPerspective.
Foque nos RISCOS e no que pode dar ERRADO.
"""
    
    if verbose:
        print("   Gerando análise pessimista via LLM...")
    
    response = bear_agent.run(prompt)
    
    # Parse da resposta
    try:
        content = str(response.content)
        
        # Remove markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        bear_dict = json.loads(content)
        
        # Valida com Pydantic
        bear_perspective = BearPerspective(**bear_dict)
        
        if verbose:
            print(f"   ✅ Análise concluída: {bear_perspective.recommended_action.value.upper()}")
            print(f"      Downside: {bear_perspective.estimated_downside:.1f}%")
            print(f"      Probabilidade: {bear_perspective.downside_probability:.0%}")
            print(f"      Confiança: {bear_perspective.confidence:.0%}")
        
        return bear_perspective
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Erro ao parsear resposta: {e}")
        
        raise ValueError(
            f"Falha ao parsear resposta do agente Bear: {e}\n"
            f"Resposta bruta: {str(response.content)[:500]}"
        )


# ============ TESTE STANDALONE ============

if __name__ == "__main__":
    # Para testar, precisa de um relatório do analista
    print("⚠️ Este agente precisa de um FundamentalReport como input.")
    print("   Execute via orchestrator.py ou crie um report manualmente para teste.")
    
    # Exemplo de teste com dados mock:
    from models.schemas import FundamentalSnapshot
    
    mock_snapshot = {
        "ticker": "TEST4.SA",
        "as_of": "2024-03-29",
        "price": 50.0,
        "pe": 45.0,
        "debt_to_equity": 2.1,
        "net_margin": 0.08,
        "roe": 0.08,
        "current_ratio": 0.7,
        "evidence": ["mock_data"]
    }
    
    mock_report = FundamentalReport(
        ticker="TEST4.SA",
        as_of="2024-03-29",
        verdict=Verdict.HOLD,
        score=55.0,
        confidence=0.75,
        summary="Empresa com valuation alto e margens pressionadas",
        rationale=[
            "P/E de 45x está acima da média",
            "D/E de 2.1x indica alto endividamento",
            "Margens em queda"
        ],
        risks=[
            "Risco de liquidez",
            "Pressão competitiva"
        ],
        snapshot=mock_snapshot
    )
    
    print("\n🧪 Testando com dados mock...")
    bear_result = run_bear(mock_report, verbose=True)
    
    print("\n" + "="*70)
    print("RESULTADO DA ANÁLISE BEARISH")
    print("="*70)
    print(f"\n🎯 Recomendação: {bear_result.recommended_action.value.upper()}")
    print(f"📉 Downside estimado: {bear_result.estimated_downside:.1f}%")
    print(f"⚠️ Probabilidade: {bear_result.downside_probability:.0%}")
    
    print(f"\n🔍 Preocupações:")
    for concern in bear_result.concerns:
        print(f"   • {concern}")
    
    print(f"\n📖 Pior Cenário:")
    print(f"   {bear_result.worst_case_scenario}")