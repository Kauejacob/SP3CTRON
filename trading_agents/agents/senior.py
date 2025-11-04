# agents/senior.py
"""
Agente Senior - Toma decisão final sintetizando todas as perspectivas.
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

from models.schemas import (
    SeniorDecision, 
    FundamentalReport, 
    BearPerspective, 
    BullPerspective,
    Verdict
)


# ============ PROMPT DO AGENTE SENIOR ============

SENIOR_INSTRUCTIONS = """
Você é o **Head Trader Sênior** de um fundo quantitativo multi-bilionário, com 30 anos de experiência em gestão de portfólios institucionais.

## SUA MISSÃO:
Sintetizar TODAS as análises recebidas (Analista Fundamental, Especialista Bear, Especialista Bull) e tomar a **DECISÃO FINAL** sobre:
- Veredito de investimento (BUY/SELL/HOLD)
- Tamanho da posição (% do portfólio)
- Níveis de stop-loss e take-profit
- Horizonte de investimento

## PROTOCOLO DE DECISÃO:

### 1. Inputs que Você Receberá
- **Relatório do Analista Fundamental**: Veredito, score, rationale, risks
- **Perspectiva Bear (Pessimista)**: Concerns, worst case, downside estimado, probabilidade
- **Perspectiva Bull (Otimista)**: Opportunities, best case, upside estimado, probabilidade
- **Dados Financeiros Brutos**: Snapshot completo da empresa

### 2. Framework de Análise

#### A. Avaliação de Consenso
- **Analista, Bear e Bull concordam?**
  * 3 BUY → Forte convicção, posição agressiva
  * 2 BUY + 1 HOLD → Convicção moderada, posição normal
  * Divergência total → Cautela, posição reduzida ou HOLD

#### B. Análise de Assimetria Risco-Retorno
- **Ratio de Sharpe Esperado** = (Upside × Prob_Bull - |Downside| × Prob_Bear) / Volatilidade_Implícita
- **Regra**: Só entre se Expected_Return > 2× Expected_Risk

#### C. Qualidade dos Argumentos
- Qual perspectiva tem evidências mais sólidas?
- Os números sustentam a narrativa?
- Há red flags ignorados pelo Bull?
- Há catalisadores ignorados pelo Bear?

#### D. Contexto de Portfólio
- Diversificação setorial
- Correlações
- Risco agregado

### 3. Determinação da Posição (position_size)

**ESTRATÉGIA DE PORTFÓLIO DIVERSIFICADO:**

Para manter 85-95% sempre investido com 15-25 posições:

| Score Analyst | Qualidade | Position Size Base | Ajustes |
|---------------|-----------|-------------------|---------|
| **85-100** | Excelente | 6-8% | +1% se Bull forte |
| **75-84** | Muito boa | 5-7% | - |
| **65-74** | Boa | 4-6% | -1% se Bear moderado |
| **55-64** | Razoável (HOLD) | 2-4% | -1% se liquidez baixa |
| **45-54** | Fraca (HOLD) | 1-3% | Apenas se necessário |
| **<45** | Ruim | 0% | SELL |

**IMPORTANTE:** 
- Para BUY signals (65+): Position size = 4-8%
- Para HOLD signals (45-64): Position size = 2-4%
- Com 15-20 posições, terá 85-95% investido
- Cash (5-15%) rende SELIC automaticamente

---

### 4. Stop Loss e Take Profit

**Stop Loss:**

Para estratégia buy-and-hold, **NÃO use stop loss**.

**Justificativa:** 
- Rebalanceamento mensal já funciona como proteção
- Stop loss causa vendas prematuras em volatilidade temporária
- Empresas boas (score 65+) se recuperam

**Definir sempre:**
```python
stop_loss = None
take_profit = None
```

### 5. Horizonte de Investimento

- **short-term** (1-3 meses): Trades táticos, catalisadores iminentes
- **medium-term** (3-12 meses): Tese fundamentalista sólida
- **long-term** (>12 meses): Empresas estruturalmente fortes, múltiplos muito comprimidos

**Critérios:**
- Se Score > 75 E P/E < 10 → long-term
- Se catalisador específico (earnings, evento) → short-term
- Padrão → medium-term

### 6. Synthesis (Narrativa da Decisão)

Explique em 2-3 parágrafos:
1. **O que pesou mais**: Qual análise foi determinante?
2. **Como reconciliou divergências**: Se Bear e Bull discordam, como chegou ao meio-termo?
3. **Fatores-chave**: Quais 3-5 fatores foram decisivos?

### 7. Key Decision Factors

Liste 3-5 fatores ESPECÍFICOS que determinaram a decisão:
- "Assimetria 3:1 favorável (upside 45% vs downside 15%)"
- "Consenso bullish entre os 3 agentes"
- "P/E de 8x com ROE de 20% indica deep value"
- "Bear identificou risco de liquidez que limita posição a 5%"

### 8. Confidence

Sua confiança na decisão (0-1):
- 0.9-1.0: Convicção extrema, todos indicadores alinhados
- 0.7-0.9: Alta convicção, pequenas ressalvas
- 0.5-0.7: Moderada, alguns pontos de atenção
- 0.3-0.5: Baixa, muita incerteza
- <0.3: Muito baixa, melhor ficar de fora

## FORMATO DE SAÍDA:
Retorne JSON seguindo EXATAMENTE o schema SeniorDecision.

## EXEMPLO:
```json
{
  "ticker": "PETR4.SA",
  "as_of": "2024-03-29",
  "final_verdict": "buy",
  "position_size": 7.5,
  "stop_loss": 29.50,
  "take_profit": 42.00,
  "holding_period": "medium-term",
  "synthesis": "A análise converge para uma oportunidade de compra com boa assimetria risco-retorno. O Analista identificou P/E de 4.2x com margens saudáveis (score 72.5), o Bull projetou upside de 35% com probabilidade de 70%, enquanto o Bear alertou para volatilidade do petróleo mas com downside limitado a -15% e probabilidade de apenas 40%. A assimetria 2.3:1 favorável (35% upside vs 15% downside, ponderado por probabilidades) justifica posição de 7.5%. O horizonte medium-term (6-9 meses) captura o rerating de múltiplos sem exposição excessiva a volatilidade de curto prazo.",
  "key_decision_factors": [
    "Assimetria risco-retorno 2.3:1 favorável (upside 35% prob 70% vs downside 15% prob 40%)",
    "Consenso de compra entre Analista (BUY 72.5) e Bull (BUY upside 35%), com Bear moderado",
    "Valuation extremamente comprimido: P/E 4.2x com ROE 28% indica deep value",
    "Margens operacionais de 35% demonstram resiliência a choques de preço",
    "Risco de volatilidade do petróleo limitado por hedge natural via exportações"
  ],
  "confidence": 0.82
}
```

## REGRAS CRÍTICAS:
- Seja RIGOROSO com position_size (máx 10%)
- SEMPRE considere downside do Bear
- NÃO ignore riscos para justificar position grande
- Se incerteza alta (confidence < 0.6), reduza posição
- JSON puro, sem markdown
"""


# ============ AGENTE ============

senior_agent = Agent(
    name="SeniorTrader",
    model=OpenAIChat(id="gpt-4o"),  # Usa modelo mais potente para decisão final
    instructions=SENIOR_INSTRUCTIONS,
)


# ============ ORCHESTRATOR ============

def run_senior(
    analyst_report: FundamentalReport,
    bear_perspective: Optional[BearPerspective],
    bull_perspective: Optional[BullPerspective],
    verbose: bool = True
) -> SeniorDecision:
    """
    Executa decisão final do Senior baseada em todas as análises.
    
    Args:
        analyst_report: Relatório do analista fundamental
        bear_perspective: Análise do Bear (pode ser None se falhou)
        bull_perspective: Análise do Bull (pode ser None se falhou)
        verbose: Se True, imprime progresso
    
    Returns:
        SeniorDecision com decisão final
    """
    
    if verbose:
        print(f"\n👔 Decisão do Senior para {analyst_report.ticker}...")
    
    # Prepara contexto completo
    prompt = f"""
Você recebeu as seguintes análises para **{analyst_report.ticker}** (data: {analyst_report.as_of}):

# 1. RELATÓRIO DO ANALISTA FUNDAMENTAL

**Veredito:** {analyst_report.verdict.value.upper()}
**Score:** {analyst_report.score:.1f}/100 (confiança: {analyst_report.confidence:.0%})

**Summary:**
{analyst_report.summary}

**Rationale:**
{chr(10).join(f"  • {r}" for r in analyst_report.rationale)}

**Risks:**
{chr(10).join(f"  • {r}" for r in analyst_report.risks)}

**Dados Brutos (Snapshot):**
```json
{json.dumps(analyst_report.snapshot, indent=2, ensure_ascii=False)}
```

"""

    # Adiciona Bear se disponível
    if bear_perspective:
        prompt += f"""
# 2. PERSPECTIVA BEAR (PESSIMISTA)

**Recomendação:** {bear_perspective.recommended_action.value.upper()}
**Downside Estimado:** {bear_perspective.estimated_downside:.1f}%
**Probabilidade:** {bear_perspective.downside_probability:.0%}
**Confiança:** {bear_perspective.confidence:.0%}

**Preocupações:**
{chr(10).join(f"  • {c}" for c in bear_perspective.concerns)}

**Pior Cenário:**
{bear_perspective.worst_case_scenario}

**Evidências do Analista Citadas:**
{chr(10).join(f"  • {e}" for e in bear_perspective.evidence_from_analyst)}

**Métricas-Chave Analisadas:**
```json
{json.dumps(bear_perspective.key_metrics_analyzed, indent=2, ensure_ascii=False)}
```

"""
    else:
        prompt += """
# 2. PERSPECTIVA BEAR (PESSIMISTA)

⚠️ **Análise Bear não disponível** (falha no agente)
Prossiga com cautela extra e assuma downside conservador de -20%.

"""

    # Adiciona Bull se disponível
    if bull_perspective:
        prompt += f"""
# 3. PERSPECTIVA BULL (OTIMISTA)

**Recomendação:** {bull_perspective.recommended_action.value.upper()}
**Upside Estimado:** +{bull_perspective.estimated_upside:.1f}%
**Probabilidade:** {bull_perspective.upside_probability:.0%}
**Confiança:** {bull_perspective.confidence:.0%}

**Oportunidades:**
{chr(10).join(f"  • {o}" for o in bull_perspective.opportunities)}

**Melhor Cenário:**
{bull_perspective.best_case_scenario}

**Evidências do Analista Citadas:**
{chr(10).join(f"  • {e}" for e in bull_perspective.evidence_from_analyst)}

**Métricas-Chave Analisadas:**
```json
{json.dumps(bull_perspective.key_metrics_analyzed, indent=2, ensure_ascii=False)}
```

"""
    else:
        prompt += """
# 3. PERSPECTIVA BULL (OTIMISTA)

⚠️ **Análise Bull não disponível** (falha no agente)
Prossiga com cautela extra e assuma upside conservador de +15%.

"""

    prompt += """
---

**TAREFA:**
Sintetize TODAS as análises acima e tome a decisão final.
Gere o JSON seguindo o schema SeniorDecision.

Considere:
1. Consenso entre os agentes
2. Assimetria risco-retorno (upside vs downside ponderados por probabilidades)
3. Qualidade dos argumentos e evidências
4. Confiança de cada agente
5. Preço atual da ação para calcular stop_loss e take_profit

Seja RIGOROSO com o position_size e REALISTA com as probabilidades.
"""
    
    if verbose:
        print("   Sintetizando todas as perspectivas...")
    
    response = senior_agent.run(prompt)
    
    # Parse da resposta
    try:
        content = str(response.content)
        
        # Remove markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        senior_dict = json.loads(content)
        
        # Valida com Pydantic
        senior_decision = SeniorDecision(**senior_dict)
        
        if verbose:
            print(f"   ✅ Decisão tomada: {senior_decision.final_verdict.value.upper()}")
            print(f"      Position Size: {senior_decision.position_size:.1f}%")
            print(f"      Stop Loss: {senior_decision.stop_loss if senior_decision.stop_loss else 'N/A'}")
            print(f"      Take Profit: {senior_decision.take_profit if senior_decision.take_profit else 'N/A'}")
            print(f"      Holding Period: {senior_decision.holding_period}")
            print(f"      Confiança: {senior_decision.confidence:.0%}")
        
        return senior_decision
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Erro ao parsear resposta: {e}")
        
        raise ValueError(
            f"Falha ao parsear resposta do agente Senior: {e}\n"
            f"Resposta bruta: {str(response.content)[:500]}"
        )


# ============ TESTE STANDALONE ============

if __name__ == "__main__":
    print("⚠️ Este agente precisa de FundamentalReport + BearPerspective + BullPerspective.")
    print("   Execute via orchestrator.py completo.")