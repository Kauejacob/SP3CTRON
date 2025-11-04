from pydantic import BaseModel, Field
from typing import  Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class Verdict(str, Enum):
    """Possíveis vereditos de trading"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class FundamentalSnapshot(BaseModel):
    """Dados brutos coletados do yfinance"""
    ticker: str
    as_of: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_out: Optional[float] = None
    
    # Valuation
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    
    # Qualidade/Lucratividade
    gross_margin: Optional[float] = None
    op_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    
    # Crescimento
    revenue_growth_yoy: Optional[float] = None
    net_income_growth_yoy: Optional[float] = None
    
    # Risco/Alavancagem
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    
    # Metadados
    evidence: List[str] = Field(default_factory=list)


class FundamentalReport(BaseModel):
    """Output do Agente Analista"""
    ticker: str
    as_of: str
    verdict: Verdict
    score: float
    confidence: float
    summary: str
    rationale: List[str]
    risks: List[str]
    snapshot: dict  # FundamentalSnapshot serializado

class BearPerspective(BaseModel):
    """Output do Agente Pessimista (Bear)"""
    ticker: str
    as_of: str
    
    # Análise pessimista
    concerns: List[str] = Field(
        description="5-7 preocupações específicas baseadas nos dados"
    )
    worst_case_scenario: str = Field(
        description="Narrativa detalhada do pior cenário possível"
    )
    
    # Probabilidades e riscos
    downside_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probabilidade estimada (0-1) do cenário negativo"
    )
    estimated_downside: float = Field(
        description="Queda percentual estimada no pior caso (ex: -20.5 para -20.5%)"
    )
    
    # Recomendação
    recommended_action: Verdict
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confiança na análise bearish (0-1)"
    )
    
    # Evidências do analista
    evidence_from_analyst: List[str] = Field(
        description="Pontos específicos do relatório do analista que suportam a visão bearish"
    )
    
    # Dados utilizados
    key_metrics_analyzed: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas-chave que embasaram a análise"
    )


class BullPerspective(BaseModel):
    """Output do Agente Otimista (Bull)"""
    ticker: str
    as_of: str
    
    # Análise otimista
    opportunities: List[str] = Field(
        description="5-7 oportunidades/catalisadores identificados"
    )
    best_case_scenario: str = Field(
        description="Narrativa detalhada do melhor cenário possível"
    )
    
    # Probabilidades e potencial
    upside_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probabilidade estimada (0-1) do cenário positivo"
    )
    estimated_upside: float = Field(
        description="Alta percentual estimada no melhor caso (ex: 35.2 para +35.2%)"
    )
    
    # Recomendação
    recommended_action: Verdict
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confiança na análise bullish (0-1)"
    )
    
    # Evidências do analista
    evidence_from_analyst: List[str] = Field(
        description="Pontos específicos do relatório do analista que suportam a visão bullish"
    )
    
    # Dados utilizados
    key_metrics_analyzed: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métricas-chave que embasaram a análise"
    )


class SeniorDecision(BaseModel):
    """Output do Agente Senior (decisão final)"""
    ticker: str
    as_of: str
    
    # Decisão final
    final_verdict: Verdict
    position_size: float = Field(
        ge=0.0, le=100.0,
        description="Tamanho da posição (% do portfólio, 0-100)"
    )
    
    # Risk management
    stop_loss: Optional[float] = Field(
        None,
        description="Preço de stop loss (se aplicável)"
    )
    take_profit: Optional[float] = Field(
        None,
        description="Preço de take profit (se aplicável)"
    )
    holding_period: str = Field(
        description="Horizonte de investimento: short-term, medium-term, long-term"
    )
    
    # Síntese
    synthesis: str = Field(
        description="Como reconciliou as visões bull/bear e chegou à decisão"
    )
    key_decision_factors: List[str] = Field(
        description="3-5 fatores-chave que determinaram a decisão"
    )
    
    # Confiança
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confiança na decisão final (0-1)"
    )


class TradingState(BaseModel):
    """Estado global compartilhado entre todos os agentes"""
    
    # Identificação
    ticker: str
    as_of: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Outputs dos agentes (preenchidos sequencialmente)
    analyst_report: Optional[FundamentalReport] = None
    bear_perspective: Optional[BearPerspective] = None
    bull_perspective: Optional[BullPerspective] = None
    senior_decision: Optional[SeniorDecision] = None
    
    # Metadados do pipeline
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    execution_time_seconds: Optional[float] = None
    
    # Status do pipeline
    pipeline_status: str = Field(
        default="initialized",
        description="initialized | analyst_done | specialists_done | senior_done | completed | failed"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }