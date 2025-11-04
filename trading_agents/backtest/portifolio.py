# backtest/portfolio.py
"""
Gestão de portfólio para o sistema de backtest.
Controla posições, execução de trades, cálculo de P&L.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """Representa uma posição em um ativo."""
    ticker: str
    shares: int
    avg_price: float
    current_price: float
    entry_date: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Valor de mercado atual da posição."""
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Custo total da posição."""
        return self.shares * self.avg_price
    
    @property
    def pnl(self) -> float:
        """Profit & Loss não realizado."""
        return self.market_value - self.cost_basis
    
    @property
    def pnl_pct(self) -> float:
        """P&L percentual."""
        return (self.pnl / self.cost_basis) * 100 if self.cost_basis > 0 else 0.0
    
    def should_stop_loss(self) -> bool:
        """Verifica se atingiu stop loss."""
        if self.stop_loss is None:
            return False
        return self.current_price <= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """Verifica se atingiu take profit."""
        if self.take_profit is None:
            return False
        return self.current_price >= self.take_profit


@dataclass
class Trade:
    """Representa uma transação executada."""
    date: str
    ticker: str
    action: str  # 'BUY' ou 'SELL'
    shares: int
    price: float
    commission: float
    total_cost: float
    reason: str  # 'INITIAL', 'REBALANCE', 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL'


class Portfolio:
    """
    Gerencia portfólio com múltiplas posições.
    """
    
    def __init__(
        self,
        initial_capital: float,
        commission_pct: float = 0.001,  # 0.1% por trade
        min_position_size: float = 0.01,  # Mínimo 1% do portfólio
        max_position_size: float = 0.15,  # Máximo 15% do portfólio
    ):
        """
        Args:
            initial_capital: Capital inicial em R$
            commission_pct: Taxa de corretagem (%)
            min_position_size: Tamanho mínimo de posição (% do portfólio)
            max_position_size: Tamanho máximo de posição (% do portfólio)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Histórico de performance
        self.history = {
            'date': [],
            'total_value': [],
            'cash': [],
            'positions_value': [],
            'num_positions': [],
            'returns': [],
        }
    
    @property
    def positions_value(self) -> float:
        """Valor total das posições."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Valor total do portfólio (cash + posições)."""
        return self.cash + self.positions_value
    
    @property
    def num_positions(self) -> int:
        """Número de posições abertas."""
        return len(self.positions)
    
    @property
    def exposure(self) -> float:
        """Exposição ao mercado (% do capital investido)."""
        return (self.positions_value / self.total_value) * 100 if self.total_value > 0 else 0.0
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Atualiza preços de todas as posições.
        
        Args:
            prices: Dict com {ticker: preço_atual}
        """
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
    
    def can_buy(self, ticker: str, target_pct: float) -> Tuple[bool, int, str]:
        """
        Verifica se pode comprar e calcula quantidade de ações.
        
        Args:
            ticker: Ticker da ação
            target_pct: Percentual alvo do portfólio (0-100)
        
        Returns:
            (pode_comprar, quantidade_ações, motivo)
        """
        # Valida target_pct
        if target_pct < self.min_position_size * 100:
            return False, 0, f"Target {target_pct:.1f}% abaixo do mínimo {self.min_position_size*100:.1f}%"
        
        if target_pct > self.max_position_size * 100:
            target_pct = self.max_position_size * 100
        
        # Calcula valor alvo
        target_value = (target_pct / 100) * self.total_value
        
        # Já tem posição?
        if ticker in self.positions:
            current_value = self.positions[ticker].market_value
            additional_value = target_value - current_value
            
            if additional_value < self.total_value * self.min_position_size:
                return False, 0, f"Incremento de {additional_value/self.total_value*100:.1f}% muito pequeno"
            
            target_value = additional_value
        
        # Tem cash suficiente?
        required_cash = target_value * (1 + self.commission_pct)
        
        if required_cash > self.cash:
            # Ajusta para o máximo possível
            target_value = self.cash / (1 + self.commission_pct)
            
            if target_value < self.total_value * self.min_position_size:
                return False, 0, f"Cash insuficiente (tem R${self.cash:,.2f}, precisa R${required_cash:,.2f})"
        
        return True, int(target_value), "OK"
    
    def buy(
        self,
        ticker: str,
        price: float,
        target_pct: float,
        date: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: str = "SIGNAL"
    ) -> Optional[Trade]:
        """
        Executa ordem de compra.
        
        Args:
            ticker: Ticker da ação
            price: Preço de compra
            target_pct: Percentual alvo do portfólio
            date: Data da transação
            stop_loss: Preço de stop loss
            take_profit: Preço de take profit
            reason: Motivo da compra
        
        Returns:
            Trade executado ou None se não foi possível
        """
        can_buy, target_value, msg = self.can_buy(ticker, target_pct)
        
        if not can_buy:
            return None
        
        # Calcula quantidade de ações
        shares = int(target_value / price)
        
        if shares == 0:
            return None
        
        # Custo total (incluindo comissão)
        trade_value = shares * price
        commission = trade_value * self.commission_pct
        total_cost = trade_value + commission
        
        # Verifica cash novamente
        if total_cost > self.cash:
            # Ajusta shares
            shares = int(self.cash / (price * (1 + self.commission_pct)))
            if shares == 0:
                return None
            
            trade_value = shares * price
            commission = trade_value * self.commission_pct
            total_cost = trade_value + commission
        
        # Atualiza cash
        self.cash -= total_cost
        
        # Atualiza ou cria posição
        if ticker in self.positions:
            pos = self.positions[ticker]
            old_cost = pos.shares * pos.avg_price
            new_cost = shares * price
            pos.shares += shares
            pos.avg_price = (old_cost + new_cost) / pos.shares
            pos.current_price = price
            
            # Atualiza stop/take se fornecidos
            if stop_loss:
                pos.stop_loss = stop_loss
            if take_profit:
                pos.take_profit = take_profit
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_price=price,
                current_price=price,
                entry_date=date,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        # Registra trade
        trade = Trade(
            date=date,
            ticker=ticker,
            action='BUY',
            shares=shares,
            price=price,
            commission=commission,
            total_cost=total_cost,
            reason=reason
        )
        self.trades.append(trade)
        
        return trade
    
    def sell(
        self,
        ticker: str,
        price: float,
        date: str,
        shares: Optional[int] = None,
        reason: str = "SIGNAL"
    ) -> Optional[Trade]:
        """
        Executa ordem de venda.
        
        Args:
            ticker: Ticker da ação
            price: Preço de venda
            date: Data da transação
            shares: Quantidade a vender (None = vende tudo)
            reason: Motivo da venda
        
        Returns:
            Trade executado ou None se não tinha posição
        """
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        
        # Se não especificou shares, vende tudo
        if shares is None:
            shares = pos.shares
        
        # Não pode vender mais do que tem
        shares = min(shares, pos.shares)
        
        if shares == 0:
            return None
        
        # Valor da venda
        trade_value = shares * price
        commission = trade_value * self.commission_pct
        net_proceeds = trade_value - commission
        
        # Atualiza cash
        self.cash += net_proceeds
        
        # Atualiza ou remove posição
        pos.shares -= shares
        
        if pos.shares == 0:
            del self.positions[ticker]
        
        # Registra trade
        trade = Trade(
            date=date,
            ticker=ticker,
            action='SELL',
            shares=shares,
            price=price,
            commission=commission,
            total_cost=-net_proceeds,  # negativo = entrada de cash
            reason=reason
        )
        self.trades.append(trade)
        
        return trade
    
    def check_stops(self, date: str) -> List[Trade]:
        """
        Verifica stop loss e take profit de todas as posições.
        
        Args:
            date: Data atual
        
        Returns:
            Lista de trades executados
        """
        executed_trades = []
        tickers_to_close = []
        
        for ticker, pos in self.positions.items():
            # Stop loss
            if pos.should_stop_loss():
                tickers_to_close.append((ticker, pos.current_price, 'STOP_LOSS'))
            
            # Take profit
            elif pos.should_take_profit():
                tickers_to_close.append((ticker, pos.current_price, 'TAKE_PROFIT'))
        
        # Executa vendas
        for ticker, price, reason in tickers_to_close:
            trade = self.sell(ticker, price, date, reason=reason)
            if trade:
                executed_trades.append(trade)
        
        return executed_trades
    
    def record_state(self, date: str):
        """
        Registra estado atual do portfólio no histórico.
        
        Args:
            date: Data do registro
        """
        total = self.total_value
        
        # Calcula retorno diário
        if len(self.history['total_value']) > 0:
            prev_value = self.history['total_value'][-1]
            daily_return = ((total / prev_value) - 1) * 100 if prev_value > 0 else 0.0
        else:
            daily_return = 0.0
        
        self.history['date'].append(date)
        self.history['total_value'].append(total)
        self.history['cash'].append(self.cash)
        self.history['positions_value'].append(self.positions_value)
        self.history['num_positions'].append(self.num_positions)
        self.history['returns'].append(daily_return)
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Retorna histórico como DataFrame.
        """
        df = pd.DataFrame(self.history)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        Retorna trades como DataFrame.
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'date': trade.date,
                'ticker': trade.ticker,
                'action': trade.action,
                'shares': trade.shares,
                'price': trade.price,
                'commission': trade.commission,
                'total_cost': trade.total_cost,
                'reason': trade.reason
            })
        
        df = pd.DataFrame(trades_data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_positions_summary(self) -> pd.DataFrame:
        """
        Retorna resumo das posições atuais.
        """
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = []
        for ticker, pos in self.positions.items():
            positions_data.append({
                'ticker': ticker,
                'shares': pos.shares,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'cost_basis': pos.cost_basis,
                'pnl': pos.pnl,
                'pnl_pct': pos.pnl_pct,
                'weight_pct': (pos.market_value / self.total_value) * 100,
                'entry_date': pos.entry_date,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit
            })
        
        df = pd.DataFrame(positions_data)
        df = df.sort_values('weight_pct', ascending=False)
        return df
    

    # backtest/portfolio.py (adicionar no final da classe Portfolio, ANTES do método summary)

    def apply_selic_to_cash(self, date: str, selic_daily_rate: float) -> None:
        """
        Aplica rendimento SELIC ao cash disponível.
        
        Args:
            date: Data atual (formato YYYY-MM-DD)
            selic_daily_rate: Taxa SELIC diária em decimal (ex: 0.00035 para ~13.5% ao ano)
        
        Returns:
            None
        """
        if self.cash > 0:
            interest = self.cash * selic_daily_rate
            self.cash += interest
            
            # Registra como "trade" para tracking
            self.trades.append(Trade(
                date=date,
                ticker='SELIC',
                action='INTEREST',
                shares=0,
                price=0.0,
                commission=0.0,
                total_cost=-interest,  # Negativo = entrada de cash
                reason='SELIC_YIELD'
            ))

    def summary(self) -> Dict:
        """
        Retorna resumo geral do portfólio.
        """
        total_return = ((self.total_value / self.initial_capital) - 1) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': self.total_value,
            'cash': self.cash,
            'positions_value': self.positions_value,
            'num_positions': self.num_positions,
            'total_return_pct': total_return,
            'total_return_brl': self.total_value - self.initial_capital,
            'exposure_pct': self.exposure,
            'num_trades': len(self.trades),
        }
    

# ============ TESTE ============

if __name__ == "__main__":
    print("🧪 TESTE DO MÓDULO PORTFOLIO")
    print("="*70)
    
    # Cria portfólio
    pf = Portfolio(initial_capital=50_000_000)  # R$ 50M
    
    print(f"Capital inicial: R$ {pf.initial_capital:,.2f}")
    print(f"Cash: R$ {pf.cash:,.2f}")
    print(f"Valor total: R$ {pf.total_value:,.2f}\n")
    
    # Simula compras
    print("Simulando compras...")
    
    trades = [
        ('PETR4.SA', 30.50, 8.0, '2024-01-02'),  # 8% do portfólio
        ('VALE3.SA', 65.20, 8.0, '2024-01-02'),
        ('ITUB4.SA', 28.90, 6.0, '2024-01-02'),
    ]
    
    for ticker, price, target_pct, date in trades:
        trade = pf.buy(ticker, price, target_pct, date, reason="INITIAL")
        if trade:
            print(f"  ✅ {trade.action} {trade.shares} {ticker} @ R${trade.price:.2f} "
                  f"(custo: R${trade.total_cost:,.2f})")
        else:
            print(f"  ❌ Não foi possível comprar {ticker}")
    
    # Atualiza preços
    print(f"\nAtualizando preços...")
    new_prices = {
        'PETR4.SA': 32.00,  # +4.9%
        'VALE3.SA': 67.50,  # +3.5%
        'ITUB4.SA': 27.80,  # -3.8%
    }
    
    pf.update_prices(new_prices)
    
    # Posições
    print(f"\n📊 Posições atuais:")
    positions_df = pf.get_positions_summary()
    print(positions_df.to_string())
    
    # Resumo
    print(f"\n📈 Resumo do portfólio:")
    summary = pf.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")