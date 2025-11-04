# tests/test_utils.py
"""
Testes unitários para yfinance_utils.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yfinance_utils import safe_div, validate_snapshot


def test_safe_div():
    """Testa divisão segura"""
    assert safe_div(10, 2) == 5.0
    assert safe_div(10, 0) is None
    assert safe_div(None, 2) is None
    assert safe_div(10, None) is None
    print("✅ safe_div: OK")


def test_validate_snapshot():
    """Testa validação de snapshot"""
    
    # Snapshot válido
    valid_snap = {
        "price": 30.5,
        "market_cap": 1e9,
        "pe": 15.0,
        "gross_margin": 0.4,
        "net_margin": 0.2,
        "roe": 0.15,
    }
    is_valid, missing_crit, missing_des = validate_snapshot(valid_snap)
    assert is_valid is True
    assert len(missing_crit) == 0
    print("✅ validate_snapshot (válido): OK")
    
    # Snapshot inválido (falta preço)
    invalid_snap = {
        "market_cap": 1e9,
        "pe": 15.0,
    }
    is_valid, missing_crit, missing_des = validate_snapshot(invalid_snap)
    assert is_valid is False
    assert len(missing_crit) > 0
    print("✅ validate_snapshot (inválido): OK")


if __name__ == "__main__":
    test_safe_div()
    test_validate_snapshot()
    print("\n🎉 Todos os testes passaram!")
    
