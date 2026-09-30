import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scr import aggregate, compute_scr


def test_hand_calculation_two_modules():
    # sqrt(100^2 + 50^2 + 2*0.25*100*50) = sqrt(15000)
    assert aggregate({"Market": 100, "Life": 50}) == pytest.approx(15000 ** 0.5)


def test_single_module_equals_its_charge():
    assert aggregate({"Market": 123.0}) == pytest.approx(123.0)


def test_diversification_never_increases_capital():
    r = compute_scr({"Market": 800, "Default": 120, "Life": 300, "Health": 40})
    assert r["BSCR"] <= r["sum_of_modules"]
    assert r["diversification_benefit"] >= 0


def test_scr_adds_op_risk_and_subtracts_adjustment():
    r = compute_scr({"Market": 100}, op_risk=10, adjustment=25)
    assert r["SCR"] == pytest.approx(100 + 10 - 25)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate({"Market": -1})
    with pytest.raises(ValueError):
        aggregate({"Crypto": 1})
