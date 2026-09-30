"""Solvency 2 standard formula: aggregation of risk modules into the SCR.

    BSCR = sqrt( sum_ij  Corr[i][j] * SCR_i * SCR_j )
    SCR  = BSCR + Op - Adj

Corr is the BSCR correlation matrix of the EU Delegated Regulation 2015/35
(Art. 21). Inputs in the example are illustrative, not real company figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODULES = ["Market", "Default", "Life", "Health", "Non-life"]

BSCR_CORR = pd.DataFrame(
    [
        [1.00, 0.25, 0.25, 0.25, 0.25],
        [0.25, 1.00, 0.25, 0.25, 0.50],
        [0.25, 0.25, 1.00, 0.25, 0.00],
        [0.25, 0.25, 0.25, 1.00, 0.25],
        [0.25, 0.50, 0.00, 0.25, 1.00],
    ],
    index=MODULES,
    columns=MODULES,
)


def aggregate(charges: dict[str, float], corr: pd.DataFrame = BSCR_CORR) -> float:
    """Square-root aggregation of capital charges with a correlation matrix."""
    unknown = set(charges) - set(corr.index)
    if unknown:
        raise ValueError(f"Unknown risk modules: {sorted(unknown)}")
    if any(v < 0 for v in charges.values()):
        raise ValueError("Capital charges must be non-negative")
    if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1):
        raise ValueError("Correlation matrix must be symmetric with a unit diagonal")
    s = pd.Series(charges, dtype=float).reindex(corr.index, fill_value=0.0)
    return float(np.sqrt(s.values @ corr.values @ s.values))


def compute_scr(charges: dict[str, float], op_risk: float = 0.0, adjustment: float = 0.0) -> dict:
    """Return BSCR, the diversification benefit and the final SCR."""
    bscr = aggregate(charges)
    undiversified = float(sum(charges.values()))
    return {
        "sum_of_modules": undiversified,
        "BSCR": bscr,
        "diversification_benefit": undiversified - bscr,
        "operational_risk": op_risk,
        "adjustment": adjustment,
        "SCR": bscr + op_risk - adjustment,
    }
