# Solvency 2 – SCR Aggregation Tool

A small Python tool that shows how an insurer combines its risk capital charges into the **Solvency Capital Requirement (SCR)** under the Solvency 2 standard formula.

## The idea in three lines
1. Each risk module (market, default, life, health, non-life) has its own capital charge.
2. The risks do not all happen at once, so they are combined with a **correlation matrix** instead of simply added up. The gap between the two is the **diversification benefit**.
3. Add operational risk and subtract the loss-absorbing adjustment to get the final SCR.

```
BSCR = sqrt( Σᵢ Σⱼ Corr(i,j) · SCRᵢ · SCRⱼ )
SCR  = BSCR + Op − Adj
```
The correlation matrix is the one from the EU Delegated Regulation 2015/35 (Art. 21).

## Example (illustrative figures, EUR millions)
| Module | Charge |
|---|---|
| Market | 800 |
| Default | 120 |
| Life | 300 |
| Health | 40 |
| Non-life | 0 |

Sum of modules 1,260 → **BSCR 978** (diversification benefit 282) → with Op = 60 and Adj = 150 → **SCR 888**.

## Run
```bash
pip install -r requirements.txt openpyxl pytest
python solvency2/report.py      # writes solvency2/scr_report.xlsx (Inputs, Correlation, Result)
python -m pytest solvency2      # 5 tests, including a hand calculation
```

## Validation
Tests check a hand-computed two-module case (√15,000), that one module alone equals its charge, that diversification never increases capital, the SCR arithmetic, and rejection of invalid inputs.

## Scope and limitations
Top-level aggregation only. A full implementation would also compute each module's own sub-modules (interest rate, equity, lapse, mortality, etc.), and the operational-risk and loss-absorbing adjustments are taken here as inputs. Figures are fictional.
