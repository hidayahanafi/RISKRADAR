"""Run the example and write an Excel report:  python solvency2/report.py"""
from pathlib import Path

import pandas as pd

from scr import BSCR_CORR, MODULES, compute_scr

# Illustrative figures (EUR millions) for a fictional savings insurer.
CHARGES = {"Market": 800.0, "Default": 120.0, "Life": 300.0, "Health": 40.0, "Non-life": 0.0}
OP_RISK, ADJUSTMENT = 60.0, 150.0

res = compute_scr(CHARGES, OP_RISK, ADJUSTMENT)
out = Path(__file__).with_name("scr_report.xlsx")
with pd.ExcelWriter(out) as xl:
    pd.DataFrame({"Module": MODULES, "Charge (EURm)": [CHARGES[m] for m in MODULES]}).to_excel(xl, sheet_name="Inputs", index=False)
    BSCR_CORR.to_excel(xl, sheet_name="Correlation")
    pd.Series(res, name="EURm").round(1).to_frame().to_excel(xl, sheet_name="Result")
print({k: round(v, 1) for k, v in res.items()})
print(f"Saved {out.name}")
