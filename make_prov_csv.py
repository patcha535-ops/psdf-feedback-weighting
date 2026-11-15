#!/usr/bin/env python3
"""make_prov_csv.py -- extracts the 77 provincial 2024 values (visitors, foreign visitors, receipts) from the
compiled data table (Table 1 of the data PDF, taken from the MOTS revised 28 Aug 2025 workbook) into prov.csv and checks
them against the published national totals.   Usage: python make_prov_csv.py Data_tourism.pdf prov.csv  (needs pdftotext)"""
import csv
import re
import subprocess
import sys

pdf, out = sys.argv[1], sys.argv[2]
txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
t1 = txt.split("\u0e15\u0e32\u0e23\u0e32\u0e07\u0e17\u0e35\u0e48 2")[0]   # Thai heading "Table 2": stop before it
rx = re.compile(r"^\s*([A-Za-z][A-Za-z .()\-]*?)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+\.\d+)\s+([\d,]+\.\d+)\s+([\d,]+\.\d+)\s+([\d,]+)\s+([\d.]+)\s*$")
f = lambda s: float(s.replace(",", ""))
rows = [(m.group(1).strip(), [f(m.group(i)) for i in range(2, 9)]) for m in map(rx.match, t1.splitlines()) if m]
assert len(rows) == 77 and len({n for n, _ in rows}) == 77, len(rows)          # region subtotal rows carry "(17)" etc. and are skipped
assert int(sum(v[0] for _, v in rows)) == 353556836                             # published national totals
assert int(sum(v[2] for _, v in rows)) == 83178986
assert abs(sum(v[3] for _, v in rows) - 2924506.59) < 0.05
assert all(abs(v[1] + v[2] - v[0]) < 0.5 and abs(v[4] + v[5] - v[3]) < 0.02 for _, v in rows)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["province", "visitors_total", "visitors_foreign", "receipts_total"])
    for n, v in rows:
        w.writerow([n, int(v[0]), int(v[2]), v[3]])
print("wrote", out, "- 77 provinces, totals reconcile with the published national row")
