"""Regenerate every figure, animation, data file, and report.

Figures 4 to 8 cache their ensembles in outputs/cache and are only
re-plotted on reruns; delete that directory to recompute everything.
The reports read the CSV files written by the figure scripts.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig00_schematic.py", "fig01_ceiling.py", "fig02_ip.py",
         "fig03_gradient.py", "fig04_tilt.py", "fig05_hover.py",
         "fig06_team.py", "fig07_visibility.py", "fig08_verification.py",
         "anim01_ceiling.py", "anim02_rain.py", "anim03_ip.py",
         "anim04_dip.py", "make_reports.py"]

t0 = time.time()
for s in ORDER:
    print(f"--- {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{s} failed")
print(f"--- done in {time.time() - t0:.1f} s")
