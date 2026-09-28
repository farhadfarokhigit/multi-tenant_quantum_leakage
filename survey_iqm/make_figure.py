"""
Boundary survey on IQM Garnet, same measurement as boundary_survey,
a different vendor's hardware.

Only the conditional spread panel, every remaining boundary plotted
regardless of whether it clears the shot-noise floor, with that floor
drawn as a reference line rather than used to filter points, so it's a
judgment call, not a cutoff made silently in code.

The 1-0 boundary is excluded, at the user's request.
"""

import csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DET = "#1F4E79"

SHOTS = 1000

rows = [{k: (float(v) if k not in ("victim", "probe") else int(v))
         for k, v in r.items()} for r in csv.DictReader(open("survey_data_iqm.csv"))]
rows = [r for r in rows if not (r["victim"] == 1 and r["probe"] == 0)]
rows.sort(key=lambda r: -r["zeta_hz"])
x = np.arange(len(rows))
g = np.array([r["gamma"] for r in rows])
c = np.array([r["contrast"] for r in rows])
gi = g / c
det = np.array([bool(r["detected"]) for r in rows])
lab = [f"{r['victim']}\u2013{r['probe']}" for r in rows]

sigma_lin = 1.0 / math.sqrt(SHOTS)
sigma_c = sigma_lin / math.sqrt(2.0)
dtheta = 2.0 * np.arcsin(np.clip(gi, -1, 1))
sigma_dtheta = math.sqrt(2.0) * sigma_lin / c
sigma_g = np.sqrt((gi * sigma_c)**2 +
                   (0.5 * c * np.abs(np.cos(dtheta / 2.0)) * sigma_dtheta)**2)

floor = 3.0 / math.sqrt(SHOTS)

fig, ax = plt.subplots(figsize=(4.4, 3.1))
ax.errorbar(x, g, yerr=sigma_g, fmt="^", ms=3.5,
            color=DET, capsize=1.5, lw=0.8, elinewidth=0.7)
ax.set_ylim(0, 0.4)
ax.set_ylabel(r"conditional spread $\Gamma$")
ax.set_xticks(x)
ax.set_xticklabels(lab, rotation=90, fontsize=6.5)
ax.tick_params(axis="y", labelsize=7.5)
ax.set_xlabel("boundary (victim\u2013probe)", fontsize=8)
ax.set_xlim(-0.8, len(x) - 0.2)
ax.grid(True, alpha=0.3, lw=0.5)

fig.tight_layout()
fig.savefig("iqm_survey.pdf")
fig.savefig("iqm_survey.png", dpi=200)
print(f"above shot-noise floor: {det.sum()}/{len(rows)} (floor = {floor:.4f})")
print(f"Gamma, all {len(rows)} boundaries: median {np.median(g):.4f}, "
      f"range {g.min():.4f} to {g.max():.4f}")
