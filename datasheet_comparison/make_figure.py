"""
Measured leakage on every coupled pair against the numbers the provider reports.

One panel per reported quantity, one point per pair.  The error bars are
propagated from the same 1000-shot, single-circuit measurement as every point in
the boundary survey.
"""

import csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DET = "#1F4E79"

SHOTS = 1000          # shots per circuit in this survey

rows = [{k: (int(v) if k in ("victim", "probe") else
             (float(v) if v != "" else float("nan")))
         for k, v in r.items()} for r in csv.DictReader(open("comparison_data.csv"))]

g = np.array([r["gamma"] for r in rows])
c = np.array([r["contrast"] for r in rows])
det = np.array([bool(r["detected"]) for r in rows])

# propagated uncertainty for a single 1000-shot, single-circuit measurement,
# the same derivation used for the boundary survey
sigma_lin = 1.0 / math.sqrt(SHOTS)
sigma_c = sigma_lin / math.sqrt(2.0)                    # radial component
gi = g / c                                              # |sin(dtheta/2)|, contrast removed
dtheta = 2.0 * np.arcsin(np.clip(gi, -1, 1))            # back out Delta_theta
sigma_dtheta = math.sqrt(2.0) * sigma_lin / c           # tangential component
sigma_g = np.sqrt((gi * sigma_c)**2 +
                  (0.5 * c * np.abs(np.cos(dtheta / 2.0)) * sigma_dtheta)**2)

PANELS = [("gate_error",           "two-qubit gate error",            True),
          ("t2_probe_us",          r"probe $T_2$  ($\mu$s)",          False),
          ("t1_victim_us",         r"victim $T_1$  ($\mu$s)",         False),
          ("readout_error_probe",  "probe readout error",             True),
          ("sx_error_probe",       "probe single-qubit gate error",   True)]

fig, ax = plt.subplots(1, len(PANELS), figsize=(11.2, 2.7), sharey=True)
for a, (key, label, logx) in zip(ax, PANELS):
    x = np.array([r[key] for r in rows])
    ok = ~np.isnan(x) & (x > 0 if logx else True)
    a.errorbar(x[ok], g[ok], yerr=sigma_g[ok], fmt="o", ms=3.0, color=DET,
               capsize=1.2, lw=0.6, elinewidth=0.5)
    if logx:
        a.set_xscale("log")
    a.set_xlabel(label, fontsize=8)
    a.tick_params(labelsize=7.5)
    a.grid(True, alpha=0.3, lw=0.5)
ax[0].set_ylabel(r"conditional spread $\Gamma$", fontsize=8)
ax[0].set_ylim(0, 0.8)
for a, t in zip(ax, "abcdefgh"):
    a.set_title(t, loc="left", fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig("datasheet_comparison.pdf")
fig.savefig("datasheet_comparison.png", dpi=200)
print(f"{len(rows)} coupled pairs, detected {det.sum()}")
print(f"Gamma median {np.median(g):.4f}, range {g.min():.4f} to {g.max():.4f}")
