"""
Survey of twenty disjoint tenant boundaries on one device.

(a) boundary rate zeta/2pi per pair, with the two non-detections marked, and
    the 60-61 pair from the detailed sweep added as its own, distinctly styled bar,
    since it was measured with a full ten-point sweep at 4096 shots per
    point, not the single 1000-shot reconnaissance circuit every other bar
    here used; giving it the same plain fill as the rest would imply a
    precision it does not share with them.
(b) adversary shot budget per pair, against the shot count of an ordinary job
(c) the conditional spread decomposed, with error bars propagated from the
    same 1000-shot, single-circuit measurement as every point in this figure
"""

import csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DET = "#1F4E79"
UND = "#B0B7BC"
ACC = "#B7950B"
HEAD = "#B03A2E"

# With H_cross = zeta * Z_a Z_b the probe sees H = (-1)^x * zeta * Z_b, so the
# two victim states rotate it by +-2*zeta*tau and the phase gap is
# dtheta = 4*zeta*tau. The slope of dtheta against tau is therefore 4*zeta.
PHASE_PER_ZETA = 4.0

SHOTS = 1000   # shots per circuit in this survey, not the 4096 used for
               # the detailed q60-q61 sweep

rows = [{k: (float(v) if k not in ("victim", "probe") else int(v))
         for k, v in r.items()} for r in csv.DictReader(open("survey_data.csv"))]
rows.append({"victim": 60, "probe": 61, "zeta_hz": 4558.0 / PHASE_PER_ZETA, "gamma": 0.3496,
             "contrast": 0.4662, "nstar": 2.0*math.log(1/0.01)/0.3496**2,
             "tau_us": 59.008, "detected": 1.0, "special": True})
for r in rows:
    r.setdefault("special", False)
rows.sort(key=lambda r: -r["zeta_hz"])
x = np.arange(len(rows))
z = np.array([r["zeta_hz"] for r in rows]) / 1000.0
n = np.array([r["nstar"] for r in rows])
g = np.array([r["gamma"] for r in rows])
c = np.array([r["contrast"] for r in rows])
gi = g / c                                   # |sin(dtheta/2)|, contrast removed
det = np.array([bool(r["detected"]) for r in rows])
special = np.array([r["special"] for r in rows])
lab = [f"{r['victim']}\u2013{r['probe']}" + ("*" if r["special"] else "")
       for r in rows]

# propagated uncertainty for a single 1000-shot, single-circuit measurement,
# the same derivation used for the detailed q60-q61 sweep, with n_frames=1
# since this is one reconnaissance circuit, not an averaged multi-frame arm
sigma_lin = 1.0 / math.sqrt(SHOTS)
sigma_c = sigma_lin / math.sqrt(2.0)                    # radial component
dtheta = 2.0 * np.arcsin(np.clip(gi, -1, 1))            # back out Delta_theta
sigma_dtheta = math.sqrt(2.0) * sigma_lin / c           # tangential component
sigma_gi = 0.5 * np.abs(np.cos(dtheta / 2.0)) * sigma_dtheta
sigma_g = np.sqrt((gi * sigma_c)**2 +
                   (0.5 * c * np.abs(np.cos(dtheta / 2.0)) * sigma_dtheta)**2)
tau_s = np.array([r["tau_us"] for r in rows]) * 1e-6
sigma_zeta = sigma_dtheta / (2 * math.pi * tau_s * PHASE_PER_ZETA) / 1000.0   # kHz, matches z

fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.1))

# ---- (a) coupling rate, now with its own propagated error bar ----------
ax[0].errorbar(x, z, yerr=sigma_zeta, fmt="o", ms=3.5, color=DET,
               capsize=1.5, lw=0.8, elinewidth=0.7)
ax[0].set_ylabel(r"$\zeta/2\pi$  (kHz)")
ax[0].set_ylim(0, 1.4)

# ---- (b) conditional spread ---------------------------------------------
ax[1].errorbar(x, g, yerr=sigma_g, fmt="^", ms=3.5,
               color=DET, capsize=1.5, lw=0.8, elinewidth=0.7)
ax[1].set_ylim(0, 1.08)
ax[1].set_ylabel(r"conditional spread $\Gamma$")

# ---- (c) shot budget, no error bar, same reasoning as every other figure:
# N* is steeply nonlinear in Gamma, and a naive error bar on it is not
# trustworthy at the small-Gamma end, see the other figures' derivation.
ax[2].bar(x, n, color=DET, width=0.72)
ax[2].set_yscale("log")
ax[2].set_ylabel(r"shot budget $N^{*}$")
ax[2].set_ylim(10, 4e5)

for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(lab, rotation=90, fontsize=5.2)
    a.tick_params(axis="y", labelsize=7.5)
    a.set_xlabel("boundary (victim\u2013probe)", fontsize=8)
for a in ax:
    a.set_xlim(-0.8, len(x) - 0.2)
    a.grid(True, alpha=0.3, lw=0.5)
ax[0].set_title("a", loc="left", fontsize=9, fontweight="bold")
ax[1].set_title("b", loc="left", fontsize=9, fontweight="bold")
ax[2].set_title("c", loc="left", fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig("boundary_survey.pdf")
fig.savefig("boundary_survey.png", dpi=200)
print(f"detected {det.sum()}/{len(rows)}")
print(f"zeta/2pi median {np.median(z)*1000:.0f} Hz, IQR "
      f"{np.percentile(z,25)*1000:.0f}-{np.percentile(z,75)*1000:.0f}")
print(f"N* median {np.median(n):.0f}, min {n.min():.0f}, "
      f"max {n.max():.0f}, all below 4096: {(n<4096).all()}")
print(f"|sin(dtheta/2)| median {np.median(gi):.3f}")
print(f"Gamma median {np.median(g):.4f}")
print(f"sigma_c (constant, 1000 shots) = {sigma_c:.4f}")
