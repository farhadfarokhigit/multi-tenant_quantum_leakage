"""
Build the manuscript figure from data_kingston_q60_q61.csv.

Three panels.
  (a) conditional phase gap against idle window, with weighted linear fits
  (b) conditional spread, with the fitted C(tau)|sin(dtheta/2)| envelope,
      dtheta = 4*zeta*tau
  (c) adversary shot budget, predicted and measured, on a log axis

Every point is fit and plotted the same way, including low-contrast ones.
The weighted fit already down-weights a point as its contrast falls, since
sigma_dtheta grows as contrast shrinks, so an unreliable point earns less
influence on the fit automatically rather than being dropped by a separate,
hand-picked cutoff. No point is excluded or marked differently from any
other. The one hard requirement this relies on is contrast staying strictly
positive everywhere in the data, since sigma_dtheta divides by it directly
and the T2* fit below takes its logarithm; a row with contrast exactly zero
would need handling this script does not attempt.
"""

import csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "data_kingston_q60_q61.csv"
# With H_cross = zeta * Z_a Z_b the probe sees H = (-1)^x * zeta * Z_b, so the
# two victim states rotate it by +-2*zeta*tau and the phase gap is
# dtheta = 4*zeta*tau. The slope of dtheta against tau is therefore 4*zeta.
PHASE_PER_ZETA = 4.0
SHOTS = 4096
ETA = 0.01          # 1 - confidence, matches the 99% used throughout the Letter
ARMS = [("bare", "no frame", "#c0392b", "o"),
        ("pfr",  "Pauli frame", "#1f77b4", "s")]


def load():
    rows = {a: [] for a, _, _, _ in ARMS}
    with open(CSV) as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if r["arm"] not in rows:
                continue
            rows[r["arm"]].append({k: float(v) if k != "arm" else v
                                   for k, v in r.items()})
    return {k: {c: np.array([d[c] for d in v]) for c in v[0] if c != "arm"}
            for k, v in rows.items()}


def sigma_dtheta(contrast, shots, n_frames):
    """Propagated error on the phase gap. Two arms, two quadratures."""
    return math.sqrt(2.0) / np.sqrt(shots * n_frames) / contrast


def sigma_contrast(shots, n_frames):
    """Propagated error on the reported contrast, the average of two
    independently estimated branch contrasts. Unlike the phase, the radial
    component of a 2D estimate with equal per-axis noise has an uncertainty
    that does not grow as the contrast itself shrinks, it is a constant set
    only by the shot count."""
    sigma_lin = 1.0 / math.sqrt(shots * n_frames)
    return sigma_lin / math.sqrt(2.0)


def sigma_gamma(contrast, dtheta, sig_c, sig_dth):
    """Propagate sigma_c and sigma_dtheta through Gamma = C|sin(dtheta/2)|."""
    term_c = np.abs(np.sin(dtheta / 2.0)) * sig_c
    term_dth = 0.5 * contrast * np.abs(np.cos(dtheta / 2.0)) * sig_dth
    return np.sqrt(term_c**2 + term_dth**2)


def wfit(t, y, s):
    w = 1.0 / s**2
    S, Sx, Sy = w.sum(), (w*t).sum(), (w*y).sum()
    Sxx, Sxy = (w*t*t).sum(), (w*t*y).sum()
    d = S*Sxx - Sx**2
    slope = (S*Sxy - Sx*Sy) / d
    inter = (Sxx*Sy - Sx*Sxy) / d
    return slope, math.sqrt(S/d), inter


def main():
    D = load()
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.3))
    tgrid = np.linspace(0, 100, 400)
    summary = []
    t2stars = {}

    for arm, label, colour, marker in ARMS:
        d = D[arm]
        t = d["tau_ns"] / 1000.0                      # microseconds
        nfr = 2 if arm == "pfr" else 1
        s = sigma_dtheta(d["contrast"], SHOTS, nfr)
        sig_c = sigma_contrast(SHOTS, nfr)
        sig_gamma = sigma_gamma(d["contrast"], d["dtheta_rad"], sig_c, s)

        slope, sslope, inter = wfit(t, d["dtheta_rad"], s)
        zeta = slope * 1e6 / (2*math.pi*PHASE_PER_ZETA)   # Hz
        szeta = sslope * 1e6 / (2*math.pi*PHASE_PER_ZETA)
        summary.append((label, zeta, szeta, slope/sslope, inter))

        # (a) phase gap
        ax[0].errorbar(t, d["dtheta_rad"], yerr=s, fmt=marker,
                       ms=4, color=colour, label=label, capsize=2, lw=1)
        ax[0].plot(tgrid, inter + slope*tgrid, "-", color=colour, lw=1, alpha=0.8)

        # envelope, Gamma_theory(tau) = C(tau)|sin(dtheta/2)|, from this
        # arm's own fitted zeta above and its own contrast decay fit here.
        # This is the one smooth theoretical curve for this arm; everything
        # derived from it, in panel (b) and panel (c) both, comes from this
        # same fit rather than from the noisy per-point measured Gamma.
        slope_c, inter_c = np.polyfit(t, np.log(d["contrast"]), 1)
        t2star = -1.0 / slope_c
        t2stars[arm] = t2star
        env = np.exp(inter_c) * np.exp(-tgrid/t2star) * np.abs(np.sin(PHASE_PER_ZETA*math.pi*zeta*tgrid*1e-6))

        # (b) conditional spread: measured markers, fitted theory curve
        ax[1].errorbar(t, d["delta"], yerr=sig_gamma, fmt=marker, ms=4,
                       color=colour, label=label, capsize=2, lw=1)
        ax[1].plot(tgrid, env, color=colour, lw=0.9, alpha=0.7)

        # (c) shot budget: N* = 2 ln(1/eta) / Gamma^2 throughout, applied
        # to the fitted envelope for the line and to each measured Gamma
        # for the markers, the same formula both places, no bootstrap.
        # env is exactly zero at tau=0, where N* is mathematically
        # infinite; that single point is a real feature of the formula,
        # not an error, so the resulting division-by-zero warning is
        # expected and suppressed rather than silenced blindly everywhere.
        with np.errstate(divide="ignore"):
            nstar_theory = 2.0*math.log(1/ETA) / env**2
        nstar_meas = 2.0*math.log(1/ETA) / d["delta"]**2
        ax[2].plot(tgrid, nstar_theory, "-", color=colour, lw=1)
        ax[2].plot(t, nstar_meas, marker, ms=4, color=colour, label=label,
                   ls="none")

    t2star = t2stars["bare"]   # kept for the printed T2* line below

    ax[0].set_xlabel(r"idle window $\tau$ ($\mu$s)")
    ax[0].set_ylabel(r"phase gap $\Delta\theta$ (rad)")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    ax[0].set_title("(a)", loc="left", fontsize=9)
    ax[0].grid(True, alpha=0.3, lw=0.5)

    ax[1].set_xlabel(r"idle window $\tau$ ($\mu$s)")
    ax[1].set_ylabel(r"conditional spread $\Gamma$")
    ax[1].legend(fontsize=6.5, frameon=False)
    ax[1].set_title("(b)", loc="left", fontsize=9)
    ax[1].grid(True, alpha=0.3, lw=0.5)

    ax[2].set_yscale("log")
    ax[2].set_ylim(top=1e10)
    ax[2].set_xlabel(r"idle window $\tau$ ($\mu$s)")
    ax[2].set_ylabel(r"shot budget $N^{*}$ (99% conf.)")
    ax[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    ax[2].set_title("(c)", loc="left", fontsize=9)
    ax[2].grid(True, alpha=0.3, lw=0.5)

    for a in ax:
        a.tick_params(labelsize=8)
        a.set_xlim(-3, 102)

    fig.tight_layout()
    fig.savefig("boundary_sweep.pdf")
    fig.savefig("boundary_sweep.png", dpi=200)

    print(f"probe T2* from contrast decay = {t2star:.1f} us "
          f"(backend reported T2 = 53.7 us)\n")
    print(f"{'arm':<32}{'zeta (Hz)':>12}{'sigma':>9}{'signif':>9}{'intercept':>11}")
    for label, z, sz, sig, inter in summary:
        print(f"{label:<32}{z:12.1f}{sz:9.1f}{sig:8.1f}s{inter:+11.4f}")
    zb, zp = summary[0][1], summary[1][1]
    print(f"\nsuppression factor {zb/zp:.1f}   residual {100*zp/zb:.1f}% of bare")


if __name__ == "__main__":
    main()
