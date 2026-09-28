"""
Fit r for both arms from fidelity_data.csv and plot survival probability
against sequence length, standard randomized benchmarking decay,
P(m) = A p^m + 1/2, r = (1-p)/2 for a single qubit.
"""
import csv, math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "fidelity_data.csv"
COL = {"bare": "#c0392b", "framed": "#1f77b4"}
LAB = {"bare": "no frame", "framed": "Pauli frame"}


def decay(m, A, p):
    return A * p**m + 0.5


def load():
    rows = list(csv.DictReader(open(CSV)))
    out = {}
    for arm in ("bare", "framed"):
        lengths = sorted(set(int(r["length"]) for r in rows if r["arm"] == arm))
        means, sems = [], []
        for L in lengths:
            vals = [float(r["survival"]) for r in rows
                    if r["arm"] == arm and int(r["length"]) == L]
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        out[arm] = (np.array(lengths), np.array(means), np.array(sems))
    return out


def main():
    data = load()
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    lgrid = np.linspace(1, max(max(data[a][0]) for a in data), 200)
    results = []

    for arm in ("bare", "framed"):
        lengths, means, sems = data[arm]
        popt, pcov = curve_fit(decay, lengths, means, p0=[0.5, 0.99],
                                sigma=sems, absolute_sigma=True)
        A, p = popt
        sigma_p = math.sqrt(pcov[1, 1])
        r = (1 - p) / 2
        sigma_r = sigma_p / 2
        results.append((arm, r, sigma_r, p, sigma_p))

        ax.errorbar(lengths, means, yerr=sems, fmt="o", ms=4, color=COL[arm],
                    label=LAB[arm], capsize=2, lw=1)
        ax.plot(lgrid, decay(lgrid, A, p), "-", color=COL[arm], lw=1, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("sequence length (Cliffords)")
    ax.set_ylabel("survival probability")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig("victim_fidelity.pdf")
    fig.savefig("victim_fidelity.png", dpi=200)

    print(f"{'arm':<10}{'r':>12}{'sigma_r':>12}{'p':>10}{'sigma_p':>10}")
    for arm, r, sigma_r, p, sigma_p in results:
        print(f"{arm:<10}{r:12.5f}{sigma_r:12.5f}{p:10.5f}{sigma_p:10.5f}")

    r_bare, sig_bare = results[0][1], results[0][2]
    r_framed, sig_framed = results[1][1], results[1][2]
    diff = r_framed - r_bare
    sig_diff = math.sqrt(sig_bare**2 + sig_framed**2)
    print(f"\ndifference r_framed - r_bare = {diff:.5f} +/- {sig_diff:.5f} "
          f"({abs(diff)/sig_diff:.1f} sigma)")


if __name__ == "__main__":
    main()
