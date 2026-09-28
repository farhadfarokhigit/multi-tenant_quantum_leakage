"""
Multi-boundary sweep.

The referee objection this answers is "how representative is one pair?".  Each
boundary costs four circuits, so twenty boundaries is one short job.  Output is
the distribution of the boundary rate zeta, the conditional spread Gamma, the
single-shot leakage and the adversary shot budget across the device.

Cost: ~20 boundaries x 4 circuits x 1000 shots ~ 30 s of processor time.
"""

from __future__ import annotations
import argparse, csv, math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# With H_cross = zeta * Z_a Z_b the probe sees H = (-1)^x * zeta * Z_b, so the
# two victim states rotate it by +-2*zeta*tau and the phase gap is
# dtheta = 4*zeta*tau. The slope of dtheta against tau is therefore 4*zeta.
PHASE_PER_ZETA = 4.0

GRAN = 16


def qprop(backend, q, which, default=100e-6):
    try:
        v = getattr(backend.qubit_properties(q), which)
        if v:
            return float(v)
    except Exception:
        pass
    try:
        return float(getattr(backend.properties(), which)(q))
    except Exception:
        return default


def pick_boundaries(backend, n, distance=1):
    """Disjoint victim/probe pairs at the requested graph distance, ranked by
    probe coherence.  Disjointness matters: reusing a qubit as probe for one
    boundary and victim for another would correlate the samples."""
    import rustworkx as rx
    g = backend.coupling_map.graph.to_undirected()
    lengths = dict(rx.all_pairs_dijkstra_path_lengths(g, lambda _: 1.0))
    cands = []
    for p in range(backend.num_qubits):
        for v, d in lengths.get(p, {}).items():
            if int(d) == distance:
                cands.append((min(qprop(backend, p, "t2"),
                                  qprop(backend, int(v), "t1") / 3.0), p, int(v)))
    cands.sort(reverse=True)
    used, out = set(), []
    for _, p, v in cands:
        if p in used or v in used:
            continue
        used.update((p, v))
        out.append((v, p))
        if len(out) == n:
            break
    return out


def circuits_for(victim, probe, nq, delay_dt):
    out, meta = [], []
    for vs in ("0", "1"):
        for basis in ("X", "Y"):
            qc = QuantumCircuit(nq, 1)
            if vs == "1":
                qc.x(victim)
            qc.h(probe)
            qc.barrier([victim, probe])
            qc.delay(delay_dt, victim, unit="dt")
            qc.delay(delay_dt, probe, unit="dt")
            qc.barrier([victim, probe])
            if basis == "Y":
                qc.sdg(probe)
            qc.h(probe)
            qc.measure(probe, 0)
            out.append(qc); meta.append((victim, probe, vs, basis))
    return out, meta


def run(circuits, backend, args, shots):
    isa = generate_preset_pass_manager(optimization_level=0,
                                       backend=backend).run(circuits)
    if args.dry_run:
        from qiskit_aer import AerSimulator
        r = AerSimulator.from_backend(backend).run(isa, shots=shots).result()
        return [r.get_counts(i) for i in range(len(isa))]
    from qiskit_ibm_runtime import SamplerV2
    s = SamplerV2(mode=backend)
    s.options.twirling.enable_gates = False
    s.options.twirling.enable_measure = False
    s.options.dynamical_decoupling.enable = False
    s.options.default_shots = shots
    job = s.run(isa, shots=shots)
    print("job id", job.job_id())
    return [list(r.data.values())[0].get_counts() for r in job.result()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--n-boundaries", type=int, default=20)
    ap.add_argument("--distance", type=int, default=1)
    ap.add_argument("--out", default="survey_data.csv")
    args = ap.parse_args()

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService(name=args.account, channel=args.channel,
                                       instance=args.instance).backend(args.backend)

    dt_ns = backend.target.dt * 1e9
    nq = backend.num_qubits
    pairs = pick_boundaries(backend, args.n_boundaries, args.distance)
    print(f"backend {backend.name}, {len(pairs)} disjoint boundaries at distance "
          f"{args.distance}")

    circuits, meta, windows = [], [], {}
    for v, p in pairs:
        tau = min(qprop(backend, p, "t2"), qprop(backend, v, "t1") / 3.0)
        d = max(GRAN, (int(round(tau * 1e9 / dt_ns)) // GRAN) * GRAN)
        windows[(v, p)] = d
        c, m = circuits_for(v, p, nq, d)
        circuits += c; meta += m

    print(f"{len(circuits)} circuits, {args.shots} shots each")
    counts = run(circuits, backend, args, args.shots)

    ev = {}
    for c, k in zip(counts, meta):
        ev[k] = 2 * c.get("0", 0) / args.shots - 1

    rows = []
    floor = 3.0 / math.sqrt(args.shots)
    for v, p in pairs:
        ex0, ey0 = ev[(v, p, "0", "X")], ev[(v, p, "0", "Y")]
        ex1, ey1 = ev[(v, p, "1", "X")], ev[(v, p, "1", "Y")]
        c0, c1 = math.hypot(ex0, ey0), math.hypot(ex1, ey1)
        dth = math.atan2(ey0, ex0) - math.atan2(ey1, ex1)
        dth = (dth + math.pi) % (2 * math.pi) - math.pi
        tau_s = windows[(v, p)] * dt_ns * 1e-9
        G = 0.5 * (c0 + c1) * abs(math.sin(dth / 2))
        rows.append(dict(victim=v, probe=p, tau_us=tau_s * 1e6,
                         contrast=0.5 * (c0 + c1),
                         zeta_hz=abs(dth) / (2 * math.pi * tau_s * PHASE_PER_ZETA),
                         gamma=G, bits=math.log2(1 + G),
                         nstar=2 * math.log(100) / max(G, 1e-9) ** 2,
                         detected=int(G > floor)))

    print(f"\n{'victim':>7}{'probe':>7}{'tau(us)':>9}{'contrast':>10}"
          f"{'zeta/2pi(Hz)':>14}{'Gamma':>9}{'bits':>8}{'N*':>9}")
    print("-" * 73)
    for r in sorted(rows, key=lambda a: -a["gamma"]):
        print(f"{r['victim']:7d}{r['probe']:7d}{r['tau_us']:9.1f}"
              f"{r['contrast']:10.3f}{r['zeta_hz']:14.0f}{r['gamma']:9.4f}"
              f"{r['bits']:8.4f}{r['nstar']:9.0f}"
              + ("" if r["detected"] else "   (below floor)"))

    det = [r for r in rows if r["detected"]]
    z = np.array([r["zeta_hz"] for r in det]); G = np.array([r["gamma"] for r in det])
    print(f"\ndetected on {len(det)} of {len(rows)} boundaries "
          f"(shot-noise floor on Gamma = {floor:.3f})")
    if len(det):
        print(f"zeta/2pi  median {np.median(z):.0f} Hz, range "
              f"{z.min():.0f} to {z.max():.0f} Hz")
        print(f"Gamma     median {np.median(G):.3f}, range "
              f"{G.min():.3f} to {G.max():.3f}")
        print(f"N*        median {2*math.log(100)/np.median(G)**2:.0f} shots, "
              f"worst boundary {2*math.log(100)/G.max()**2:.0f} shots")
        print("\nThis is the figure that answers 'how representative is one pair'.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
