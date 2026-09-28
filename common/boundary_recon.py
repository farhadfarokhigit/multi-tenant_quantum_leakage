"""
Step 1 reconnaissance.  Four circuits, ~2 seconds of QPU time.

Answers one question.  Is there a measurable conditional phase on this pair?

Probe idles in |+> for tau = T2 of the probe, which is where the measured
spread delta(tau) = exp(-tau/T2)|sin(2*zeta*tau)| peaks near tau = T2.  Victim held in |0> then |1>.
Probe read in X and Y so the phase is recovered without a reference.

Usage
    python recon.py --dry-run
    python recon.py --backend ibm_torino
    python recon.py --backend ibm_torino --pair 42 43
"""

from __future__ import annotations
import argparse, math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

SHOTS = 1000
GRANULARITY = 16
# With H_cross = zeta * Z_a Z_b the probe sees H = (-1)^x * zeta * Z_b, so the
# two victim states rotate it by +-2*zeta*tau and the phase gap is
# dtheta = 4*zeta*tau. The slope of dtheta against tau is therefore 4*zeta.
PHASE_PER_ZETA = 4.0


def probe_t2(backend, q):
    """T2 in seconds, from whichever API the backend exposes."""
    try:
        t2 = backend.qubit_properties(q).t2
        if t2:
            return float(t2)
    except Exception:
        pass
    try:
        return float(backend.properties().t2(q))
    except Exception:
        return 100e-6


def victim_t1(backend, q) -> float:
    """T1 in seconds, from whichever API the backend exposes."""
    try:
        t1 = backend.qubit_properties(q).t1
        if t1:
            return float(t1)
    except Exception:
        pass
    try:
        return float(backend.properties().t1(q))
    except Exception:
        return 100e-6


def best_pair(backend, distance=1):
    """Pick the highest-T2 qubit that has a partner at the requested distance."""
    import rustworkx as rx
    g = backend.coupling_map.graph.to_undirected()
    lengths = dict(rx.all_pairs_dijkstra_path_lengths(g, lambda _: 1.0))
    cands = []
    for p in range(backend.num_qubits):
        partners = [v for v, d in lengths.get(p, {}).items() if int(d) == distance]
        if partners:
            cands.append((probe_t2(backend, p), p, int(partners[0])))
    if not cands:
        raise RuntimeError(f"no pair at distance {distance}")
    t2, probe, victim = max(cands)
    return victim, probe, t2


def circuits(victim, probe, n_qubits, delay_dt):
    out, meta = [], []
    for vs in ("0", "1"):
        for basis in ("X", "Y"):
            qc = QuantumCircuit(n_qubits, 1)
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
            out.append(qc); meta.append((vs, basis))
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--account", default=None,
                    help="named saved account, e.g. unimelb")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--distance", type=int, default=1)
    ap.add_argument("--pair", nargs=2, type=int, default=None)
    args = ap.parse_args()

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService(name=args.account,
                                       channel=args.channel,
                                       instance=args.instance).backend(args.backend)

    if args.pair:
        victim, probe = args.pair
        t2 = probe_t2(backend, probe)
    else:
        victim, probe, t2 = best_pair(backend, args.distance)

    # The victim must survive the window too.  If it relaxes partway through,
    # the conditional phase is randomized and the excited arm loses contrast,
    # which is what happened on q88/q89.  Cap by victim T1 as well.
    t1v = victim_t1(backend, victim)
    tau_s = min(t2, t1v / 3.0)
    if tau_s < t2:
        print(f"window capped by victim T1 = {t1v*1e6:.1f} us")

    dt_ns = backend.target.dt * 1e9
    tau_ns = tau_s * 1e9
    delay_dt = max(GRANULARITY,
                   (int(round(tau_ns / dt_ns)) // GRANULARITY) * GRANULARITY)

    print(f"backend {backend.name}   victim q{victim}   probe q{probe}")
    print(f"probe T2 = {t2*1e6:.1f} us   idle window = {delay_dt*dt_ns/1000:.1f} us")

    qcs, meta = circuits(victim, probe, backend.num_qubits, delay_dt)
    isa = generate_preset_pass_manager(optimization_level=0, backend=backend).run(qcs)

    if args.dry_run:
        from qiskit_aer import AerSimulator
        res = AerSimulator.from_backend(backend).run(isa, shots=args.shots).result()
        counts = [res.get_counts(i) for i in range(len(isa))]
    else:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)          # job mode, Open plan safe
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.dynamical_decoupling.enable = False
        job = sampler.run(isa, shots=args.shots)
        print("job id", job.job_id())
        out = job.result()
        counts = [list(r.data.values())[0].get_counts() for r in out]

    ev = {}
    for c, (vs, basis) in zip(counts, meta):
        ev[(vs, basis)] = 2.0 * c.get("0", 0) / args.shots - 1.0

    th0 = math.atan2(ev[("0", "Y")], ev[("0", "X")])
    th1 = math.atan2(ev[("1", "Y")], ev[("1", "X")])
    c0 = math.hypot(ev[("0", "X")], ev[("0", "Y")])
    c1 = math.hypot(ev[("1", "X")], ev[("1", "Y")])
    dth = (th0 - th1 + math.pi) % (2 * math.pi) - math.pi
    delta = 0.5 * (c0 + c1) * abs(math.sin(dth / 2))

    floor = 3.0 / math.sqrt(args.shots)            # ~3 sigma on the spread
    tau_meas = delay_dt * dt_ns * 1e-9
    # H_cross = zeta Z_a Z_b puts the probe at -2*zeta with the victim in |0>
    # and +2*zeta with it in |1>, so the accumulated phase gap is 4*zeta*tau.
    zeta = abs(dth) / (2 * math.pi * PHASE_PER_ZETA * tau_meas) if tau_meas else float("nan")

    print(f"\ncontrast  {c0:.3f} / {c1:.3f}   (low contrast means tau > T2)")
    if min(c0, c1) < 5.0 / math.sqrt(args.shots):
        print("  WARNING contrast at the noise floor in one arm.  The phase "
              "from atan2 is meaningless.  Shorten the window and rerun.")
        print("  Note the contrast ASYMMETRY is itself a leakage channel.")
    print(f"phase gap {dth:+.4f} rad")
    print(f"spread    {delta:.4f}   shot-noise floor {floor:.4f}")
    print(f"implied   zeta/2pi = {zeta/1e3:.3f} kHz")
    if delta > floor:
        n_star = 2 * math.log(100.0) / max(delta, 1e-9) ** 2
        print(f"\nGO.  Signal is above the floor.  Adversary shot budget "
              f"~{n_star:.0f} at 99 percent confidence.")
    else:
        print("\nNO GO on this pair.  Try --distance 2, or --pair with a "
              "different couple, before increasing shots.")


if __name__ == "__main__":
    main()
