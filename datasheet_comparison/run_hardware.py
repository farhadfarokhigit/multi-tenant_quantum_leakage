"""
Every coupled pair on one device, against the numbers the provider reports.

The question this answers is whether a tenant could have read the leakage off
the device datasheet.  The conditional spread Gamma is measured on every
neighbouring pair, with the same four circuits per boundary as the survey, and
each pair's measured Gamma is stored next to the calibration numbers the
backend reports for the same qubits at the time of the run: the two-qubit gate
error on the pair, T1 and T2 of both qubits, readout error and single-qubit
gate error.

Each coupled pair is measured once, with the lower-numbered qubit as the victim
and the higher-numbered one as the probe.  The idle window of a pair is the
smaller of the probe T2 and a third of the victim T1, so the probe is not
dephased much beyond its coherence and the victim has not relaxed.

Cost: ~176 boundaries x 4 circuits x 1000 shots ~ 5 min of processor time.
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
TWO_QUBIT_GATES = ("cz", "ecr", "cx")


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


def all_boundaries(backend, n):
    """Every coupled pair once, lower-numbered qubit as victim.  A pair shares
    qubits with its neighbours, but each circuit touches only its own two
    qubits, so the boundaries are measured one at a time and do not interact."""
    edges = sorted({tuple(sorted(e)) for e in backend.coupling_map.get_edges()})
    return edges[:n] if n else edges


def reported(backend, v, p):
    """Calibration numbers the backend reports for this boundary, nan when the
    backend does not report a number."""
    t = backend.target

    def err(name, qargs):
        try:
            e = t[name][qargs].error
            return float("nan") if e is None else float(e)
        except Exception:
            return float("nan")

    def prop(q, which):
        try:
            x = getattr(backend.qubit_properties(q), which)
            return float("nan") if x is None else float(x) * 1e6
        except Exception:
            return float("nan")

    gate = next((g for g in TWO_QUBIT_GATES if g in t.operation_names), None)
    gate_error = float("nan")
    if gate is not None:
        errs = [err(gate, q) for q in ((v, p), (p, v))]
        errs = [e for e in errs if not math.isnan(e)]
        gate_error = float(np.mean(errs)) if errs else float("nan")
    return dict(gate_error=gate_error,
                t1_victim_us=prop(v, "t1"), t2_victim_us=prop(v, "t2"),
                t1_probe_us=prop(p, "t1"), t2_probe_us=prop(p, "t2"),
                readout_error_victim=err("measure", (v,)),
                readout_error_probe=err("measure", (p,)),
                sx_error_victim=err("sx", (v,)),
                sx_error_probe=err("sx", (p,)))


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
    ap.add_argument("--n-boundaries", type=int, default=0,
                    help="only the first N coupled pairs, 0 for all of them")
    ap.add_argument("--out", default="comparison_data.csv")
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
    pairs = all_boundaries(backend, args.n_boundaries)
    print(f"backend {backend.name}, {len(pairs)} coupled pairs")

    circuits, meta, windows, rep = [], [], {}, {}
    for v, p in pairs:
        tau = min(qprop(backend, p, "t2"), qprop(backend, v, "t1") / 3.0)
        d = max(GRAN, (int(round(tau * 1e9 / dt_ns)) // GRAN) * GRAN)
        windows[(v, p)] = d
        rep[(v, p)] = reported(backend, v, p)
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
        row = dict(victim=v, probe=p, tau_us=tau_s * 1e6,
                   contrast=0.5 * (c0 + c1),
                   zeta_hz=abs(dth) / (2 * math.pi * tau_s * PHASE_PER_ZETA),
                   gamma=G, bits=math.log2(1 + G),
                   nstar=2 * math.log(100) / max(G, 1e-9) ** 2,
                   detected=int(G > floor))
        row.update(rep[(v, p)])
        rows.append(row)

    det = [r for r in rows if r["detected"]]
    z = np.array([r["zeta_hz"] for r in det]); G = np.array([r["gamma"] for r in det])
    print(f"\ndetected on {len(det)} of {len(rows)} boundaries "
          f"(shot-noise floor on Gamma = {floor:.3f})")
    if len(det):
        print(f"zeta/2pi  median {np.median(z):.0f} Hz, range "
              f"{z.min():.0f} to {z.max():.0f} Hz")
        print(f"Gamma     median {np.median(G):.3f}, range "
              f"{G.min():.3f} to {G.max():.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
