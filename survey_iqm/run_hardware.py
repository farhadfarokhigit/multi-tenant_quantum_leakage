"""
Boundary survey, IQM hardware via IQM's own Resonance service directly.

Same measurement as boundary_survey, a different vendor's
superconducting hardware, to check whether the coherent-versus-stochastic
distinction and the frame's effectiveness are IBM-specific or general to
tunable-coupler architectures. Same circuits, same analysis, only the
submission layer differs.

This talks to IQM directly (qiskit-iqm, IQMProvider), not through the
Open Quantum aggregator. That's a deliberate change: going through Open
Quantum produced three different failures across three vendors in one
session (a transpiler-accepted-but-scheduler-rejected 'delay' on both IQM
Garnet and Emerald, and a Rigetti backend object missing its own coupling
map entirely), all inside Open Quantum's own layer, not on the vendor
hardware itself. IQM's own maintained client is a more standard, better
tested path, and this uses the free Resonance Starter tier (30 credits a
month, no card needed) rather than Open Quantum's credits.

pip install qiskit-iqm

Get an API token from your Resonance dashboard (resonance.iqm.tech), then
either set it as the IQM_TOKEN environment variable or pass --token
directly.

One piece of this is flagged rather than asserted: qprop() below tries the
same qubit_properties()/properties() calls the IBM version used to look up
per-qubit T1/T2 for choosing each boundary's idle window. Whether an
IQMBackend exposes calibration data through the same calls is not
something I could confirm without a live account to test against. The
function already falls back to a fixed default if those calls fail, so
the script will still run either way, just with a less-tailored idle
window if the calibration lookup doesn't come through the same API.

Also unconfirmed: whether IQM's own service accepts 'delay' on real jobs
the way the transpiler accepts it locally. Open Quantum's scheduler
rejected it outright even though it transpiled cleanly, so this is worth
checking on a small circuit before trusting a full run, not assumed to be
fixed just because the aggregator is out of the picture.
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
    """Per-qubit T1 or T2, in seconds. Falls back to a fixed default if the
    backend doesn't expose calibration data through either of these calls,
    see the module docstring, this fallback is untested against Open
    Quantum's actual backend object."""
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
    probe coherence. Disjointness matters: reusing a qubit as probe for one
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
    isa = generate_preset_pass_manager(optimization_level=1,
                                       backend=backend).run(circuits)
    if args.dry_run:
        from qiskit_aer import AerSimulator
        r = AerSimulator.from_backend(backend).run(isa, shots=shots).result()
        return [r.get_counts(i) for i in range(len(isa))]

    # IQM's own service, unlike Open Quantum, returns standard binary-keyed
    # counts, so no key-format conversion is needed here. Dynamical
    # decoupling would actively suppress the crosstalk this experiment
    # measures, the same reason the IBM scripts explicitly disable
    # twirling and DD rather than trust the default. IQM's native
    # CircuitCompilationOptions defaults dd_mode to DISABLED already, but
    # this sets it explicitly rather than relying on an inherited default
    # that hasn't been directly confirmed for this specific run() path.
    from iqm.iqm_client.models import CircuitCompilationOptions, DDMode
    compilation_options = CircuitCompilationOptions(dd_mode=DDMode.DISABLED)
    job = backend.run(isa, shots=shots, circuit_compilation_options=compilation_options)
    print("job id", job.job_id())
    result = job.result()
    return [result.get_counts(i) for i in range(len(isa))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantum-computer", default="garnet",
                     help="IQM device name, e.g. garnet or emerald")
    ap.add_argument("--token", default=None,
                     help="IQM Resonance API token; defaults to the IQM_TOKEN "
                          "environment variable if not given")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--n-boundaries", type=int, default=20)
    ap.add_argument("--distance", type=int, default=1)
    ap.add_argument("--out", default="survey_data_iqm.csv")
    args = ap.parse_args()

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
        print("dry run: using a local fake backend, not IQM, "
              "since this only checks the circuit-building logic")
    else:
        from iqm.qiskit_iqm import IQMProvider
        provider = IQMProvider("https://resonance.iqm.tech/",
                               quantum_computer=args.quantum_computer,
                               token=args.token)
        # use_metrics=True is required for backend.qubit_properties() to
        # return real calibration data; without it every T1/T2 lookup in
        # qprop() silently falls through to its hardcoded default, which is
        # exactly what happened on the first real run of this script,
        # confirmed directly against a live account (t1/t2 came back None
        # without this flag, real numbers with it).
        backend = provider.get_backend(use_metrics=True)

    dt_ns = backend.target.dt * 1e9 if backend.target.dt else 1.0
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

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(rows)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
