"""
Cross-tenant boundary leakage on IBM Quantum hardware.

The probe qubit idles in |+> next to a victim qubit held in |0> or |1>. The
idle window is swept, the probe is read in the X and Y bases, and the
accumulated phase difference between the two victim states gives the static
ZZ rate and the conditional spread, from which the predicted adversary shot
budget follows. The same sweep is repeated with a uniform Pauli frame on the
victim's boundary qubit, implemented exactly as an equal-weight average over
frame circuits, and the residual gap and the resulting upper bound on leakage
are reported. Both arms are written to one CSV for make_figure.py.

Requires qiskit >= 2.0 and qiskit-ibm-runtime >= 0.40.
Tested here against qiskit 2.5.2 and qiskit-ibm-runtime 0.49.0.
"""

from __future__ import annotations
import argparse, csv, math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------------------------
# THREE THINGS THAT WILL SILENTLY RUIN THIS EXPERIMENT
#
# 1. The Sampler applies Pauli twirling by default in some configurations.
#    Gate twirling IS the defence we are trying to measure.  If it is on in
#    the bare arm the attack signal disappears and the result looks like a null.
#    We force twirling off everywhere and switch it on only by constructing the
#    frame circuits ourselves, so that we control exactly what is twirled.
#
# 2. Dynamical decoupling echoes away the static ZZ.  It must be off.  If a
#    provider enables DD by default on idle qubits, that is itself a finding
#    and should be reported, but it destroys the bare-arm signal.
#
# 3. The transpiler will happily delete an idle window or move a delay across a
#    barrier at optimization_level > 0.  Use level 0 and explicit barriers.
# ---------------------------------------------------------------------------

# With H_cross = zeta * Z_a Z_b the probe sees H = (-1)^x * zeta * Z_b, so the
# two victim states rotate it by +-2*zeta*tau and the phase gap is
# dtheta = 4*zeta*tau. The slope of dtheta against tau is therefore 4*zeta.
PHASE_PER_ZETA = 4.0

# The measured spread is delta(tau) = exp(-tau/T2)|sin(2*zeta*tau)|, which for
# small angles is 2*zeta*tau*exp(-tau/T2) and peaks near tau = T2. Sweep in
# units of the probe's T2 rather than in absolute ns, and straddle the peak so
# the fit has curvature to work with.
DELAY_FRACTIONS_OF_T2 = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8]
SHOTS = 4096


# ------------------------------ circuits -----------------------------------

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


def probe_t2(backend, q) -> float:
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


def _align(dt_ns: float, delay_ns: float, granularity: int = 16) -> int:
    """Round a delay in ns to an integer number of dt, aligned to granularity."""
    n = int(round(delay_ns / dt_ns))
    return max(0, (n // granularity) * granularity)


def boundary_circuit(victim: int, probe: int, n_qubits: int,
                     delay_dt: int, victim_state: str, basis: str,
                     frame: str = "I", occupancy: int = 0,
                     granularity: int = 16) -> QuantumCircuit:
    """
    victim_state : '0', '1', or '+'  -- the bit the adversary is trying to learn
    basis        : 'X' or 'Y'        -- probe readout basis
    frame        : 'I','X','Y','Z'   -- victim-side Pauli frame (logically identity)
    occupancy    : if > 0, drive the victim with this many X pulses instead of
                   idling, to expose schedule leakage that survives the frame
    """
    qc = QuantumCircuit(n_qubits, 1)

    if victim_state == "1":
        qc.x(victim)
    elif victim_state == "+":
        qc.h(victim)

    qc.h(probe)                       # probe into the equatorial plane
    qc.barrier([victim, probe])

    if frame != "I":                  # frame gate
        getattr(qc, frame.lower())(victim)
    qc.barrier([victim, probe])

    if occupancy:
        # Even number of X pulses, logically identity, but the victim is driven.
        # Sub-delays must also respect the pulse-alignment granularity or the
        # transpiler raises ConstrainedReschedule.  The victim window is longer
        # than the probe window by 2*occupancy*t_X, of order a few hundred ns,
        # which is immaterial here because this option tests presence of drive and not
        # a precise phase.
        per = ((delay_dt // (2 * occupancy)) // granularity) * granularity
        for _ in range(occupancy):
            qc.x(victim)
            if per: qc.delay(per, victim, unit="dt")
            qc.x(victim)
            if per: qc.delay(per, victim, unit="dt")
        if delay_dt: qc.delay(delay_dt, probe, unit="dt")
    elif delay_dt:
        qc.delay(delay_dt, victim, unit="dt")
        qc.delay(delay_dt, probe, unit="dt")

    qc.barrier([victim, probe])
    if frame != "I":                  # frame correction
        getattr(qc, frame.lower())(victim)
    qc.barrier([victim, probe])

    if basis == "Y":
        qc.sdg(probe)
    qc.h(probe)
    qc.measure(probe, 0)              # the adversary never touches the victim
    return qc


# ------------------------------ analysis -----------------------------------

def expval(counts: dict, shots: int) -> float:
    p0 = counts.get("0", 0) / shots
    return 2.0 * p0 - 1.0


def phase_from(ex: float, ey: float) -> float:
    return math.atan2(ey, ex)


def contrast(ex: float, ey: float) -> float:
    return math.hypot(ex, ey)


def delta_from_phase_gap(dtheta: float, c0: float, c1: float) -> float:
    """Trace distance between the two probe states, damped by measured contrast."""
    return 0.5 * (c0 + c1) * abs(math.sin(dtheta / 2.0))


def shot_budget(delta: float, conf: float = 0.99, collective: bool = False) -> float:
    """Chernoff shot budget.  collective=True needs quantum memory across shots."""
    if delta <= 0:
        return float("inf")
    s = min(delta, 1.0 - 1e-12)
    xi = -math.log(math.sqrt(1.0 - s * s))          # local, optimal single-shot basis
    if collective:
        xi *= 2.0                                    # quantum Chernoff exponent
    return math.log(1.0 / (1.0 - conf)) / xi


def leakage_bound(delta: float, M: int = 2) -> float:
    """
    Bits.  For M = 2 the maximal quantum leakage is EXACTLY log2(1 + Gamma).
    The relaxation to log2(1 + M*Gamma) is only needed for M > 2, and quoting
    it at M = 2 overstates the leakage by roughly a factor of 1.8.
    """
    if M == 2:
        return math.log2(1.0 + delta)
    return min(math.log2(M), math.log2(1.0 + M * delta))



def build_jobs(victim, probe, n_qubits, delays_dt, arms):
    """arms: list of (label, victim_state, frame, occupancy)."""
    circuits, meta = [], []
    for label, vs, frame, occ in arms:
        for d in delays_dt:
            for basis in ("X", "Y"):
                circuits.append(boundary_circuit(victim, probe, n_qubits,
                                                 d, vs, basis, frame, occ))
                meta.append(dict(arm=label, delay_dt=d, basis=basis,
                                 victim_state=vs, frame=frame, occupancy=occ))
    return circuits, meta


def analyse(results, meta, dt_ns, shots, label_a, label_b, frames_a=("I",),
            frames_b=("I",)):
    """Average over frames with equal weight, then compare arm A against arm B."""
    table = {}
    for r, m in zip(results, meta):
        table[(m["arm"], m["frame"], m["delay_dt"], m["basis"])] = r

    delays = sorted({m["delay_dt"] for m in meta})
    rows = []
    for d in delays:
        def frame_avg(arm, frames, basis):
            vals = [expval(table[(arm, f, d, basis)], shots)
                    for f in frames if (arm, f, d, basis) in table]
            return float(np.mean(vals)) if vals else float("nan")

        ex0, ey0 = frame_avg(label_a, frames_a, "X"), frame_avg(label_a, frames_a, "Y")
        ex1, ey1 = frame_avg(label_b, frames_b, "X"), frame_avg(label_b, frames_b, "Y")
        if any(math.isnan(v) for v in (ex0, ey0, ex1, ey1)):
            continue          # this arm was not run at this delay
        th0, th1 = phase_from(ex0, ey0), phase_from(ex1, ey1)
        c0, c1 = contrast(ex0, ey0), contrast(ex1, ey1)
        dth = (th0 - th1 + math.pi) % (2 * math.pi) - math.pi
        delta = delta_from_phase_gap(dth, c0, c1)
        rows.append(dict(delay_ns=d * dt_ns, dtheta=dth, contrast=0.5 * (c0 + c1),
                         delta=delta,
                         q0=(1 + delta) / 2, q1=(1 - delta) / 2,
                         Nstar_pred=shot_budget(delta),
                         Nstar_coll=shot_budget(delta, collective=True),
                         leak_bits=leakage_bound(delta),
                         p0=(1 + ey0) / 2, p1=(1 + ey1) / 2))
    return rows


def fit_zz(rows, dt_ns):
    """Linear fit of the phase gap against idle time.  Returns zeta/2pi in Hz,
    where the fitted slope is PHASE_PER_ZETA * zeta."""
    t = np.array([r["delay_ns"] for r in rows]) * 1e-9
    y = np.unwrap(np.array([r["dtheta"] for r in rows]))
    good = t > 0
    if good.sum() < 2:
        return float("nan")
    slope = np.polyfit(t[good], y[good], 1)[0]      # rad/s
    return abs(slope) / (2 * math.pi * PHASE_PER_ZETA)


def report(rows, title, dt_ns):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(f"{'idle (ns)':>10} {'dtheta':>9} {'contrast':>9} {'delta':>9} "
          f"{'N* pred':>9} {'leak (bits)':>12}")
    print("-" * 88)
    for r in rows:
        print(f"{r['delay_ns']:10.0f} {r['dtheta']:9.4f} {r['contrast']:9.4f} "
              f"{r['delta']:9.5f} {r['Nstar_pred']:9.1f} "
              f"{r['leak_bits']:12.5f}")
    print(f"\nfitted static ZZ  zeta/2pi = {fit_zz(rows, dt_ns)/1e3:.2f} kHz")


# ------------------------------ runner -------------------------------------

def get_backend(args):
    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(name=args.account, channel=args.channel,
                                   instance=args.instance)
    return service.backend(args.backend), service


def run(circuits, backend, service, args, shots):
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa = pm.run(circuits)

    if args.dry_run:
        from qiskit_aer import AerSimulator
        sim = AerSimulator.from_backend(backend)
        res = sim.run(isa, shots=shots).result()
        return [res.get_counts(i) for i in range(len(isa))]

    from qiskit_ibm_runtime import SamplerV2

    def _configure(s):
        # See the warning block at the top of this file.
        s.options.twirling.enable_gates = False
        s.options.twirling.enable_measure = False
        s.options.dynamical_decoupling.enable = False
        s.options.default_shots = shots
        return s

    if args.job_mode:
        # Open plan accounts are restricted to job mode.
        job = _configure(SamplerV2(mode=backend)).run(isa, shots=shots)
        print("job id", job.job_id())
        out = job.result()
    else:
        from qiskit_ibm_runtime import Batch
        with Batch(backend=backend) as batch:
            job = _configure(SamplerV2(mode=batch)).run(isa, shots=shots)
            print("job id", job.job_id())
            out = job.result()
    return [r.data.c0.get_counts() if hasattr(r.data, "c0")
            else list(r.data.values())[0].get_counts() for r in out]


def pick_pair(backend, distance=1):
    """Choose a victim and a probe at the requested coupling-graph distance."""
    cm = backend.coupling_map
    import rustworkx as rx
    g = cm.graph.to_undirected()
    lengths = rx.all_pairs_dijkstra_path_lengths(g, lambda _: 1.0)
    for v, dists in lengths.items():
        for p, dd in dists.items():
            if int(dd) == distance:
                return int(v), int(p)
    raise RuntimeError(f"no pair at distance {distance}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--account", default=None,
                    help="named saved account, e.g. unimelb")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--job-mode", action="store_true",
                    help="use job mode instead of Batch (Open plan)")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--distance", type=int, default=1)
    ap.add_argument("--pair", nargs=2, type=int, default=None)
    ap.add_argument("--out", default="data_kingston_q60_q61.csv",
                    help="output CSV holding both arms")
    args = ap.parse_args()

    backend, service = get_backend(args)
    dt_ns = backend.target.dt * 1e9
    n_qubits = backend.num_qubits
    victim, probe = (tuple(args.pair) if args.pair
                     else pick_pair(backend, args.distance))
    t2 = min(probe_t2(backend, probe), victim_t1(backend, victim) / 3.0)
    delays_dt = [_align(dt_ns, f * t2 * 1e9) for f in DELAY_FRACTIONS_OF_T2]
    print(f"probe T2 = {t2*1e6:.1f} us   longest idle window = "
          f"{max(delays_dt)*dt_ns/1000:.1f} us")

    print(f"backend {backend.name}  dt = {dt_ns:.4f} ns  "
          f"victim q{victim}  probe q{probe}  distance {args.distance}")

    # For a victim in a computational basis state the Z frame acts as the
    # identity and the Y frame reproduces X up to a global phase, so {I, X}
    # is the EXACT uniform Pauli twirl here, not an approximation to it.
    # Use ("I","X","Y","Z") only if you also run a superposition victim.
    frames = ("I", "X")
    data_arms = ([("v0", "0", f, 0) for f in frames]
                 + [("v1", "1", f, 0) for f in frames])

    circuits, meta = build_jobs(victim, probe, n_qubits, delays_dt, data_arms)
    print(f"build: {len(circuits)} circuits")
    counts = run(circuits, backend, service, args, args.shots)

    bare = analyse(counts, meta, dt_ns, args.shots, "v0", "v1",
                   ("I",), ("I",))
    report(bare, "bare, frame = I only", dt_ns)

    pfr = analyse(counts, meta, dt_ns, args.shots, "v0", "v1",
                  frames, frames)
    report(pfr, "uniform Pauli frame, data leakage", dt_ns)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "tau_ns", "dtheta_rad",
                                          "contrast", "delta"])
        w.writeheader()
        for arm_name, rows in (("bare", bare), ("pfr", pfr)):
            for r in rows:
                w.writerow({"arm": arm_name, "tau_ns": r["delay_ns"],
                           "dtheta_rad": r["dtheta"],
                           "contrast": r["contrast"], "delta": r["delta"]})
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
