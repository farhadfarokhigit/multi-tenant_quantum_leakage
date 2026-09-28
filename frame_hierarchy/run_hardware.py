"""
Non-idle victim, and the frame hierarchy on hardware.

This is the experiment that closes the gap between the theorem and the
demonstration.  Result 2 requires a uniform Pauli frame over the victim's WHOLE
register.  The earlier measurement used a boundary-only frame and was legitimate
only because the victim idled.  Here the victim runs a circuit that transports
its secret from an interior qubit onto the boundary qubit, which is exactly the
counterexample of Supplemental Sec. S4, and we test three arms.

  bare      no frame                      -> leaks
  local     uniform Pauli on the boundary  -> STILL LEAKS, because the
            qubit only                       randomization happens before the
                                             transport
  full      uniform Pauli on the whole     -> leakage vanishes
            victim register

Both twirls are enumerated exactly rather than sampled.  With a two-qubit victim
register the local frame is 4 Paulis and the full frame is 16, so the equal-weight
average over circuits IS the uniform twirl, with no sampling error.

Register layout is a line, v0 (interior, holds the secret) - va (boundary) - p
(adversary probe).  The victim's layer is a SWAP of v0 and va, so the secret
arrives at the boundary after the frame has already acted.

Cost: ~336 circuits at 4096 shots, roughly 7 minutes.
"""

from __future__ import annotations
import argparse, math, itertools, csv
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

GRAN = 16
PAULIS = ("I", "X", "Y", "Z")
FRACTIONS = [0.25, 0.5, 0.75, 1.0]        # of the usable window


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


def find_line(backend):
    """A 3-qubit path v0 - va - p, ranked by the probe's coherence."""
    import rustworkx as rx
    g = backend.coupling_map.graph.to_undirected()
    nbr = {i: set(g.neighbors(i)) for i in g.node_indices()}
    best = None
    for va in nbr:
        for p in nbr[va]:
            for v0 in nbr[va]:
                if v0 == p:
                    continue
                score = min(qprop(backend, p, "t2"),
                            qprop(backend, va, "t1") / 3.0,
                            qprop(backend, v0, "t1") / 3.0)
                if best is None or score > best[0]:
                    best = (score, int(v0), int(va), int(p))
    return best[1], best[2], best[3]


def apply_pauli(qc, q, letter):
    if letter == "X": qc.x(q)
    elif letter == "Y": qc.y(q)
    elif letter == "Z": qc.z(q)


def build(v0, va, probe, nq, delay_dt, secret, basis, frame, transport=True):
    """
    frame is a dict {qubit: pauli letter}.  The correction after the victim's
    layer is the frame conjugated through it, which for a SWAP is the same
    Paulis with the two qubits exchanged.  Net effect on the victim's logical
    state is the identity, which is what randomized compiling guarantees.
    """
    qc = QuantumCircuit(nq, 1)
    if secret:
        qc.x(v0)                                  # secret lives on the interior
    qc.h(probe)
    qc.barrier([v0, va, probe])

    for q, P in frame.items():                    # frame, before the victim layer
        apply_pauli(qc, q, P)
    qc.barrier([v0, va, probe])

    if transport:
        qc.swap(v0, va)                           # victim's own circuit
    qc.barrier([v0, va, probe])

    if delay_dt:                                  # boundary acts here
        for q in (v0, va, probe):
            qc.delay(delay_dt, q, unit="dt")
    qc.barrier([v0, va, probe])

    if transport:                                 # correction, conjugated
        corr = {v0: frame.get(va, "I"), va: frame.get(v0, "I")}
    else:
        corr = frame
    for q, P in corr.items():
        apply_pauli(qc, q, P)
    qc.barrier([v0, va, probe])

    if basis == "Y":
        qc.sdg(probe)
    qc.h(probe)
    qc.measure(probe, 0)
    return qc


def frames_for(arm, v0, va):
    if arm == "bare":
        return [{}]
    if arm == "local":                            # boundary qubit only
        return [{va: P} for P in PAULIS]
    return [{v0: A, va: B} for A in PAULIS for B in PAULIS]   # full register


def run(circuits, backend, args, shots):
    isa = generate_preset_pass_manager(optimization_level=0,
                                       backend=backend).run(circuits)
    if args.dry_run:
        from qiskit_aer import AerSimulator
        r = AerSimulator.from_backend(backend).run(isa, shots=shots).result()
        return [r.get_counts(i) for i in range(len(isa))]
    from qiskit_ibm_runtime import SamplerV2, Batch
    out = []
    with Batch(backend=backend) as batch:
        s = SamplerV2(mode=batch)
        s.options.twirling.enable_gates = False
        s.options.twirling.enable_measure = False
        s.options.dynamical_decoupling.enable = False
        s.options.default_shots = shots
        jobs = [s.run(isa[i:i+300], shots=shots) for i in range(0, len(isa), 300)]
        for j in jobs:
            print("   job", j.job_id())
        for j in jobs:
            out += [list(r.data.values())[0].get_counts() for r in j.result()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--instance", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--triple", nargs=3, type=int, default=None,
                    help="v0 va probe, a connected line")
    ap.add_argument("--out", default="hierarchy_data.csv")
    args = ap.parse_args()

    if args.dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService(name=args.account, channel=args.channel,
                                       instance=args.instance).backend(args.backend)

    v0, va, probe = args.triple if args.triple else find_line(backend)
    dt_ns = backend.target.dt * 1e9
    nq = backend.num_qubits
    usable = min(qprop(backend, probe, "t2"),
                 qprop(backend, va, "t1") / 3.0, qprop(backend, v0, "t1") / 3.0)
    delays = [max(GRAN, (int(round(f * usable * 1e9 / dt_ns)) // GRAN) * GRAN)
              for f in FRACTIONS]

    print(f"backend {backend.name}")
    print(f"victim interior q{v0}, victim boundary q{va}, adversary probe q{probe}")
    print(f"usable window {usable*1e6:.1f} us, delays "
          f"{[round(d*dt_ns/1000,1) for d in delays]} us")

    circuits, meta = [], []
    for arm in ("bare", "local", "full"):
        for fr in frames_for(arm, v0, va):
            for secret in (0, 1):
                for d in delays:
                    for basis in ("X", "Y"):
                        circuits.append(build(v0, va, probe, nq, d, secret,
                                              basis, fr))
                        meta.append((arm, tuple(sorted(fr.items())), secret,
                                     d, basis))
    print(f"{len(circuits)} circuits "
          f"(bare 1 frame, local 4 frames, full 16 frames, all enumerated)")
    counts = run(circuits, backend, args, args.shots)

    ev = {}
    for c, k in zip(counts, meta):
        ev[k] = 2 * c.get("0", 0) / args.shots - 1

    floor = 3.0 / math.sqrt(args.shots * 16)
    print(f"\n{'arm':>7}{'delay(us)':>11}{'contrast':>10}{'dtheta':>10}"
          f"{'Gamma':>9}{'bits':>8}")
    print("-" * 55)
    res = {}
    for arm in ("bare", "local", "full"):
        frs = [tuple(sorted(f.items())) for f in frames_for(arm, v0, va)]
        res[arm] = []
        for d in delays:
            def avg(secret, basis):
                return float(np.mean([ev[(arm, f, secret, d, basis)] for f in frs]))
            ex0, ey0 = avg(0, "X"), avg(0, "Y")
            ex1, ey1 = avg(1, "X"), avg(1, "Y")
            c0, c1 = math.hypot(ex0, ey0), math.hypot(ex1, ey1)
            dth = math.atan2(ey0, ex0) - math.atan2(ey1, ex1)
            dth = (dth + math.pi) % (2 * math.pi) - math.pi
            G = 0.5 * (c0 + c1) * abs(math.sin(dth / 2))
            res[arm].append(dict(delay_us=d * dt_ns / 1000, contrast=0.5*(c0+c1),
                                 dtheta=dth, gamma=G, bits=math.log2(1 + G)))
            print(f"{arm:>7}{d*dt_ns/1000:11.1f}{0.5*(c0+c1):10.3f}{dth:10.4f}"
                  f"{G:9.4f}{math.log2(1+G):8.4f}")

    gmax = {a: max(r["gamma"] for r in res[a]) for a in res}
    print(f"\npeak Gamma:  bare {gmax['bare']:.4f}   local frame "
          f"{gmax['local']:.4f}   full frame {gmax['full']:.4f}")
    print(f"shot-noise floor on Gamma (16 frames averaged): {floor:.4f}\n")

    if gmax["local"] > 3 * floor and gmax["full"] < 3 * floor:
        print("RESULT  The frame hierarchy is confirmed on hardware.  A")
        print("        boundary-only frame fails once the victim transports its")
        print("        secret to the boundary, and the full-register frame")
        print("        closes the channel.  This is Supplemental Sec. S4")
        print("        measured rather than simulated.")
    elif gmax["local"] <= 3 * floor:
        print("RESULT  The local frame did not fail.  Check that the SWAP was")
        print("        not optimized away and that the secret really reaches the")
        print("        boundary, by rerunning with --dry-run and inspecting the")
        print("        transpiled circuit.")
    else:
        print("RESULT  The full-register frame did not close the channel.  Either")
        print("        the boundary reaches the victim outside the framed qubits,")
        print("        or Sampler gate twirling is interfering.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "delay_us", "contrast",
                                          "dtheta", "gamma", "bits"])
        w.writeheader()
        for arm in ("bare", "local", "full"):
            for row in res[arm]:
                w.writerow({"arm": arm, **row})
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
