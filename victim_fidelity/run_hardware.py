"""
Victim-side paired randomized benchmarking.

Tests whether the Pauli frame of Result 2, physically inserted around the
victim's own gates, costs the victim anything on its own qubit. This is a
different question from the probe-side benchmarking dropped from this
project (see the top-level README): that measured whether the frame's
effect on the boundary channel shows up in the adversary's calibration
data. This measures whether the frame's own gate overhead shows up in the
victim's.

Two arms, same random Clifford sequences in both, so any difference
between them isolates the frame's own cost rather than sequence-to-sequence
variation:

  bare    ordinary single-qubit randomized benchmarking on the victim,
          nothing added.
  framed  before every one of the same Clifford gates, a fresh random
          Pauli P is drawn and applied, then the Clifford C, then the
          correction C P^dagger C^dagger. This layer's net effect is
          exactly C regardless of P (Result 2's proof specialized to a
          single qubit with no boundary), so both arms accumulate the
          identical net Clifford across the sequence and share the same
          final inverting gate.

Every block, the Pauli, the Clifford and the correction, is followed by a
barrier. Without barriers the transpiler multiplies the whole single-qubit
sequence together, finds the identity, and removes it, so the hardware runs
an empty circuit and the survival probability is only the readout baseline.
With the barriers each block is compiled on its own and the framed circuits
carry roughly twice the physical pulses of the bare ones. This measures the
worst case, where the frame is executed as distinct pulses. If the compiler
were allowed to merge the frame into neighbouring gates, both arms would
compile to the same circuit and the frame would cost nothing by
construction. check_compiled() prints the pulse counts and refuses to
submit if the sequences have collapsed.

No boundary and no idle window are needed here, unlike every other
hardware script in this repository: the question is purely whether the
frame's own gates cost the victim anything, which has nothing to do with
leakage or with the coupling to a probe.
"""

from __future__ import annotations
import argparse, csv, math
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128]
SAMPLES_PER_LENGTH = 21   # 8 lengths x 21 samples x 2 arms = 336 sequences


def random_pauli_clifford(rng):
    """One of I, X, Y, Z as a Clifford object, drawn uniformly."""
    label = rng.choice(["I", "X", "Y", "Z"])
    qc = QuantumCircuit(1)
    if label == "X":
        qc.x(0)
    elif label == "Y":
        qc.y(0)
    elif label == "Z":
        qc.z(0)
    return Clifford(qc)


def build_sequence(rng, length):
    """One random Clifford sequence, and the two circuits built from it:
    bare (just the Cliffords) and framed (each Clifford wrapped in a fresh
    Pauli and its exact conjugated correction). Both end with the same
    inverting gate, since both accumulate the same net Clifford."""
    net = Clifford(QuantumCircuit(1))          # identity
    bare = QuantumCircuit(1)
    framed = QuantumCircuit(1)

    for _ in range(length):
        C = random_clifford(1, seed=rng)
        bare.compose(C.to_circuit(), inplace=True)
        bare.barrier()

        P = random_pauli_clifford(rng)
        # qiskit's Clifford.compose(X) means "self, then X" in circuit
        # order, which is the reverse of standard right-to-left matrix
        # multiplication. The correction that makes this layer's net
        # effect exactly C regardless of P, in matrix terms
        # correction = C . P^dagger . C^dagger, translates in this
        # circuit-order convention to C^dagger then P^dagger then C.
        correction = C.adjoint().compose(P.adjoint()).compose(C)
        for block in (P, C, correction):
            framed.compose(block.to_circuit(), inplace=True)
            framed.barrier()

        net = net.compose(C)

    inverse_circuit = net.adjoint().to_circuit()
    bare.compose(inverse_circuit, inplace=True)
    framed.compose(inverse_circuit, inplace=True)
    bare.measure_all()
    framed.measure_all()
    return bare, framed


def build_all(seed):
    rng = np.random.default_rng(seed)
    rows = []   # (arm, length, sample_idx), circuit
    for length in LENGTHS:
        for i in range(SAMPLES_PER_LENGTH):
            bare, framed = build_sequence(rng, length)
            rows.append((("bare", length, i), bare))
            rows.append((("framed", length, i), framed))
    return rows


def physical_pulses(qc):
    """Physical single-qubit pulses (sx and x) in a compiled circuit. rz is
    a virtual, zero-duration gate on IBM hardware, so it is not counted."""
    ops = qc.count_ops()
    return sum(n for name, n in ops.items() if name in ("sx", "x"))


def check_compiled(entries, compiled):
    """Print the mean physical pulse count per arm and length, and refuse to
    go on if the compiler has collapsed the sequences. A benchmarking
    sequence followed by its exact inverse multiplies to the identity, so
    without barriers the transpiler removes the whole thing and the
    hardware runs an empty circuit, which measures only readout error."""
    table = {}
    for (key, _), qc in zip(entries, compiled):
        arm, length, _ = key
        table.setdefault((arm, length), []).append(physical_pulses(qc))
    print("mean physical pulses per compiled circuit")
    print(f"{'length':>8}{'bare':>10}{'framed':>10}")
    for length in LENGTHS:
        print(f"{length:8d}{np.mean(table[('bare', length)]):10.1f}"
              f"{np.mean(table[('framed', length)]):10.1f}")
    top = max(LENGTHS)
    bare_top = np.mean(table[("bare", top)])
    framed_top = np.mean(table[("framed", top)])
    if bare_top < top / 4 or framed_top < 1.25 * bare_top:
        raise RuntimeError(
            "the compiled circuits do not carry gates that scale with "
            "sequence length, or the framed arm is not heavier than the "
            "bare arm. The compiler has probably merged or removed the "
            "sequence, so this run would not measure gate error or frame "
            "overhead. Not submitting.")


def survival(counts, shots):
    return counts.get("0", 0) / shots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--account", default=None)
    ap.add_argument("--channel", default="ibm_quantum_platform")
    ap.add_argument("--backend", default="ibm_kingston",
                     help="defaults to ibm_kingston, the same device used "
                          "for the survey and the frame hierarchy, so this measures the same "
                          "hardware the leakage results themselves came from")
    ap.add_argument("--qubit", type=int, default=60,
                     help="physical qubit to run the victim's RB on; "
                          "defaults to 60, the actual victim qubit used "
                          "throughout this paper's other ibm_kingston "
                          "measurements, not an arbitrary choice")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="fidelity_data.csv")
    args = ap.parse_args()

    entries = build_all(args.seed)
    print(f"built {len(entries)} circuits "
          f"({len(LENGTHS)} lengths x {SAMPLES_PER_LENGTH} samples x 2 arms)")

    if args.dry_run:
        pm = generate_preset_pass_manager(
            optimization_level=1, basis_gates=["rz", "sx", "x", "cz"])
        check_compiled(entries, [pm.run(c) for _, c in entries])
        print("dry run: circuits built and compiled locally, nothing submitted")
        return

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(channel=args.channel, token=args.account)
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False))
    print(f"running on {backend.name}")

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                       initial_layout=[args.qubit])
    isa_circuits = [pm.run(c) for _, c in entries]
    check_compiled(entries, isa_circuits)

    sampler = SamplerV2(mode=backend)
    # Gate twirling is the kind of randomisation under test and dynamical
    # decoupling changes what runs, so both are forced off, as in the other
    # scripts in this repository.
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.dynamical_decoupling.enable = False
    job = sampler.run(isa_circuits, shots=args.shots)
    print(f"submitted job {job.job_id()}, waiting for results")
    result = job.result()

    rows = []
    for (arm, length, i), res in zip((e[0] for e in entries), result):
        counts = res.data.meas.get_counts()
        rows.append(dict(arm=arm, length=length, sample=i,
                          survival=survival(counts, args.shots)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "length", "sample", "survival"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
