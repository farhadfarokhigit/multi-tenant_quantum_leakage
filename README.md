# Multi-tenant QPU leakage, code and data

Code and hardware data behind the figures in "Coherence Rather Than Error
Rate Governs Privacy in Multi-Tenant Quantum Computing." Organized by
figure, so each figure's script and data sit together.

## Structure

```
common/
    boundary_recon.py        quick single-boundary check: measures zeta and
                              picks a safe idle window before a full sweep

boundary_sweep/              the detailed q60-q61 sweep, with and without the frame
    run_hardware.py          submits the circuits, writes the CSV
    make_figure.py           reads the CSV, produces the figure
    data_kingston_q60_q61.csv
    boundary_sweep.pdf, .png

boundary_survey/             twenty-one boundaries on one device
    run_hardware.py
    make_figure.py
    survey_data.csv
    boundary_survey.pdf, .png

frame_hierarchy/             frame hierarchy on an active victim
    run_hardware.py
    make_figure.py
    hierarchy_data.csv
    hierarchy.pdf, .png

survey_iqm/                  the same survey on IQM hardware
    run_hardware.py
    make_figure.py
    survey_data_iqm.csv
    iqm_survey.pdf, .png

victim_fidelity/             randomized benchmarking of the victim qubit,
                             with and without the frame
    run_hardware.py
    make_figure.py
    fidelity_data.csv
    victim_fidelity.pdf, .png

datasheet_comparison/        every coupled pair on one device, measured leakage
                             against the numbers the provider reports
    run_hardware.py
    make_figure.py
    comparison_data.csv
    datasheet_comparison.pdf, .png

```

## Convention for the coupling rate

The boundary interaction is taken as $H_\times=\zeta Z_a\otimes Z_b$ with
$\zeta$ an angular frequency, and $\zeta/2\pi$ is quoted in hertz. With the
victim in $\ket{x}$ the probe sees $(-1)^x\zeta Z_b$, so the two victim
states rotate it by $\pm2\zeta\tau$ and the phase gap is
$\Delta\theta=4\zeta\tau$, with $\Gamma=C(\tau)|\sin(2\zeta\tau)|$.

## Setup

```
pip install -r requirements_IBM.txt
```

```
pip install -r requirements_IQM.txt
```

Regenerating a figure from the included CSV needs only `numpy` and
`matplotlib` (plus `scipy` for the fidelity figure), no quantum account.
Re-running an IBM hardware script needs `qiskit`, `qiskit-ibm-runtime`, and
access to the named backend. The IQM survey uses IQM's own client instead.
The IQM survey in `survey_iqm` talks to IQM's own service. Install
`iqm-client[qiskit]` in a separate environment, since it requires an older
`qiskit` than the IBM scripts, and get an API token from your IQM Resonance
account. These different requirements so run the appropriate one for your test.

## Running a figure

Each figure folder is self-contained. To rebuild a figure from the data
already here:

```
cd boundary_sweep
python3 make_figure.py
```

This reads the CSV in that same folder and writes the figure back into it.
The other figure folders work the same way.

To collect fresh hardware data instead of using what's included, run that
folder's `run_hardware.py` first. The IBM scripts take `--account` and
`--channel` for IBM Quantum credentials, and `--shots` for the shot count
per circuit. The defaults match what is in the paper, 4096 for the sweep,
the frame hierarchy and the fidelity test, and 1000 for the surveys and
the datasheet comparison.
Submitting real hardware jobs consumes your IBM Quantum processor-time
allocation, so check the shot count and circuit count before running, since
cost scales with both.

```
cd boundary_sweep
python3 run_hardware.py --account <your-account> --channel ibm_quantum_platform --shots 4096
python3 make_figure.py
```

The sweep script runs both arms, bare and Pauli frame, and writes
`data_kingston_q60_q61.csv` for `make_figure.py`.

The fidelity test in `victim_fidelity` defaults to `ibm_kingston` and qubit
60. Its `--dry-run` option compiles the circuits locally and prints how many
physical pulses each arm carries, and the script refuses to submit if
compilation has collapsed the sequences.

```
cd survey_iqm
python3 run_hardware.py --quantum-computer garnet --n-boundaries 20 --token <your-token>
python3 make_figure.py
```

or set the token in the `IQM_TOKEN` environment variable and leave out
`--token`.
