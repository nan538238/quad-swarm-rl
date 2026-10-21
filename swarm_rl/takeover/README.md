# Four-UAV takeover diagnostics

This is an experimental task adapter over the existing QuadSwarm six-DoF
`QuadrotorDynamics`, with its NumPy nonlinear position controller. It is not a
reproduction of upstream MARL results or an implementation of a validated new
learning method. All current results are preliminary infrastructure evidence.

Four fixed slots move at 0.3 m/s. A scripted operator temporarily overrides one
UAV's velocity command to visit an outward inspection point, then releases it.
Other UAVs receive current takeover flags, relative positions/velocities and
their own slot errors. Actors cannot observe future commands, operator targets,
or the exact release time. Actions are velocity residuals over slot tracking.

`nominal` continues slot tracking. `wait` applies bounded residuals to hold each
teammate's position at takeover onset, then rejoins. Both use identical action
limits; `wait` is not an unlimited stop command.

Recovery requires every UAV to satisfy both position and velocity tolerances
continuously for 0.5 seconds. Failures retain a 10-second timeout cost. A crash
before handback has no observed recovery time and must count as a failed
scenario, never be dropped from evaluation. Inspection completion, collision,
progress and recovery must be reported together. End-of-episode progress can
hide transient delays, so the raw trajectories and takeover progress are kept.

The initial diagnostic uses five paired engineering seeds, zero motor noise,
no obstacles, fixed dynamics and outward detours. Coupling is limited to contact
termination and a joint recovery criterion: it may reveal that the proposed
coordination problem is too weak. Results cannot establish algorithmic novelty,
real human collaboration, or publication readiness.

## Running on n1-research

The isolated environment is `/home/u2025111685/.venvs/quad-swarm`.
Run from `/home/u2025111685/projects/quad-swarm-rl`:

```bash
/home/u2025111685/.venvs/quad-swarm/bin/python -m unittest discover -s tests -p test_takeover_env.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/u2025111685/.venvs/quad-swarm/bin/python -m swarm_rl.takeover.diagnose --output takeover_runs/manual-001
```

Output directories must be new; prior artifacts are never overwritten.
Diagnostics run on CPU. The server GPU is available for later learning work;
its user-scoped driver-library workaround is:

```bash
export LD_LIBRARY_PATH=/home/u2025111685/.local/lib/codex-n1-nvidia:${LD_LIBRARY_PATH:-}
```

Tested environment uses Python 3.11.5, NumPy 1.26.4, SciPy 1.14.1,
Numba 0.60.0 and Gymnasium 0.28.1. It inherits PyTorch 2.1.1 from the existing
PyTorch environment. This differs from upstream's Python >=3.11.10 / PyTorch
2.5.0 packaging requirements. Modules are run directly from the checkout;
the full upstream training dependency stack has not been validated.

`snapshot/restore` preserves dynamics, rotor filters, controller and RNG state.
`evaluate_immediate_handback` rolls out the fixed nominal recovery controller
and restores live state, recording extra physics steps. It is a label primitive,
not a trained recovery estimator. Partial-horizon labels are explicitly marked.

Governance, approved experiment cards, immutable manifests and audit records
are maintained separately in the adjacent `build-research-evidence` project.

## Geometry and recovery-prediction diagnostics

`geometry_diagnose` runs the E0002 balanced eight-direction sweep with nominal,
bounded wait, all-phase reactive, and post-handback-only reactive controllers.
Its output retains all scenarios, collision phases, and transient patrol lag.
The geometry adapter preserves the original environment at its default angle.

`recovery_diagnose` implements E0003: collect full-horizon hypothetical handback
labels at two takeover snapshots, then compare a constant predictor, a four-feature
ridge regressor and a 164-input neural cost regressor. It keeps raw observations,
missing-snapshot reasons, reference rollout costs, checkpoints and source hashes.
Normalization and ridge fitting use training scenarios only. Validation selects
the neural epoch, so validation scores are development results, not final tests.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m swarm_rl.takeover.geometry_diagnose --config configs/takeover_e0002.json --output takeover_runs/new-geometry-run
python -m swarm_rl.takeover.recovery_diagnose --config configs/takeover_e0003.json --output takeover_runs/new-predictor-run
```

Use the server venv and driver-library environment above. E0003 requests CUDA
and fails explicitly if it is unavailable. Both commands need a fresh output
directory. No MARL control policy has been trained by these diagnostic commands;
prediction accuracy alone cannot establish improved team behavior.

## CP-C2-01 bounded policy screen

`policy_train` implements the E0004 shared decentralized 41-input actor and
separate centralized team critic. All 32768 live control transitions count;
only autonomous teammate rows during takeover contribute actor gradients.
The complete transition stream trains the critic, including nominal patrol
and handback. Actual episode termination and the 24-second limit stop GAE;
rollout boundaries bootstrap. Gaussian latents are retained for stable tanh
log-probabilities. Fixed feature scales do not depend on validation data.

The fixed final checkpoint is reloaded for 80 deterministic validation cases.
No checkpoint is selected by validation. Frozen E0003 predictor identity is
asserted before and after dense training. Raw training transitions, updates,
episodes, validation trajectories and cases, incomplete final episode, source
copies, hashes and live/label costs are retained in each fresh output folder.
The orchestrator's manifest wrapper must authorize and record scientific runs.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m swarm_rl.takeover.policy_train --config configs/takeover_e0004.json --variant none --output takeover_runs/NEW-none
```

E0004 variants are `none`, `analytic`, `dense`, and `rules`. `rules` evaluates
nominal, bounded wait during takeover, reactive during takeover, and separately
reactive all phases. The `sparse` implementation requires a separately activated
E0005 configuration and reuses E0004 baseline artifacts. All outputs are
preliminary, with one training seed and no final-test scenarios. Recovery cost
is 10 seconds for any collision or absent recovery; inspection failure is
reported separately. Compare integrated patrol lag only on paired common-success
cases, because crashes shorten the observed duration.
