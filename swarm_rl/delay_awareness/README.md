# Delay-aware swarm diagnostic

This is a new four-UAV route task using Quad-Swarm dynamics and the existing low-level controller. There is no human operator, takeover, keyboard input, or handback. Earlier takeover experiments and negative results remain separate.

Each UAV observes its own current state and time-stamped delayed peer position/velocity packets. The initial shared packet is available before departure. Control runs at 10 Hz; fixed delays are 0, 100, 200, 400 and 800 ms. These are hypothetical settings, not measured hardware latency. 800 ms is a stress case excluded from the advance/stop gate.

Rules include nominal flight, reactive repulsion, and sampled predictive avoidance, each with raw, constant-velocity (CV), or exact-current-state information as applicable. Predictive avoidance evaluates 75 three-dimensional velocity candidates over 1.2 seconds. This is a custom simple baseline, not an ORCA or full MPC reproduction. The oracle is a same-controller information reference, not an optimal-control upper bound.

MLP and eight-frame GRU actors share local input information, approximately matched actor capacity, centralized current-observation critics, PPO settings, and training budget. The GRU re-encodes a finite window with zero initial hidden state, rather than carrying an indefinite hidden state. Each pilot uses one training seed and 16,384 steps; neither convergence nor a definitive comparison of network architectures is established.

## Reproduce

From the repository root, in the existing quad-swarm Python environment:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest tests/test_delay_awareness.py tests/test_delay_learning.py -q
python -m swarm_rl.delay_awareness.diagnose --output takeover_runs/delay-new/rules
python -m swarm_rl.delay_awareness.learning --variant mlp --output takeover_runs/delay-new/mlp
python -m swarm_rl.delay_awareness.learning --variant gru --output takeover_runs/delay-new/gru
python swarm_rl/delay_awareness/report.py takeover_runs/delay-new
```

Output paths must be new, and tracked source must be committed before full runs. The configuration is `configs/delay_awareness_pilot.json`. The completed pilot used commit `8c9f30110057a173788261eb07c2ae160c38209d`; subsequent documentation/report commits do not change its source provenance. Add `--smoke` to individual runners for an abbreviated integration check (not accepted by the full report verifier).

Artifacts include source snapshots, hashes, per-case trajectories, training transitions, checkpoints, optimizer/RNG state, reload verification and aggregate metrics. Raw artifacts remain in `takeover_runs/delay-pilot-001` on the server and local workstation; they are ignored by Git. Compact verified summaries are committed alongside this document.

## Scope

Two route families, eight development scenario seeds per cell, four UAVs, perfect sensing, synchronized timestamps, fixed all-to-all communication delay, and no packet loss or obstacles. No real-world or human collaboration claim follows. Success requires all vehicles to arrive and settle; collisions and timeouts both incur a 20-second completion cost. Low collision rate alone is insufficient.
