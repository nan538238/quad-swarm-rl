# Delay-awareness pilot 001 — 2026-10-22

## Decision

The prespecified engineering screening gate is **not passed**. Do not start a complex learned prediction module or claim paper-level novelty from this pilot. This rejects the current justification for added complexity, not every possible delay-aware multi-UAV research problem.

All 720 evaluation cases completed: 560 rules and 80 each for MLP and GRU. Each learning run completed 16,384 steps with one training seed. Ten new implementation tests passed. Artifact verification checked 1,486 file hashes, exact case coverage, paired initial conditions, zero-delay estimator equivalence, invariant oracle/nominal control outcomes across delay, and checkpoint reload identity. See `verification.json` and `VERIFIED_TABLES.md`.

## Crossing routes

Successes out of eight (each policy-delay cell uses the same eight scenario seeds):

| Policy | 0 ms | 100 ms | 200 ms | 400 ms | 800 ms stress |
|---|---:|---:|---:|---:|---:|
| Predictive avoidance, raw packets | 8 | 8 | 8 | 7 | 1 |
| Predictive avoidance, CV extrapolation | 8 | 8 | 8 | 7 | 7 |
| Predictive avoidance, current-state reference | 8 | 8 | 8 | 8 | 8 |
| MLP | 1 | 1 | 1 | 2 | 3 |
| Eight-frame GRU | 0 | 0 | 0 | 0 | 0 |

For these predictive rules every failure was a collision. At 400 ms, mean failure-inclusive completion costs were 10.575 s (raw), 10.723 s (CV), and 9.277 s (current state). At 800 ms they were 18.812 s, 10.883 s, and 9.277 s. These costs assign 20 s to failure; they are not means restricted to successful flights.

Nominal flight collided in all eight crossing cases at every delay. Reactive repulsion also collided in all cases at 0–400 ms, including with current-state information. Its failures therefore do not establish a delay-induced difficulty. At 800 ms reactive CV succeeded once; the other reactive variants still failed throughout.

The zero-delay predictive baseline succeeds throughout, so crossing is solvable. At 100–200 ms success is saturated. At 400 ms the one-case gap falls below the prespecified exploratory threshold of two cases; this is not proof that delay has no effect. At 800 ms a delay effect is visible, but CV already recovers six successes. The remaining one-case oracle gap does not justify a complex predictor from eight seeds.

## Learning and outward control

All rule policies succeeded on all outward cases, with mean cost 9.040 s. Both learning policies failed every outward case by timeout. In crossing, MLP had one collision at every delay; its other failures were timeouts. GRU had no crossing successes, zero collisions at 0–400 ms and one collision at 800 ms. Its apparently low collision rate cannot be presented as improved team safety because task completion failed.

Learning baselines are insufficiently successful under this bounded budget. This does not establish that GRU/MARL cannot solve the task, nor that history is useless. A meaningful learned-method comparison would first need validated task learning and multiple training seeds, followed by untouched test scenarios.

## Next scientific decision

Keep this as a reproducible negative pilot. Before further method development, identify an externally motivated setting where CV breaks down (for example maneuvering peers with realistic communication traces), freeze that protocol, and rerun matched raw/CV/current-state controls. Do not increase delay or engineer maneuvers solely to force the proposed method to win. A new setting needs evidence of both a real information deficit and a meaningful control benefit beyond strong simple baselines. This pilot can support a methods/protocol draft, but not a positive effectiveness or novelty claim.

No statistical significance or generalization is claimed: eight development scenarios per cell, one learning seed, fixed delays, perfect measurement, and two route families are limited evidence. The information reference is not an optimal controller. Previous takeover findings remain unchanged and are not repurposed as positive evidence.
