# Completed H100 results

Both curriculum and shuffled mixture completed **48,210 updates** with seed 0.
At gamma 12, curriculum reached **0.11312** test loss versus **0.45671** for
shuffled mixture, a difference of **-0.34360 nats** (75.2% lower).
This single-seed result supports an ordering benefit on the target distribution.
Shuffled mixture also learned, and it retained lower loss at gammas 1–4.
The final curriculum model is therefore not better at every complexity.

Hard-only was stopped at the user's request. Its last retained checkpoint was
update 17,688 with monitored gamma-12 test loss 1.81649; the last observed
optimizer update was 18,180. That partial monitoring result is not a matched
final endpoint and does not establish that hard-only could never converge.

## Matched recipe and interpretation

The two completed conditions used the same initial weights, model, update count,
answers per update, learning rate, and evaluation banks. Shuffled mixture used
exactly the same multiset of scheduled gamma values in a shuffled order; it did
not replay the identical ASTs. Packing makes dense compute and elapsed time vary.
See [GPU_RUN.md](GPU_RUN.md) for the 201.5M-parameter model, measured batch/context
choice, constant learning rate after warmup, and linear gamma schedule.

These are final answer-only cross-entropies in nats, using fresh final banks of
128 examples per field, gamma, and split (1,024 examples per aggregate below).
Fields receive equal weight. Both “train” and “test” are independent procedural
evaluation banks; neither supplies optimizer updates. Monitoring curves used
smaller, separate banks, so their last values can differ from these endpoints.

Only one seed was run. There is no across-seed uncertainty estimate, and this
experiment does not isolate the effect of sample order from recency: curriculum
ends with sustained gamma-12 training while mixture remains shuffled.

## Final losses at every gamma

| Gamma | Curriculum train | Curriculum test | Shuffled train | Shuffled test |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.51574 | 0.56513 | 0.00594 | 0.03000 |
| 2 | 0.42815 | 0.33588 | 0.05564 | 0.04719 |
| 3 | 0.21123 | 0.17853 | 0.06423 | 0.07349 |
| 4 | 0.10275 | 0.08361 | 0.08154 | 0.07188 |
| 5 | 0.02936 | 0.02603 | 0.08862 | 0.09703 |
| 6 | 0.02224 | 0.02236 | 0.18219 | 0.16355 |
| 7 | 0.01584 | 0.01666 | 0.25035 | 0.22906 |
| 8 | 0.03694 | 0.04574 | 0.32477 | 0.29993 |
| 9 | 0.03149 | 0.02527 | 0.36075 | 0.32367 |
| 10 | 0.05595 | 0.04217 | 0.39115 | 0.38235 |
| 11 | 0.07412 | 0.06551 | 0.39310 | 0.44624 |
| 12 | 0.08612 | 0.11312 | 0.47815 | 0.45671 |

## Gamma-12 test loss by field

| Field | Curriculum | Shuffled |
| ---: | ---: | ---: |
| 2 | 0.01138 | 0.00007 |
| 3 | 0.00607 | 0.00200 |
| 5 | 0.05470 | 0.00360 |
| 7 | 0.10328 | 0.10909 |
| 11 | 0.02877 | 0.23549 |
| 13 | 0.11948 | 0.61245 |
| 17 | 0.27908 | 1.06881 |
| 19 | 0.30219 | 1.62222 |

## Source reports and checkpoints

The final JSON reports were downloaded from each finished W&B run's
`experiment-report` artifact without changing the measurements:

- [Curriculum final report](results/gpu_curriculum_final.json) · [W&B run](https://wandb.ai/kanhar/finite-field-curriculum/runs/5c9ddae9)
- [Shuffled final report](results/gpu_mixture_final.json) · [W&B run](https://wandb.ai/kanhar/finite-field-curriculum/runs/889b6983)
- [Summary, source references, and report hashes](results/gpu_summary.json)
- [Hard-only stop decision](results/gpu_run_plan.json) · [W&B run](https://wandb.ai/kanhar/finite-field-curriculum/runs/40282133)
- [Loss workspace](https://wandb.ai/kanhar/finite-field-curriculum?nw=7qw55xp483s)

The VM stopped at **2026-10-17 01:40 UTC** and its Compute Engine state was
confirmed `TERMINATED` later that morning. Model/optimizer checkpoints remain
on its retained persistent boot disk at:

```text
/home/kanhar/nanoGPT/out/gpu-64layer-gamma12/curriculum-seed0/checkpoint.pt
/home/kanhar/nanoGPT/out/gpu-64layer-gamma12/hard-seed0/checkpoint.pt
/home/kanhar/nanoGPT/out/gpu-64layer-gamma12/mixture-seed0/checkpoint.pt
```

The repository stores small reports and figures. It does not contain model
checkpoints or credentials. W&B holds the final report artifacts; model weights
were not uploaded. See [CLOUD.md](CLOUD.md) for access to the retained disk.
