# Audit of the unexpected shuffled curve

On 2026-10-17, the gamma-18 W&B panel showed curriculum and hard-only learning
while shuffled remained near loss 3, with a jump around update 13,000. This audit
found evidence of optimization instability in shuffled, but did **not** identify
an implementation error or establish the cause of that instability. Shuffled
was still running when these snapshots were taken.

Runs belong to group `gpu-24layer-gamma20-matched-updates`:
[curriculum](https://wandb.ai/kanhar/minmax-curriculum/runs/09e0db5a),
[hard-only](https://wandb.ai/kanhar/minmax-curriculum/runs/c9126dcd), and
[shuffled](https://wandb.ai/kanhar/minmax-curriculum/runs/fae1c271).
See the [run recipe](GPU_GAMMA20.md) for configuration and checkpoint locations.

## Checks against the actual runs

- All saved configurations differ only in `condition`. All 17 recorded source
  hashes match the local source for each run. The initial evaluation grids are
  exactly identical across the three conditions. Training constructs a fresh
  model and optimizer after resetting the seed for each condition.
- Every downloaded W&B loss point matches a local monitoring or final report:
  26 history rows for each completed arm and 19 for the shuffled snapshot.
- All 793 sampled training records have the expected batch ID, seed, gamma, and
  learning rate. The learning rate remains 0.0006 after 955 warmup updates.
- The complete 26,668-entry shuffled plan is exactly a permutation of the
  curriculum plan, including gamma and generator seed. Reconstructing 33 logged
  batches (8,448 expressions, including batches around the loss jump) reproduces
  the logged structural statistics and packing. An independent numerical stack
  evaluator agrees with every reconstructed answer; supervised target order
  also matches the examples.
- There is no shuffle-run restart in the captured runner/watchdog logs. The
  earlier Spot eviction interrupted curriculum, before either control began.

The [log audit record](results/gpu_gamma20_log_audit.json) includes run IDs,
counts, endpoints, and the largest recorded gradient norms. Raw downloaded
logs and diagnostic scripts are retained locally in
`out/minmax-gamma20-audit/`; this directory is ignored by Git.

## Instability in shuffled

The recorded norm is the global gradient norm **before** clipping to 1.0:

| Condition | Largest recorded norm | Update |
| --- | ---: | ---: |
| Curriculum | 40.72 | 16,910 |
| Hard-only | 26.30 | 20,520 |
| Shuffled | 33,675.69 | 14,345 |

These are maxima in the sampled logs (every 95 updates, plus the final update),
not measurements of every gradient. Shuffled already has a norm of 710.28 at
update 12,635, and reaches 16,982.68 at update 13,585. Its gamma-18 monitoring
loss rises from 3.0549 at update 12,518 to 3.5569 at update 13,656. Low-gamma
evaluation deteriorates at the same time. This is evidence of a training
instability; clipping means these numbers are **not** optimizer update norms.
The run was also learning slowly before the jump, so the spike alone does not
explain its entire disadvantage.

Shuffling operates on whole batches: each batch has one gamma, and gamma changes
between updates. It includes the entire curriculum ramp and plateau, not just
the ramp. Exact exposure matching holds over the completed schedule, not at
each prefix. Abrupt complexity changes are a possible contributor, but this
audit does not demonstrate that mechanism or prove that the LR is responsible.

## Independent checkpoint evaluation

The saved checkpoints were evaluated in a separate process on 512 newly
generated expressions at each of gamma 18 and 20, using the seed namespace
`screenshot-audit-20261017`. Labels were independently recomputed with a
numerical stack evaluator. The same examples were evaluated both with normal
packing and individually with only `<expression>=` in the input: no answer
tokens or previous expressions. Checkpoint weight hashes differ across arms.

Gamma-18 results on this diagnostic bank:

| Condition | Checkpoint update | Packed loss / accuracy | Prefix-only loss / accuracy |
| --- | ---: | ---: | ---: |
| Curriculum | 26,668 | 0.1059 / 97.07% | 0.1130 / 95.90% |
| Hard-only | 26,668 | 0.1388 / 95.31% | 0.2321 / 92.97% |
| Shuffled | 22,760 | 2.7915 / 24.02% | 2.8828 / 21.88% |

This reproduces the qualitative result without exposing answer tokens to the
model. The [checkpoint audit](results/gpu_gamma20_checkpoint_audit.json) also
contains gamma-20 results and weight hashes. These diagnostic measurements are
separate from the official final test banks, and the shuffled checkpoint has
fewer updates than the other two.

Shuffled completed while the audit was being documented. W&B marks run
`fae1c271` finished with all 26,668 updates, and its uploaded `final.json`
confirms the fresh final evaluation. Official results on 4,096 fresh test
examples per gamma are:

| Condition | Gamma-18 loss / accuracy | Gamma-20 loss / accuracy |
| --- | ---: | ---: |
| Curriculum | 0.1717 / 95.51% | 0.1834 / 95.09% |
| Hard-only | 0.1670 / 95.48% | 0.1858 / 94.75% |
| Shuffled | 2.3551 / 33.33% | 2.3906 / 32.96% |

The [matched final summary](results/gpu_gamma20_summary.json) was reconstructed
from the downloaded final reports using the existing comparison function,
which validates configuration, completed updates, and final test seed/count.
Shuffled recovered somewhat after the spike but still substantially
underperformed at the full matched budget.

Thus this seed supports faster early learning with curriculum, but not a
substantial final-loss advantage over hard-only at these gammas. It does not
support the hypothesis that hard-only cannot learn this task. Success at these
gammas also does not establish uniform mastery: at gamma 1 the final accuracies
are 41.87% for curriculum and 66.48% for hard-only.

No training code, hyperparameters, run identities, or update budgets were changed
by the audit. A useful follow-up diagnostic would be a fresh shuffled run with
a lower **constant** post-warmup LR and denser gradient logging. Any such run
would be a separate ablation; a fair new three-condition comparison would apply
the same changed recipe to all three arms. No additional training was launched.
