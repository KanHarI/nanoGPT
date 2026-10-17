# Workplan: finite-field curriculum experiment

Branch: `experiment/finite-field-curriculum`, based on the existing checkout.
Initial CPU experiments are complete. The user subsequently authorized the
H100 follow-up described in [GPU_RUN.md](GPU_RUN.md).

## Question and primary outcome

Does training from low to high AST complexity reduce final answer cross-entropy
on the fixed high-gamma distribution at equal updates and supervised examples?
The hypothesis is tested, not assumed. Faster convergence and accuracy are
secondary outcomes. A shuffled-mixture control tests whether ordering helps
beyond exposure to easy examples.

## Agreed task

- Prime fields of orders 2, 3, 5, 7, 11, 13, 17, 19, equally likely per item.
- Character-level GPT; one character per number, including `j` for field order 19.
- Random binary ASTs rendered in RPN, separated by newlines.
- Answer-only loss and metrics, following `=`.
- Procedural training/evaluation from the same sampler, with independent seeds.
- Gamma controls operator-count distribution; defaults and arithmetic conventions
  are specified in [README.md](README.md).
- Exponentiation is optional and **off for the first experiments**.
- Division is now optional and **off for new experiments**, following the user's
  request to eliminate division-by-zero rejection. Historical CPU runs used division.

## Execution stages

1. **Implementation:** isolate data, training, evaluation, and reporting from the
   original nanoGPT text-training script. Save full configurations and RNG state.
2. **Correctness:** exhaustively check binary arithmetic, validate generated ASTs
   with a separate stack evaluator, check field/size sampling, answer masking,
   causal isolation, evaluation independence, and exact CPU resume.
3. **Pipeline sanity:** overfit 16 fixed items; this is a diagnostic, not a
   procedural-generalization result.
4. **CPU pilot:** run hard/curriculum/mixture with seed 0, using the fixed pilot
   recipe (300 updates, batch 16, 2 layers, embedding width 64, gamma 1→16).
   Record runtime, label-frequency baselines, and final losses. This pilot cannot
   establish a reliable curriculum effect.
5. **Main CPU comparison:** freeze a practical recipe after pilot inspection and
   run five paired seeds. The supplied CPU preset is the initial candidate.
   Report every seed, final target loss, paired differences, uncertainty, and
   generalization across gamma/fields/sizes/depths.
6. **Follow-up:** if needed, increase budget or capacity symmetrically, vary target
   gamma, or run a separate power-enabled experiment. Keep these distinct from
   the original comparison. Decide about cloud hardware only after CPU evidence.

Stages 1–5 are complete for the supplied CPU recipe: correctness and overfitting
checks, the three-run pilot, and all 15 runs of the five-seed comparison.
See [RESULTS.md](RESULTS.md) for measured outcomes and their limitations. Stage 6
continues with the larger GPU recipe. A Google Cloud project and H100 VM have
been provisioned; batch/context calibration is complete. See [CLOUD.md](CLOUD.md).

## Larger-network follow-up

The initial losses are near uniform guessing within each known field:
`mean(log(p)) = 2.01095056` nats. The small curriculum model's 2.01594 loss and
37.7% literal-copy accuracy do not establish adequate arithmetic learning.

The first larger follow-up used [cpu_large.json](configs/cpu_large.json):
4 layers × 256 dimensions, 3.17M parameters, and 20,000 updates with 32 answers
per update. This is 7.8× the parameter count and 20× the supervised-answer budget
of the initial CPU recipe. Keep the arithmetic distribution, gamma range,
40% ramp fraction, and power-disabled setting fixed. Use learning rate 0.0003
and 200 warmup updates, as in the larger reference model's optimizer recipe.

Start with seed 0 across all three conditions, using identical initialization,
budgets, and evaluation streams. Inspect easy-gamma accuracy as well as high-gamma
loss relative to uniform and field-frequency baselines. Report the full outcome,
including a null result; do not treat a single seed as evidence of robustness.
Additional paired seeds follow only after assessing this larger training budget.

The user stopped this follow-up during the curriculum arm. Its latest durable
checkpoint is update 14,000; the other two arms never started. Results and the
observed loss of easy-item accuracy are recorded in [LARGER_CPU.md](LARGER_CPU.md).

## W&B and the next larger run

Connect [W&B](TRACKING.md) before launching another long run. Keep tracking
settings separate from experimental settings, verify that logging and online
resume preserve training, and import committed CPU metrics without rerunning
training. Record incomplete runs as checkpointed, not completed comparisons.

The authorized GPU follow-up uses 64 layers, width 512, 8 heads, FF width 2048,
and gamma 1–12. Both division and exponentiation are disabled. Benchmarking on
the H100 selected 1,024 expressions/update and 256-character packed contexts.
Freeze 48,210 updates per condition, including a linear gamma ramp over 19,284
updates (approximately 50 minutes of training). Use one seed for curriculum,
hard-only, and shuffled mixture. LR warms up for 482 updates and remains
constant at 0.0003 thereafter. W&B reports only independent procedural train/test
loss for each gamma/field; optimization diagnostics remain local. The systemd
training job starts with curriculum, then hard-only, then mixture and stops the
VM afterward. See [GPU_RUN.md](GPU_RUN.md) for the exact recipe and measurements.

The curriculum arm subsequently completed. The user requested stopping hard-only
early and proceeding to shuffled mixture. Its latest retained checkpoint is
update 17,688 (target 48,210); it is labelled `early_stopped` and excluded from
full-budget paired comparisons. The active run plan and monitor now require only
the completed curriculum arm and the shuffled arm before comparison/shutdown.

## Controls and interpretation

Keep model initialization paired by seed; match optimizer, learning-rate
schedule, expressions per batch, context width, total updates, and supervised answers.
Packed row counts and compute differ with gamma; wall time is not matched.
Use common evaluation seeds and a fresh final evaluation stream. Preserve field
and size when rejecting invalid division. Log actual sizes, operator frequencies,
rejections, input characters, and time to expose distribution/computation effects.

The primary result is at the final scheduled update, not the best observed
checkpoint. Evaluate all arms at high gamma even while curriculum trains on easy
items. A single seed is a pipeline result. With multiple seeds, report paired
seed-level intervals and the individual outcomes; do not inflate sample size by
treating individual evaluation expressions as separate training experiments.

If the shuffled mixture matches curriculum, easy-example exposure may explain
the benefit. If all arms mostly learn answer frequencies, the arithmetic-learning
question is still unresolved. A null or reversed effect is a valid result.
