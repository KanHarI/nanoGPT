# Proposed curriculum experiments

Originally requested as documentation only on 2026-10-17. The user subsequently
authorized implementing min/max with a substantially smaller model; see the
[min/max implementation](../minmax/README.md), [CPU and GPU results](../minmax/RESULTS.md),
and [lower-LR gamma-20 comparison](../minmax/GPU_GAMMA20_LOW_LR.md).
Symbolic differentiation remains documentation only. The shared protocol below
describes the intended controlled experiments; the initial min/max CPU recipes
are development budgets, not a calibrated 50-minute GPU curriculum.

| Proposal | Prediction | Main difficulty |
| --- | --- | --- |
| [Symbolic differentiation](SYMBOLIC_DIFFERENTIATION.md) | A derivative AST in RPN | Applying and composing differentiation rules; variable-length output |
| [Nested min/max and negation](NESTED_MIN_MAX.md) | One of 41 signed numeral tokens | Binary/ternary choices and unary negation; tree shape and RPN stack requirements |
| [Numerical integrals](NUMERICAL_INTEGRALS.md) | A compact sign/exponent/mantissa answer | Domain-aware ASTs, reliable numerical labels, and learning integration |

Numerical integration has a local AST preview, authorized on 2026-10-24, with
bounds-first RPN and saved samples at gamma 1, 3, 6, 12, and 20. Five displayed
examples also have checked numerical answers and bit-encoding previews. The later
[integral training implementation](../integrals/README.md) adds stable quantized
labels, full-sequence NLL, and per-gamma W&B metrics. It subsequently completed
the [H100 curriculum/shuffled comparison](../integrals/GPU_STAGED_120M.md).
Its generation and answer format choices are documented separately; the older
schedules below do not describe that numerical-integral run configuration.

The latest integral generator adds bounds -2 and 2 plus FLOOR, CEIL, and FRAC,
for 19 operators and 21 equally probable bound pairs. Its regenerated histogram
survey contains 1,000 accepted numerical values at every gamma 1..20, with failed
candidates replaced at the same K and bounds and logged separately. See the
[current survey and results](NUMERICAL_INTEGRALS.md#expanded-bounds-and-discontinuous-operators-current-survey).

Min/max is the suggested first follow-up: its answer has one character, has an
unambiguous exact score, and uses the same answer-only objective as finite-field
arithmetic. Differentiation introduces sequence generation and equivalent answer
representations as additional experimental variables. Neither task guarantees a
curriculum advantage; hard-only learning successfully is a valid outcome.

The min/max proposal includes binary and ternary min/max plus unary negation,
with one dedicated token for every integer from -20 through 20. Operators are
sampled without any alternation requirement; the size curriculum keeps their
probabilities fixed. Its signed numeral encoding is specific to that task and
does not change the differentiation proposal's vocabulary.

## Shared proposed comparison

Use fresh model weights and one paired seed initially. The existing architecture
(64 layers, width 512, 8 heads, FF width 2048) is a candidate for continuity in
the differentiation proposal. For min/max, the requested smaller starting model
is now 24 layers, width 256, 4 heads, FF width 1024 (18,931,968 parameters),
following the user's revision. Earlier CPU observations used 4 layers and width 128.
Benchmark batch size and context on the new task before freezing a run budget;
do not assume the arithmetic throughput, batch size, or update budget transfers.
Use linear LR warmup followed by constant LR, with the same optimizer settings
and initialization across conditions.

For total updates U and ramp length R >= 2, use zero-indexed update t:

`gamma(t) = 1 + 11 * min(t / (R - 1), 1)`.

The initial target is gamma 12. Calibrate R to approximately 50 minutes of
training, then freeze R and U before comparing arms. This is an update schedule;
evaluation, checkpointing, and runtime differences add elapsed time. A 40% ramp
fraction is a starting candidate from the previous experiment, not a fixed new
budget or a wall-clock stopping rule.

| Condition | Exposure |
| --- | --- |
| Curriculum | Gamma 1 to 12 during the ramp, then gamma 12 |
| Hard-only | Gamma 12 throughout |
| Shuffled | A permutation of exactly the curriculum's generated batches |
| Optional shuffled-ramp | Shuffle only the ramp batches; keep the exact same final gamma-12 batches in their original order |

Give each logical batch a fixed gamma, generator version, and independent seed.
Regenerate it from that identity when needed. Shuffling those identities matches
the actual ASTs, answers, and within-batch order without storing a dataset.
This is stronger than the previous finite-field control, which matched the
gamma multiset but did not replay identical examples. Shuffled-ramp additionally
matches the final period of hard training, addressing the recency difference
between curriculum and a fully shuffled stream.

Match updates and examples per update across all arms. Curriculum and shuffled
also have exactly matched input/output token totals. Hard-only can have a
different token total and compute cost; record both rather than claiming an
equal-compute comparison. Use the same fixed context and packing algorithm in
all arms, with complete newline-separated examples and no truncation.

Training and evaluation are procedural, with independent random streams. Retain
the previously agreed interpretation of train/test curves: two evaluation banks
from the same generator, neither used for optimizer updates. Fix monitoring
banks across checkpoints and conditions, and use larger fresh banks at the final
scheduled update. Natural repetition, especially among small expressions, means
these banks are not guaranteed to contain unseen ASTs.

Evaluate every integer gamma 1 through 12 and report task-specific diagnostics.
Freeze evaluation sizes and frequency after measuring cost. Do not stop the
hard-only arm based on its relative performance if a full-budget comparison is
the goal; explicitly mark any requested early stop and exclude it from that
comparison. One seed gives a preliminary comparison, not an across-seed
uncertainty estimate. The hypothesis is that curriculum helps, not that the
controls must fail.
