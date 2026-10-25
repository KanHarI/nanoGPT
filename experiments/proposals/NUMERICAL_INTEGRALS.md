# Numerical integral curriculum: design, AST preview, and label survey

Status: the local preview and numerical survey were followed by the requested
[training implementation and CPU smoke test](../integrals/README.md), authorized
on 2026-10-24/25. That package adds a complete quantizer, stable-label acceptance,
full-sequence next-token NLL, and per-gamma W&B metrics. This document retains the
generator design and historical preview/survey results. No cloud deployment was
performed.

Current generator: `integral-ast-preview-v2-bounds2-rounding`, with bounds extended
to +/-2 and floor/ceiling/fractional-part operators. The latest survey fills 1,000
accepted numerical outcomes per gamma; rejected candidates are stored separately.
Historical examples and the original survey below retain their original grammar.

## Representation

Each item starts with **both integral bounds**, followed by the integrand in RPN:

```text
<lower bound> <upper bound> <integrand RPN> INT_X = <answer bits>\n
```

`INT_X` is a three-child AST node: lower bound, upper bound, integrand. This is
postorder traversal of that whole AST. The model sees the interval before it
reads the integrand. Every displayed token name is a single logical token;
spaces are for human readability, not a proposed character tokenizer.

For example, the motivating integral has input:

```text
-1 1 x COS SQRT INT_X =
```

The candidate-preview generator prints the input through `=`. The training labeler
adds a complete checked answer and newline, with packing only at complete-record
boundaries.

## Vocabulary

Integrand leaves: `x` and nine atomic constants:

| Tokens | Values |
| --- | --- |
| `-1`, `0`, `1`, `2`, `3` | The respective integers |
| `PI`, `E` | Mathematical pi and e |
| `HALF`, `THIRD` | Exact rational constants 1/2 and 1/3 |

There are 19 integrand operators, each counting as one node:

| Arity | Tokens |
| --- | --- |
| Binary | `+`, `-`, `*`, `/` |
| Unary | `NEG`, `SQUARE`, `SQRT`, `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`, `LN`, `EXP`, `W0`, `FLOOR`, `CEIL`, `FRAC` |

`EXP` means e raised to its argument. Trigonometric functions use radians, inverse
trigonometric functions use their principal real branches, and `W0` is the
principal real Lambert W branch. The preview samples bounds from
`-2, -1, NEG_HALF, 0, HALF, 1, 2`, where `NEG_HALF` is one atomic **bound** token
meaning -1/2. The new -2 bound is also atomic. Neither -2 nor NEG_HALF is an
additional integrand leaf in this preview; the requested nine constants remain.
Sampling two distinct grid points uniformly and sorting yields 21 intervals,
each with probability 1/21, independently of gamma.

`FLOOR(u)` is the largest integer <= u; `CEIL(u)` is the smallest integer >= u.
`FRAC(u) = u - FLOOR(u)`, so fractional part is nonnegative, including for negative
inputs: `FRAC(-1/3) = 2/3`. Each operation is one AST node and one logical token.

## Sampling choices for this preview

These probabilities are explicit starting assumptions, not calibrated settings:

1. Draw K from Poisson(gamma), conditioned on 0 <= K <= 32. Gamma is the Poisson
   parameter; an individual AST need not have exactly gamma operators. Its actual
   K is recorded. Conditioning changes the mean, especially near the cap.
2. Draw a pair of distinct points uniformly from the bounds grid and sort them.
3. At a non-leaf node, choose unary/binary with probability 1/2 each. A unary child
   gets K-1 operators. For binary nodes, split K-1 uniformly across the K possible
   ordered allocations. This is not uniform sampling over all AST shapes.
4. At a leaf, choose `x` with probability 1/2; otherwise sample uniformly from the
   nine constants.
5. Construct children first. Choose uniformly from eligible operators of the
   selected arity by trying a random permutation of that operator list.
6. If a tree cannot be completed, retry at the **same K and bounds**, with a finite
   failure limit. Never silently replace a difficult size with an easier one.
7. In the accepted-value survey, a failed numerical evaluation also causes a new
   AST to be generated at the **same K and bounds**. Each slot has an independent
   deterministic seed, so retries do not change later slots' sampled constraints.
   Rejected candidates are excluded from the accepted CSV and logged separately.

All operators are enabled at every gamma. Their realized frequencies vary because
the children determine admissibility. Whole-tree rejection can also bias the
accepted shapes; attempts and rejection reasons are recorded. Constants,
cancellations, double negation, and other algebraic simplifications remain in the
raw distribution; no simplification or operator alternation is imposed.

The outer `INT_X` node and constant leaves do not count toward K. A binary
multiplication `x x *` and a unary square `x SQUARE` both count as one operator,
but have different token lengths and tree shapes.

## Range propagation and numerical precautions

Every node carries an interval covering its possible values for all x in the
integration interval. This metadata is not in the model input.

- Addition, subtraction, multiplication, and division use interval arithmetic;
  division requires the denominator interval to exclude zero.
- Square includes its interior minimum when the input crosses zero. Sine and
  cosine include interior extrema. Tangent is eligible only if the cosine interval
  excludes zero, certifying that no pole lies in the whole input range.
- Log requires a strictly positive lower bound; square root requires a
  nonnegative lower bound. Arcsine/arccosine require the entire interval in [-1,1].
- Atan, exp, log, and W0 are increasing on their admitted domains; arccos decreases.
  W0 requires a lower bound certified to be at least -1/e. An inconclusive bound
  near that branch point is conservatively rejected.
- Floor maps [L,U] to [floor(L),floor(U)], and ceiling analogously. Fractional
  part maps an interval contained in one integer bin by subtraction of that bin's
  integer; if the interval crosses an integer, its conservative enclosure is
  [0,1]. The closed upper endpoint is an enclosure of a limiting value; actual
  FRAC values are strictly below 1.
- Bounds use directed rounding through mpmath interval arithmetic at 40 decimal
  digits. Scalar diagnostics run separately at 80 digits. Printed JSON interval
  endpoints are approximate displays, not reusable certified enclosures.

Two library limitations required explicit handling in the prototype:

1. Installed mpmath 1.3.0 does not provide working interval Lambert W. The preview
   proposes scalar endpoint estimates, then certifies brackets through the
   monotonic function w*exp(w) for w >= -1. A rounded scalar W value alone is not
   treated as a bound.
2. In that version, `iv.atan2(-1, 1)` at 40 digits fails to enclose -pi/4. The
   preview calls the underlying directed-rounding atan on monotonic endpoints.
   Arcsine uses `2*atan(u/(1+sqrt(1-u**2)))`; arccos uses pi/2 minus arcsine.
   A regression check covers the negative-angle failure.

mpmath describes its interval support as experimental, so these checks do not
amount to formal verification of the library or the eventual labeler. See its
[interval arithmetic documentation](https://mpmath.readthedocs.io/en/latest/contexts.html#arbitrary-precision-interval-arithmetic-iv).

**Preview-only resource guard:** reject a candidate operation if the propagated
interval exceeds magnitude 1,000,000. This keeps nested exponentials and numerical
diagnostics bounded. It is an extra generator restriction, can reject valid trees,
is fixed across gamma, and is recorded separately from domain failures. It is not
the proposed answer range, and its suitability for the final dataset is undecided.

Conservative ranges can be loose because repeated subexpressions are correlated.
For example, interval subtraction does not automatically recognize `x-x = 0`.
Future refinement could reduce unnecessary rejections without changing semantics.

Rounding operators introduce a separate numerical issue at integer boundaries.
For example, `floor(sin(pi))` must not evaluate to -1 because of a tiny negative
rounding residual. Near a jump, the evaluators use an interval enclosure at the
quadrature point to decide the side. If it cannot decide, the candidate is rejected
as `UncertainDiscontinuity`. This can reject mathematically valid expressions,
but avoids silently assigning unstable answers. Structural integer-valued
subtrees and exact integer intervals are recognized directly. The vectorized
refinement evaluator uses the same guarded scalar decisions at near-boundary
points, including positive exponentials that underflow to zero in float arithmetic.

## Output design agreed so far

The proposed answer has one sign bit, five exponent bits, and eight explicit
mantissa bits, with field separators and distinct bit-token families. Its value is

```text
(-1)^sign * (M / 256) * 2^(E - 16)
```

E ranges from 0 to 31 and M from 0 to 255. Use canonical normalization (M >= 128
except at minimum exponent) and canonical all-zero positive zero. This supports a
smallest positive magnitude of 2^-24 and maximum magnitude 32640. Following the
user's subsequent revision, training uses **NLL on all real next tokens**, masking
only padding. Answer-bit NLL remains the main task metric, with separate sign,
exponent, mantissa, non-answer, and full-sequence losses. Final integral range
checks and stable rounding under quadrature refinement are separate from AST
domain checks. The candidate generator does not perform them; the new
[training labeler and complete quantizer](../integrals/README.md#numeric-labels)
do. The selected-answer preview below is a historical normal-value encoding check
for five examples.

## Initial examples (historical v1) and current candidate preview

The saved `integral_ast_samples.json` is the original v1 artifact: five bounds and
16 operators. To inspect candidates from the **current** generator without
overwriting that historical artifact:

```bash
.venv/bin/python experiments/proposals/integral_ast_preview.py \
  --gammas 1 3 6 12 20 --count 10 --audit-count 200 \
  --seed 20261024 \
  --output out/integral-ast-preview-v2/candidates.json

.venv/bin/python -m unittest experiments.proposals.test_integral_ast_preview -v
```

The [historical saved JSON](integral_ast_samples.json) contains the first ten unfiltered
samples at each gamma, with parsed expressions, bounds-first RPN, actual K, depth,
and interval displays. Each gamma has its own deterministic seed, so changing the
order or number of other gamma levels does not change its samples. Audit counts
include the saved samples, rather than adding a second batch of that size.

Original v1 validation: ten domain/rounding/serialization regression tests, plus 200 sampled
trees at each of five gamma levels. For every tree, 17 x values including both
endpoints are checked at every node against the stored bounds; an independent
RPN stack evaluation is compared with recursive AST evaluation. These grid checks
are diagnostics, not the mechanism for certifying domains. All 16 operators occur
in this audit. The saved examples are not yet labeled training data.
The current candidate-only CLI records uncertain point audits explicitly; these
raw AST candidates are distinct from the accepted-value survey dataset.

## Numerical answers for the five displayed examples

At the user's request, [integral_answer_preview.py](integral_answer_preview.py)
evaluates the five examples previously shown in the conversation, without changing
their ASTs or bounds. Reproduce with:

```bash
.venv/bin/python -m experiments.proposals.integral_answer_preview
```

It uses the existing RPN evaluator for tanh-sinh integration at 40 digits with four
subintervals and 70 digits with eight subintervals. Independent checks use an
analytic antiderivative for the gamma-1 example, and separately written formulas
with 50-digit Gauss-Legendre integration for the others. The gamma-12 check uses
the change of variables x=-1+t^2 to smooth the acos endpoint. All three estimates
must agree to `1e-30 * max(1, abs(answer))` and round to the same bit sequence.
This is a numerical consistency check, not formally certified quadrature.

The [saved answer JSON](integral_answer_samples.json) contains the original inputs,
approximations, estimated quadrature errors, differences between checks, and bit
previews. Decimal values below are approximate; decoded quantized values are exact.
The bit display is `sign | exponent | mantissa`, MSB first within each field.
Actual training vocabulary would use the distinct field-specific bit tokens and
separators discussed above.

| Gamma | Actual K | Integral, approximately | Answer bits | Decoded quantized value |
| ---: | ---: | ---: | --- | ---: |
| 1 | 1 | 0.442971535211 | `0 / 01111 / 11100011` | 0.443359375 |
| 3 | 4 | 0.0922388694831 | `0 / 01101 / 10111101` | 0.09228515625 |
| 6 | 7 | 0.400216049882 | `0 / 01111 / 11001101` | 0.400390625 |
| 12 | 13 | 0.620271567843 | `0 / 10000 / 10011111` | 0.62109375 |
| 20 | 18 | -4.09665436207 | `1 / 10011 / 10000011` | -4.09375 |

The normal-value encoding helper is deliberately limited to these nonzero normal
examples. It is not the full dataset quantizer and does not implement the pending
zero/subnormal and ambiguous-rounding policies. The remaining 45 saved AST samples
are still unlabeled.

## Original v1 survey: 1,000 evaluated expressions per gamma

The user subsequently requested a background survey of the **numerical integral
values**, rather than AST sizes. The local CPU survey drew exactly 1,000 ASTs at
each integer gamma 1..20 (20,000 total), with the same generator version, seed
20261024, bounds grid, and preview magnitude guard. Each gamma draws two distinct
points uniformly from {-1,-1/2,0,1/2,1}; sorting gives ten equally probable
intervals. Bounds are chosen before the integrand and retained on generation
retries. Each gamma's first ten ASTs match the earlier saved preview exactly.

No answer quantization or output-range filter is applied. Each sampled AST is
retained even if numerical integration fails; no easier replacement is drawn.
This preserves the requested sample set and makes numerical exclusions visible.

The [survey implementation](integral_distribution.py) checks fast scalar evaluation
against 50-digit AST evaluations at two interior points. It requires two converged
QUADPACK results with changed subdivisions/tolerances to agree within
`1e-8 * max(1, abs(value))`, with suitably small estimated errors. Candidate passes
use 1, 7, 31, and 63 initial pieces and subdivision limits of 300, 800, 2,000, and
4,000, stopping after two successful agreeing passes. It also requests high-precision
tanh-sinh audits every 200 samples and uses a 25/40-digit fallback for failures.
See [SciPy's quad documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
for the meaning and limitations of the numerical error estimates.

The initial pass accepted 19,950 results. A [refinement pass](integral_distribution_refine.py)
revisited those same 50 unresolved ASTs using vectorized composite Gauss-Legendre
quadrature with a cosine change of variables to smooth common endpoint behavior.
It compares orders 32 and 48 over progressively refined meshes, then checks order
40 on a doubled mesh. Agreement thresholds remain `1e-8 * max(1, abs(value))`.
This recovered 45 values. These numerical consistency checks are not a formal
integration certificate or a complete training-label acceptance policy.

Final outcome (see [saved summary](integral_distribution_summary.json)):

- **20,000 ASTs drawn and evaluated; 19,995 accepted (99.975%).**
- Five unresolved cases: gamma/sample 8/417, 10/44, 11/191, 17/799, 20/195.
  They involve strong oscillation or nonsmooth inverse-trigonometric compositions.
  Their ASTs and failure reasons are retained; they are excluded from histograms.
- 1,188 accepted values satisfy |value| <= 1e-8. This is a numerical near-zero
  category, not a symbolic proof of an exactly zero integral.
- Accepted values range from approximately -278,222.77 to +287,233.12. Eleven
  exceed the proposed answer range of +/-32,640 and remain included in the survey.
- The overall median is approximately 0.271; the central 98% are approximately
  between -8.16 and +51.95. Means are strongly affected by the tails.

Artifacts are under `out/integral-value-survey/`:

| Artifact | Purpose |
| --- | --- |
| `histograms.html` | Standalone interactive report: gamma selection, linear/signed-log value axes, adjustable bins, shared ranges, optional log counts and near-zero filtering; click a bin to inspect its expressions |
| `histograms.png`, `histograms.pdf` | Twenty histograms with identical bins and axes |
| `heatmap.png`, `heatmap.pdf` | Gamma versus value, colored by percentage per bin |
| `values.csv` | All 20,000 outcomes, including unresolved statuses and RPN inputs |
| `summary.json`, `summary.csv` | Counts, quantiles, ranges, and diagnostic rates by gamma |
| `records/gamma_XX.jsonl` | Complete per-expression records, numerical checks, and retained initial failures |

The default x-axis transform is `sign(y)*log10(1+abs(y))`, preserving both signs
and zero. Bins are equally spaced in that transformed coordinate. Bar heights
are **counts per bin, not densities per unit of untransformed value**. The heatmap
uses percentages and logarithmic color intensity to reveal rare tails. No tail
clipping is used. The interactive report offers linear axes for comparison.

Reproduce using the [pinned local dependencies](requirements-integral-preview.txt):

```bash
uv pip install --python .venv/bin/python -r experiments/proposals/requirements-integral-preview.txt
.venv/bin/python -m experiments.proposals.integral_distribution \
  --count 1000 --gammas 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --workers 6 --output out/integral-value-survey
.venv/bin/python -m experiments.proposals.integral_distribution_refine \
  --output out/integral-value-survey
.venv/bin/python -m unittest experiments.proposals.test_integral_distribution -v
```

The original survey and refinement were launched as detached local background
processes with logs under the output directory. The report was refreshed as
results arrived. Completed runs can be inspected without a server or connection.
Six additional numerical/plotting tests compare the earlier high-precision
examples, known signed/odd integrals, the endpoint transform, constant integrands,
and histogram coverage of zero and both tails.

## Expanded bounds and discontinuous operators: current survey

The subsequent requested rerun combines both changes: the seven-point bounds
grid {-2,-1,-1/2,0,1/2,1,2} and all 19 operations, including FLOOR, CEIL, and FRAC.
It targets **1,000 accepted integral values at each gamma 1..20**. Numerical
failures are replaced at the same sampled K and bounds rather than remaining in
the exported dataset. The JSONL rejection logs retain the candidate's original
slot, constraints, RPN, numerical diagnostics, and failure reason.

Each slot uses a seed derived from generator version, seed 20261024, gamma, and
slot index. It has a limit of 100 candidate attempts, failing explicitly if that
slot cannot be filled. It never falls back to a different K or a shorter interval.
This preserves the distribution of slot constraints; the accepted AST distribution
is still conditioned on numerical evaluability. Comparison with the first survey
also changes the grammar and sampling scheme, so any distribution differences
cannot be attributed to the bound change alone.

The background job uses six CPU workers, the same quadrature tolerance policy,
20-second initial evaluation budgets and 30-second refinement budgets per
candidate. The 1e6 per-node range guard remains. Results are unquantized; values
outside the proposed final answer format are retained for this distribution
diagnostic. Full bit-label acceptance and output-range filtering are implemented
separately in the training package; these historical survey values are unchanged.

Current artifacts are under `out/integral-value-survey-bounds2-rounding/`:

- `histograms.html`: interactive report with current bounds and rejection counts.
- `histograms.png` / `.pdf`, `heatmap.png` / `.pdf`: regenerated plots.
- `values.csv`, `records/gamma_XX.jsonl`: accepted outcomes only.
- `rejections/gamma_XX.jsonl`: excluded candidates and reasons.
- `source_snapshot/`: source files and hashes for the executed version.

Reproduce the current accepted-value survey:

```bash
.venv/bin/python -m experiments.proposals.integral_distribution \
  --count 1000 --workers 6 --output out/integral-value-survey-bounds2-rounding
.venv/bin/python -m unittest \
  experiments.proposals.test_integral_ast_preview \
  experiments.proposals.test_integral_distribution -v
```

The tests cover all 21 possible bound pairs, fixed-K/bounds replacements, retry
independence from future slots, restart behavior, negative fractional parts,
interval wrapping, known floor/ceiling/FRAC integrals, rejection of ambiguous
integer-boundary decisions, and vectorized ceiling of underflowing positive values.

Final current-run results are in
[integral_distribution_bounds2_rounding_summary.json](integral_distribution_bounds2_rounding_summary.json):
20,000 accepted outcomes (exactly 1,000 at every gamma), from 21,120 candidates;
1,120 failures were excluded and replaced with the original slot constraints.
All 21 possible bound pairs and all 19 operators appear. The accepted ASTs contain
8,722 FLOOR, 8,767 CEIL, and 8,850 FRAC nodes. There are 2,015 numerical near-zero
results and seven values outside the proposed answer format; those seven remain
in this unquantized histogram diagnostic. Values range from about -97,585.82 to
+913,181.94.

The final audit rechecked all 35 composite-refinement results with the vectorized
integer-boundary guard, with zero changes to their values. Audit details are in
`vector_guard_audit.json`; reviewed source hashes are under
`source_snapshot/final_review/`. All 24 tests pass. The final CSV was checked to
contain only 20,000 accepted rows, and every excluded candidate was checked against
its replacement's K and bounds. These data remain numerical distribution previews,
not a fully quantized training corpus.
