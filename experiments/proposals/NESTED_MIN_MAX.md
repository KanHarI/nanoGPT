# Proposal: nested min/max and negation curriculum

Status: **implemented, with completed CPU and GPU experiments** in
[experiments/minmax](../minmax/README.md), with a smaller
18,931,968-parameter main model (24 layers, width 256), following the user's
revision. The initial CPU learning measurements used 4 layers and width 128.
This design includes binary and ternary min/max,
unary negation, and single-token integers from -20 through 20. There is no
generator-enforced alternation. Use the [shared comparison protocol](README.md).
The [completed results](../minmax/RESULTS.md) include gamma-12 and gamma-20
comparisons plus a lower-LR repeat; extended diagnostic suites remain proposals.

## Question

Does increasing expression size gradually help a character-level model learn
hierarchical min/max and negation evaluation, compared with starting at the target
complexity or shuffling the exact same examples?

This task has a single, exact, one-character answer. It isolates evaluation of
nested choices without symbolic output generation or finite-field arithmetic.
It is suitable for a curriculum experiment, but a curriculum advantage and
failure of hard-only training are hypotheses, not assumptions.

## Vocabulary and the motivating example

Use **41 distinct numeral tokens**, one for each integer from -20 through 20,
inclusive. Leaves are sampled uniformly and independently from all 41 values;
repeats are allowed. These are ordinary ordered integers, not field elements.
There is no field prefix, and a negative literal never needs a separate sign token.

A proposed one-character encoding preserves the character-level model:

| Numeral tokens | Integer values |
| --- | --- |
| `0` through `9` | 0 through 9 |
| `a` through `k` | 10 through 20 |
| `A` through `T` | -1 through -20, respectively |

For example, `A` means -1, `J` means -10, `T` means -20, and `k` means 20.
Integer comparisons use the decoded values, not the alphabetic order or token IDs.
Human-facing displays may show `-20`; the model sees its single token `T`.

Include all five operators from the start, each with a distinct fixed-arity token:

| Token | Meaning | Arity |
| --- | --- | ---: |
| `m` | Binary minimum | 2 |
| `n` | Binary maximum | 2 |
| `t` | Ternary minimum | 3 |
| `u` | Ternary maximum | 3 |
| `~` | Negation | 1 |

This replaces the earlier `m`/`M`/`t`/`T` operator proposal: uppercase `M` and `T`
now denote negative integers. The proposed vocabulary has 48 characters: 41
numerals, five operators, `=`, and newline. No operator collides with a numeral.
Negation is a real AST operation, separate from an atomic negative literal:
`C` represents -3 directly, while `3~` computes it by negating 3.

Min/max select an operand and negation reverses its sign, so every intermediate
value and answer remains in [-20, 20]. No clipping, overflow convention, or
additional numeral tokens are needed.

Each line is `<expression RPN>=<answer numeral>\n`. Because arities are fixed,
postfix expressions need no commas, spaces, parentheses, or explicit length tokens.

The user's example is:

```text
min(max(1, 2, 0), max(3, 1, 2)) = min(2, 3) = 2
```

Preserve its ternary max nodes and binary min root directly:

```text
120u312um=2
```

This AST has three operators and six leaves. Do not lower ternary operators into
binary chains: both arities are part of the initial task, and each node counts
once when measuring size. Other examples under the proposed encoding:

```text
C2m=C
3~=C
T~=k
3~2n=2
A02t=A
```

These mean min(-3, 2) = -3; neg(3) = -3; neg(-20) = 20;
max(-3, 2) = 2; and min(-1, 0, 2) = -1. Fixed arities make every expression
unambiguous even though unary, binary, and ternary nodes can be freely nested.

## Make grouping matter

Sample operators independently of the parent operator. **Do not enforce min/max
alternation**, reject same-operator chains, require a mixture of operator types,
or filter semantically easy trees. All-min and all-max trees, repeated values,
ties, and consecutive negations are legitimate samples. Keep the sampled AST
unsimplified so that the data distribution is defined by generation alone.

Without negation, an all-min tree computes the global minimum and an all-max
tree the global maximum. Double negation and other identities can also make a
large AST semantically simple. Record these properties as diagnostics rather than
altering the generator to remove them. Size is a proxy for difficulty, not a
guarantee that every larger tree is harder.

A useful structural diagnostic uses the same numeral multiset, operator counts,
and root operator, but changes grouping:

```text
min(max(0, 1), max(2, 3)) = 1
01n23nm=1

min(max(0, 3), max(1, 2)) = 2
03n12nm=2
```

A predictor that only uses the bag of leaves and root operator cannot get both
right. Construct paired evaluation cases of this kind; report them separately
from samples drawn from the training distribution.

Separately report diagnostic slices for same-operator chains, alternating paths,
mixed arities, and negation. Hand-constructed pairs are additional evaluations,
not constraints or rejection criteria on training examples. Hold operator
probabilities fixed throughout the size curriculum.

## Gamma and tree shape

For each example:

1. Sample K from Poisson(gamma), conditioned on 0 <= K <= 32.
2. At K = 0, sample one of the 41 numerals uniformly. Otherwise choose one of
   `m`, `n`, `t`, `u`, `~` with fixed probabilities, independently of its ancestors.
   The initial proposal is probability 1/5 for each operator, at every gamma.
3. For the chosen arity a, distribute the remaining K-1 operator nodes uniformly
   among all ordered a-tuples of nonnegative counts summing to K-1, and recurse
   on the children in order. Negation has one child; binary and ternary nodes
   have two and three. This is not uniform over all possible tree shapes.
4. Evaluate exactly and render the unsimplified tree in RPN.

Use gamma 1 through 12 initially, following the shared linear schedule. Count
unary, binary, and ternary operators equally: `K = N1 + N2 + N3`. Such a tree has
`1 + N2 + 2*N3` leaves, and `1 + N1 + 2*N2 + 3*N3` expression characters. A complete
line, including `=`, the answer, and newline, therefore has
`4 + N1 + 2*N2 + 3*N3` characters, at most `3*K + 4`. At K = 32, the maximum is
100 characters. A 128-character context can accommodate every example; larger
contexts may improve packing. Benchmark before fixing context and batch size.

Record three separate complexity measurements:

- Operator count K, counts by arity, and actual expression length. Unlike a
  binary-only tree, K alone no longer determines length.
- AST depth, negation count, and maximum min/max alternations along a root-to-leaf
  path, ignoring intervening negation nodes for this structural diagnostic.
- Maximum stack occupancy during left-to-right RPN evaluation.

AST depth and RPN stack occupancy are not interchangeable: left- and right-heavy
trees can have the same depth but require different stack capacity. Gamma is a
size-distribution parameter, not exact depth or guaranteed monotonic difficulty.
For example, a fully balanced binary tree of depth 12 has 4,095 operators and
8,191 expression tokens; gamma 12 here does not mean depth 12.

A later experiment could hold K fixed and vary tree shape/stack occupancy to
separate length from working-memory difficulty. Do not combine that intervention
with the initial size curriculum. Report balanced, skewed, and mirrored-tree
diagnostics at comparable sizes to assess those distinctions.

Negation can exchange min and max through identities such as
`-min(a, b) = max(-a, -b)`. Consequently, the structural alternation count alone
does not capture all semantic complexity; report it alongside negation and stack
usage, without normalizing or filtering the generated trees.

## Training and evaluation

Use the same three primary arms, optionally the shuffled-ramp control, and the
same warmup-plus-constant LR described in the shared protocol. Exact shuffled
batches match signed numeral values, arities, negations, ties, trees, gamma
exposure, and answers. The hard-only arm samples gamma 12 throughout, with the
same underlying generator.

Loss is cross-entropy over the one answer character after `=`, using the full
vocabulary softmax. Mask all other positions, including newline and padding.
At evaluation, predict that single token; punctuation or an operator is wrong.
Use whole-line packing and ordinary causal attention as in the finite-field
experiment; earlier complete examples can remain visible within each context.

The primary endpoint is mean final answer cross-entropy at gamma 12 on fresh
procedural samples, with accuracy as a secondary endpoint. Monitor independent
train/test banks at every gamma 1 through 12. There are no field-specific curves;
split diagnostics by root operator (all five choices), operator count, arity,
negation count, depth, stack occupancy, and answer value. Report K = 0 cases
separately from root-operator slices.

The answer is a leaf value or its negation, so it need not occur literally among
the leaves. It always has the magnitude of some leaf, but is not uniformly
distributed. Symmetric integer sampling and equal min/max probabilities do not
by themselves imply a uniform answer distribution.
Use a separate calibration stream to fit label-frequency baselines at each gamma,
including a root-conditioned baseline. Also evaluate global-minimum/global-maximum
heuristics over the literal leaf values and a root-selected global-extreme
heuristic on min/max-root slices; these deliberately ignore internal negation
and grouping. Report their coverage and the root-negation slice separately.
Uniform guessing among the 41 numeral values has loss ln(41), approximately
3.7136 nats, but exceeding that baseline alone is weak evidence that the model
learned the tree computation.

Keep the natural generator distribution as the primary evaluation. Add balanced
answer/root slices and structure-sensitive pairs as clearly labelled diagnostics;
do not silently rebalance the main task or suppress ties. A model should succeed
on pairs with identical leaf bags and different answers before claiming it has
learned hierarchical evaluation.

Evaluate larger gamma values, such as 16 and 24, as distribution-shift probes.
With the same K <= 32 cap these are not strictly unseen expression sizes. A true
size-extrapolation test must use K > 32 and sufficient context, specified separately.

Implemented correctness checks cover independent AST
and stack evaluation, all five operator arities, the displayed examples, the
41-value encoding and inverse mapping, range closure under negation, answer
masks, packing boundaries, and exact seeded batch replay. See the
[implementation and test commands](../minmax/README.md). This document retains
the broader experimental design, including evaluations beyond the initial suite.

## Related work

[Nangia and Bowman, ListOps](https://aclanthology.org/N18-4013/)
uses nested list operations including min and max as a diagnostic for learning
structure. Its representation is prefix-style and its operator set is broader.
This proposal is a restricted postfix task with a controlled size curriculum;
it is not the standard ListOps benchmark or a claim of a new task family.
