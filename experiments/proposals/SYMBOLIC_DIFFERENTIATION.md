# Proposal: symbolic differentiation in RPN

Status: **documentation only; not implemented**. This records the proposal
discussed on 2026-10-17. Use the [shared comparison protocol](README.md).

## Task and vocabulary

Given a function of one real variable x, predict its symbolic first derivative.
Use a character-level decoder and serialize both ASTs in reverse Polish notation.
The initial grammar is closed under differentiation:

| Token | Meaning | Arity |
| --- | --- | ---: |
| `x` | Variable | 0 |
| `0`, `1`, `2`, `3` | Integer constants | 0 |
| `+` | Addition | 2 |
| `-` | Subtraction | 2 |
| `*` | Multiplication | 2 |
| `N` | Negation | 1 |
| `S` | Sine | 1 |
| `C` | Cosine | 1 |
| `E` | Exponential | 1 |

`E` is the function exp, not a numeral or a leaf. Each numeral and each operator
has one character. Division, logarithms, and powers are possible later extensions,
outside the initial grammar. No finite-field modulus applies to this task.

Each line is `<function RPN>=<derivative RPN>\n`. The separator denotes the
differentiation task, not equality between the original function and its derivative.
Differentiation is always with respect to x, so no additional task token is needed.

The motivating example is
`d/dx(sin(x + exp(x))) = cos(x + exp(x)) * (1 + exp(x))`:

```text
xxE+S=xxE+C1xE+*
```

Other intended canonical examples:

```text
x=1
3=0
xS=xC
xC=xSN
xE=xE
xx*=xx+
xxS*=xSxxC*+
```

## Target construction

Generate a function AST, derive a second AST recursively, and serialize it using
fixed rule order. Proposed rules, with D denoting differentiation:

| Input | Raw derivative |
| --- | --- |
| x | 1 |
| constant | 0 |
| u + v | D(u) + D(v) |
| u - v | D(u) - D(v) |
| u * v | D(u) * v + u * D(v) |
| -u | -D(u) |
| sin(u) | cos(u) * D(u) |
| cos(u) | -(sin(u) * D(u)) |
| exp(u) | exp(u) * D(u) |

Use a small deterministic simplifier for the target: neutral zero/one arithmetic,
multiplication by zero, negation of zero, and double negation. Preserve operand
order and tree grouping otherwise. Do not expand, factor, collect like terms, or
fold arbitrary integer expressions into constants outside the numeral vocabulary.
These are canonical outputs of this procedure, not a unique algebraic normal form.
For instance, D(x*x) is represented as x+x, while 2*x remains mathematically valid.
Freeze the precise rewrite rules and their application order before generating data.

An independent symbolic system should verify generated targets on a representative
sample before any training. Do not let its general-purpose simplifier define the
training targets implicitly: it can reorder operands or introduce other functions.

## Procedural distribution and curriculum

Let gamma be the rate of a Poisson-distributed number K of input operator nodes,
conditioned on 0 <= K <= K_max. Start with gamma 1 through 12 and a candidate
K_max of 32. Unary and binary nodes each count as one operator. Choose an enabled
operator, allocate K-1 remaining operators to its children, and recurse; for a
binary node, choose the left allocation uniformly from 0 through K-1. This is
not uniform sampling over all unary/binary trees.

Keep operator and leaf probabilities fixed across gamma and conditions. A
starting candidate is uniform selection among the seven operators, with each
leaf x with probability 0.8 and each of 0, 1, 2, 3 with probability 0.05.
Inspect the resulting zero-derivative and constant-function rates before freezing
the distribution. Any rejection or explicit nontrivial-example mixture must be
specified and shared across arms; changing it during training would add another
curriculum variable. Gamma controls sampled input size, not simplified derivative
size or a guarantee of semantic difficulty.

All operators appear from the start. The first experiment increases composition
complexity while keeping the grammar fixed. Introducing new functions in stages
would be a separate experiment.

## Loss, context, and evaluation

Supervise all derivative characters and the terminating newline after `=`; mask
the input function, separator target, and padding. In next-token indexing, the
logits at `=` predict the first derivative character. The newline receives loss
because the model must learn when a variable-length answer ends.

Proposed training objective: average token cross-entropy within each expression,
then average over expressions. Also report token-weighted loss as a diagnostic.
This keeps long derivative targets from dominating solely through their length.

Differentiation can repeat input subtrees and produce much longer output. Choose
context from a safe bound on the complete input-plus-target length under K_max,
then benchmark packing. Do not silently truncate or reject long targets to fit
the arithmetic experiment's context. Record input/output lengths, tree depth,
chain-rule nesting, and zero-derivative rate at each gamma.

At each evaluation gamma, measure answer loss, unconstrained greedy exact-match
accuracy, valid RPN rate, and semantic derivative correctness. Generation begins
at `=` without the reference answer and ends at newline or the fixed length cap.
Missing termination, stack underflow, and a final stack size other than one are
invalid predictions. Report validity independently of loss.

Use symbolic equivalence checks for alternative valid answers; numerical checks
are supplementary evidence, not proofs. Bound verification time, report unresolved
cases separately, and track numerical failures from extreme nested exponentials.
Neither a timeout nor overflow should silently remove a difficult test example.

Additional evaluation should distinguish increased gamma under the same K cap
from genuinely unseen sizes beyond that cap, and test composition depth and rule
families. Fresh procedural samples alone are not a guarantee of unseen structures.

## Related work

[Lample and Charton, Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412)
provides precedent for procedural symbolic-expression tasks. This proposal uses
postfix differentiation, the specific restricted grammar above, and matched
curriculum controls; it is not a reproduction of that paper's experiment.
