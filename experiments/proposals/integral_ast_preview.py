"""Local AST preview only: range-aware generation, no labels or training.

Run with the repository's .venv (mpmath 1.3.0 is already installed):
  .venv/bin/python experiments/proposals/integral_ast_preview.py

Whitespace separates readable names for atomic tokens; it is not tokenization.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

from mpmath import iv, mp
from mpmath.libmp import mpf_atan, round_ceiling, round_floor

# This standalone, single-threaded preview uses separate interval/scalar precision.
iv.dps = 40
mp.dps = 80
VERSION = "integral-ast-preview-v2-bounds2-rounding"
CONSTANTS = ("-1", "0", "1", "PI", "E", "2", "3", "HALF", "THIRD")
BOUND_TOKENS = ("-2", "-1", "NEG_HALF", "0", "HALF", "1", "2")
ROUNDING_OPS = ("FLOOR", "CEIL", "FRAC")
UNARY = ("NEG", "SQUARE", "SQRT", "SIN", "COS", "TAN", "ASIN", "ACOS",
         "ATAN", "LN", "EXP", "W0", *ROUNDING_OPS)
BINARY = ("+", "-", "*", "/")
ARITY = {**dict.fromkeys(UNARY, 1), **dict.fromkeys(BINARY, 2)}
# Preview resource guard, not the answer encoding's range or a finalized dataset rule.
PREVIEW_MAGNITUDE_LIMIT = mp.mpf("1e6")


class Ineligible(ValueError):
    pass


class UncertainDiscontinuity(ValueError):
    """A numerical enclosure cannot decide which side of a jump is intended."""


def constant(token, ctx):
    if token == "PI":
        return +ctx.pi
    if token == "E":
        return ctx.exp(1)
    if token in ("HALF", "THIRD", "NEG_HALF"):
        return ctx.mpf(-1 if token == "NEG_HALF" else 1) / (3 if token == "THIRD" else 2)
    return ctx.mpf(token)


def low(value):
    return mp.mpf(value.a)


def high(value):
    return mp.mpf(value.b)


def interval_atan(value):
    """Monotone endpoints with directed rounding, avoiding iv.atan2 in 1.3.0.

    iv.atan2(-1, 1) fails to enclose -pi/4 in that version because of an
    intermediate rounding step. Use the underlying directed atan directly.
    """
    return iv.make_mpf((mpf_atan(value._mpi_[0], iv.prec, round_floor),
                        mpf_atan(value._mpi_[1], iv.prec, round_ceiling)))


def lambert_endpoint(z):
    """Enclose W0(z) by verifying a bracket through monotone w*exp(w).

    mp.lambertw proposes the bracket; directed interval arithmetic verifies it.
    Thus this does not assume that a rounded scalar W result is an enclosure.
    """
    if z == 0:
        return iv.mpf(0)
    estimate = mp.re(mp.lambertw(z, 0))
    radius = mp.mpf("1e-30") * max(1, abs(estimate))
    for _ in range(40):
        left = iv.mpf(max(mp.mpf(-1), estimate - radius)).a
        right = iv.mpf(estimate + radius).b
        left_ok = low(left) == -1 or high(left * iv.exp(left)) <= z
        right_ok = low(right * iv.exp(right)) >= z
        if left_ok and right_ok:
            return iv.mpf([left, right])
        radius *= 10
    raise Ineligible("W0 bracket could not be certified")


def interval_apply(token, values):
    """Conservative interval extension, with domain checks before evaluation."""
    u = values[0]
    L, U = low(u), high(u)
    if token == "+":
        result = u + values[1]
    elif token == "-":
        result = u - values[1]
    elif token == "*":
        result = u * values[1]
    elif token == "/":
        v = values[1]
        if low(v) <= 0 <= high(v):
            raise Ineligible("denominator interval includes zero")
        result = u / v
    elif token == "NEG":
        result = -u
    elif token == "SQUARE":
        result = u ** 2  # Interval integer power includes the interior minimum.
    elif token == "FLOOR":
        result = iv.mpf([mp.floor(L), mp.floor(U)])
    elif token == "CEIL":
        result = iv.mpf([mp.ceil(L), mp.ceil(U)])
    elif token == "FRAC":
        # Fractional part wraps at every integer. The closed enclosure [0,1]
        # includes the limiting value 1 even though frac itself is always <1.
        result = u - mp.floor(L) if mp.floor(L) == mp.floor(U) else iv.mpf([0, 1])
    elif token == "SQRT":
        if L < 0:
            raise Ineligible("sqrt input not certified nonnegative")
        result = iv.sqrt(u)
    elif token in ("SIN", "COS"):
        # mpmath's interval versions include interior extrema.
        result = (iv.sin if token == "SIN" else iv.cos)(u)
    elif token == "TAN":
        cosine = iv.cos(u)
        if low(cosine) <= 0 <= high(cosine):
            raise Ineligible("tangent interval may contain a pole")
        result = iv.tan(u)
    elif token in ("ASIN", "ACOS"):
        if L < -1 or U > 1:
            raise Ineligible("inverse trig input outside [-1,1]")
        # asin(u) = 2 atan(u / (1 + sqrt(1-u^2))), also valid at +/-1.
        radicand = 1 - u ** 2
        sine_inverse = 2 * interval_atan(u / (1 + iv.sqrt(radicand)))
        result = sine_inverse if token == "ASIN" else iv.pi / 2 - sine_inverse
        if token == "ACOS":
            # Intersect with the known principal range to remove rounding spill.
            result = iv.mpf([max(iv.mpf(0), result.a), min(iv.pi.b, result.b)])
    elif token == "ATAN":
        result = interval_atan(u)
    elif token == "LN":
        if L <= 0:
            raise Ineligible("log input not certified positive")
        result = iv.ln(u)
    elif token == "EXP":
        # Avoid computing an exponential that necessarily exceeds the preview cap.
        if U > mp.log(PREVIEW_MAGNITUDE_LIMIT):
            raise Ineligible("preview magnitude guard")
        result = iv.exp(u)
    elif token == "W0":
        branch_point = -1 / iv.exp(1)
        if L < high(branch_point):
            raise Ineligible("W0 input not certified >= -1/e")
        result = iv.mpf([lambert_endpoint(L).a, lambert_endpoint(U).b])
    else:
        raise ValueError(f"Unknown operator {token}")
    if max(abs(low(result)), abs(high(result))) > PREVIEW_MAGNITUDE_LIMIT:
        raise Ineligible("preview magnitude guard")
    if not (mp.isfinite(low(result)) and mp.isfinite(high(result))):
        raise Ineligible("nonfinite interval")
    return result


def scalar_apply(token, values):
    if token in BINARY:
        a, b = values
        return {"+": lambda: a + b, "-": lambda: a - b,
                "*": lambda: a * b, "/": lambda: a / b}[token]()
    functions = {"NEG": lambda x: -x, "SQUARE": lambda x: x*x,
                 "SQRT": mp.sqrt, "SIN": mp.sin, "COS": mp.cos, "TAN": mp.tan,
                 "ASIN": mp.asin, "ACOS": mp.acos, "ATAN": mp.atan,
                 "LN": mp.log, "EXP": mp.exp, "W0": mp.lambertw}
    functions.update({"FLOOR": mp.floor, "CEIL": mp.ceil,
                      "FRAC": lambda x: x-mp.floor(x)})
    return functions[token](values[0])


def point_interval(node, x):
    """Recompute an interval at one exact numerical quadrature argument."""
    if not node.children:
        return iv.mpf(x) if node.token == "x" else constant(node.token, iv)
    return interval_apply(node.token, [point_interval(c, x) for c in node.children])


def integer_valued(node):
    if node.token in ("FLOOR", "CEIL"):
        return True
    if node.token in ("-1", "0", "1", "2", "3"):
        return True
    if node.token in ("+", "-", "*", "NEG", "SQUARE"):
        return all(integer_valued(child) for child in node.children)
    if node.interval is not None:
        with mp.workdps(max(mp.dps, iv.dps+20)):
            L, U = low(node.interval), high(node.interval)
            return L == U and L == mp.floor(L)
    return False


def checked_rounding(token, child, x, value, guard=None):
    """Reject ambiguous jump-side decisions instead of trusting rounded floats.

    In particular floor(sin(pi)) must not become -1 just because a library's
    approximation to sin(pi) has a tiny negative residual. Such undecidable
    enclosures are rejected; no symbolic simplifier is introduced here.
    """
    guard = 64*mp.eps if guard is None else mp.mpf(guard)
    rounding = mp.ceil if token == "CEIL" else mp.floor
    if integer_valued(child) or abs(value-mp.nint(value)) > guard*max(1,abs(value)):
        rounded = rounding(value)
    else:
        # Preserve the interval endpoints when MP quadrature uses <40 digits.
        with mp.workdps(max(mp.dps, iv.dps+20)):
            interval = point_interval(child, x)
            left, right = rounding(low(interval)), rounding(high(interval))
            if left != right:
                raise UncertainDiscontinuity(f"{token} input cannot be resolved at an integer boundary")
            rounded = left
            if token == "FRAC":
                value = child.evaluate(x)
    return value-rounded if token == "FRAC" else rounded


@dataclass(frozen=True)
class Node:
    token: str
    children: tuple
    interval: object

    def tokens(self):
        return [token for child in self.children for token in child.tokens()] + [self.token]

    def parsed(self):
        if not self.children:
            return {"PI": "pi", "E": "e", "HALF": "(1/2)",
                    "THIRD": "(1/3)", "-1": "(-1)"}.get(self.token, self.token)
        args = [child.parsed() for child in self.children]
        if self.token in BINARY:
            return f"({args[0]} {self.token} {args[1]})"
        if self.token == "NEG":
            return f"(-{args[0]})"
        if self.token == "SQUARE":
            return f"({args[0]})^2"
        return f"{self.token.lower()}({args[0]})"

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def depth(self):
        return 0 if not self.children else 1 + max(c.depth() for c in self.children)

    def evaluate(self, x, check_ranges=False):
        if not self.children:
            result = x if self.token == "x" else constant(self.token, mp)
        else:
            values = [c.evaluate(x, check_ranges) for c in self.children]
            result = (checked_rounding(self.token, self.children[0], x, values[0])
                      if self.token in ROUNDING_OPS else scalar_apply(self.token, values))
        if check_ranges:
            if not mp.isfinite(result) or mp.im(result) != 0:
                raise AssertionError(f"Non-real or nonfinite {self.token}: {result}")
            # Scalar checks are diagnostics, not the source of the interval bounds.
            tolerance = mp.mpf("1e-65") * max(1, abs(result))
            assert low(self.interval) - tolerance <= result <= high(self.interval) + tolerance, (
                self.token, result, self.interval)
        return result


def evaluate_rpn(tokens, x):
    stack = []
    for token in tokens:
        if token in ARITY:
            n = ARITY[token]
            if len(stack) < n:
                raise ValueError("Malformed RPN")
            args = stack[-n:]
            del stack[-n:]
            stack.append(scalar_apply(token, args))
        else:
            stack.append(x if token == "x" else constant(token, mp))
    if len(stack) != 1:
        raise ValueError("Malformed RPN")
    return stack[0]


class Generator:
    def __init__(self, seed, max_operators=32):
        self.rng = random.Random(seed)
        self.maximum = max_operators
        self.rejections = Counter()

    def tree(self, k, x_interval):
        if k == 0:
            token = "x" if self.rng.random() < 0.5 else self.rng.choice(CONSTANTS)
            return Node(token, (), x_interval if token == "x" else constant(token, iv))
        if self.rng.random() < 0.5:
            children = (self.tree(k-1, x_interval),)
            candidates = list(UNARY)
        else:
            left_k = self.rng.randrange(k)
            children = (self.tree(left_k, x_interval), self.tree(k-1-left_k, x_interval))
            candidates = list(BINARY)
        # A random permutation makes the first eligible operator uniform among
        # eligible operators of the selected arity, conditional on the children.
        self.rng.shuffle(candidates)
        for token in candidates:
            try:
                return Node(token, children, interval_apply(token, [c.interval for c in children]))
            except Ineligible as exc:
                self.rejections[f"{token}: {exc}"] += 1
        raise Ineligible("no admissible parent of selected arity")

    def sample(self, gamma):
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("Gamma must be finite and nonnegative")
        if gamma == 0:
            k = 0
        else:
            log_weights = [n*math.log(gamma)-math.lgamma(n+1) for n in range(self.maximum+1)]
            offset = max(log_weights)
            k = self.rng.choices(range(self.maximum+1), [math.exp(w-offset) for w in log_weights])[0]
        a_index, b_index = sorted(self.rng.sample(range(len(BOUND_TOKENS)), 2))
        a, b = BOUND_TOKENS[a_index], BOUND_TOKENS[b_index]
        return self.sample_fixed(k, a, b)

    def sample_fixed(self, k, a, b):
        """Generate a fresh AST without redrawing its operator count or bounds."""
        if not 0 <= k <= self.maximum or a not in BOUND_TOKENS or b not in BOUND_TOKENS:
            raise ValueError("Invalid fixed size or bounds")
        if constant(a, mp) >= constant(b, mp):
            raise ValueError("Lower bound must be smaller than upper bound")
        domain = iv.mpf([constant(a, iv).a, constant(b, iv).b])
        for attempt in range(1000):
            try:
                return a, b, k, self.tree(k, domain), attempt
            except Ineligible:
                self.rejections["whole-tree retry at fixed K and bounds"] += 1
        raise RuntimeError(f"Failed to fill K={k}, bounds={a},{b}; no easier fallback")


def record(a, b, k, tree, retries):
    tokens = tree.tokens()
    assert sum(t in ARITY for t in tokens) == k
    return {"bounds": [a, b], "operators": k, "depth": tree.depth(),
            "contains_x": "x" in tokens, "whole_tree_retries": retries,
            "parsed": f"Integral[{a}, {b}] {tree.parsed()} dx",
            "rpn": " ".join([a, b, *tokens, "INT_X", "="]),
            "range_display_approx": [mp.nstr(low(tree.interval), 14), mp.nstr(high(tree.interval), 14)]}


def audit_tree(a, b, k, tree, points=17):
    tokens = tree.tokens()
    assert sum(t in ARITY for t in tokens) == k
    lower, upper = constant(a, mp), constant(b, mp)
    for i in range(points):
        x = lower + (upper-lower)*i/(points-1)
        recursive = tree.evaluate(x, check_ranges=True)
        stack = evaluate_rpn(tokens, x)
        assert mp.almosteq(recursive, stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gammas", nargs="+", type=float, default=[1, 3, 6, 12, 20])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--audit-count", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.audit_count < 0:
        parser.error("count must be positive and audit-count nonnegative")
    result = {"version": VERSION, "seed": args.seed, "constants": CONSTANTS,
              "bounds_grid": BOUND_TOKENS, "max_operators": 32,
              "binary_probability": 0.5, "variable_leaf_probability": 0.5,
              "preview_node_magnitude_limit": str(PREVIEW_MAGNITUDE_LIMIT),
              "interval_digits": iv.dps, "scalar_audit_digits": mp.dps,
              "answer_labels_generated": False, "gammas": {}}
    for gamma in args.gammas:
        key = format(gamma, "g")
        seed = int.from_bytes(hashlib.sha256(f"{VERSION}:{args.seed}:{key}".encode()).digest()[:8], "big")
        generator = Generator(seed)
        samples = []
        operators = Counter()
        uncertain_audits = Counter()
        total_k = 0
        trees_with_x = 0
        for index in range(max(args.count, args.audit_count)):
            a, b, k, tree, retries = generator.sample(gamma)
            point_audit = "passed"
            try:
                audit_tree(a, b, k, tree)
            except UncertainDiscontinuity as exc:
                point_audit = str(exc)
                uncertain_audits[point_audit] += 1
            operators.update(n.token for n in tree.walk() if n.children)
            total_k += k
            trees_with_x += int("x" in tree.tokens())
            if index < args.count:
                samples.append({**record(a, b, k, tree, retries),"point_audit":point_audit})
        result["gammas"][key] = {"samples": samples, "audited_trees": max(args.count, args.audit_count),
                                 "mean_operators": total_k/max(args.count, args.audit_count),
                                 "trees_with_x": trees_with_x,
                                 "operator_counts": dict(operators),
                                 "uncertain_point_audits":dict(uncertain_audits),
                                 "ineligible_candidates": dict(generator.rejections)}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
