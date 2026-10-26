"""Accepted procedural labels, atomic vocabulary, and whole-record packing."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from experiments.proposals.integral_ast_preview import (
    ARITY, BOUND_TOKENS, CONSTANTS, Generator,
    VERSION as AST_VERSION, mp, record,
)
from experiments.proposals.integral_distribution import deadline, evaluate, high_precision_pair
from .encoding import (EXPONENT_TOKENS, MANTISSA_TOKENS,
                       SIGN_TOKENS, decode, stable_encoding)

VERSION = "integral-full-sequence-v2-independent-hard:"+AST_VERSION
VOCAB = tuple(dict.fromkeys((*CONSTANTS, *BOUND_TOKENS, "x", *ARITY, "INT_X", "=", "\n",
                            "<EXP>", "<MANT>", *SIGN_TOKENS, *EXPONENT_TOKENS,
                            *MANTISSA_TOKENS, "<PAD>")))
STOI = {token:index for index,token in enumerate(VOCAB)}
IGNORE_INDEX = -1
FIELD_IDS = {"sign": tuple(STOI[t] for t in SIGN_TOKENS),
             "exponent": tuple(STOI[t] for t in EXPONENT_TOKENS),
             "mantissa": tuple(STOI[t] for t in MANTISSA_TOKENS)}


def stream_seed(seed, *identity):
    return int.from_bytes(hashlib.sha256(":".join(map(str,(VERSION,seed,*identity))).encode()).digest()[:8], "big")


def example_plan(gamma, seed, count, gamma_sampling="fixed"):
    """Draw each sample's gamma independently of AST RNG and rejection retries."""
    if gamma_sampling not in ("fixed", "uniform_integer"):
        raise ValueError("Unsupported gamma sampling")
    if not math.isfinite(gamma) or gamma < (1 if gamma_sampling == "uniform_integer" else 0):
        raise ValueError("Invalid gamma or gamma ceiling")
    for index in range(count):
        sample_gamma = (random.Random(stream_seed(seed, "sample-gamma", index)).randint(1, math.floor(gamma))
                        if gamma_sampling == "uniform_integer" else gamma)
        yield sample_gamma, stream_seed(seed, "item", index)


@dataclass(frozen=True)
class Example:
    prefix: tuple
    answer: tuple
    value: str
    uncertainty: str
    gamma: float
    operators: int
    attempts: int = 1

    def __post_init__(self):
        if len(self.prefix) < 5 or self.prefix[-2:] != ("INT_X", "="):
            raise ValueError("Invalid integral prefix")
        if self.prefix[0] not in BOUND_TOKENS or self.prefix[1] not in BOUND_TOKENS:
            raise ValueError("Invalid bounds")
        if BOUND_TOKENS.index(self.prefix[0]) >= BOUND_TOKENS.index(self.prefix[1]):
            raise ValueError("Bounds must be increasing")
        depth = 0
        for token in self.prefix[2:-2]:
            if token in ARITY:
                if depth < ARITY[token]:
                    raise ValueError("Invalid RPN stack")
                depth += 1-ARITY[token]
            elif token in CONSTANTS or token == "x":
                depth += 1
            else:
                raise ValueError("Invalid integrand token")
        if depth != 1:
            raise ValueError("Integrand must leave one RPN result")
        if sum(t in ARITY for t in self.prefix) != self.operators:
            raise ValueError("Wrong AST operator count")
        if decode(self.answer) != stable_encoding(self.value,self.uncertainty):
            raise ValueError("Cached target is not stable under its numerical uncertainty")

    @property
    def tokens(self):
        return self.prefix+self.answer+("\n",)

    @property
    def text(self):
        return " ".join(self.tokens[:-1])+"\n"

    def to_dict(self):
        return {"prefix":list(self.prefix),"answer":list(self.answer),"value":self.value,
                "uncertainty":self.uncertainty,"gamma":self.gamma,"operators":self.operators,
                "attempts":self.attempts}

    @classmethod
    def from_dict(cls, row):
        return cls(**{**row,"prefix":tuple(row["prefix"]),"answer":tuple(row["answer"])})


def label_from_checks(value, diagnostics):
    """Conservative engineering margin around the numerical checks, not a proof."""
    checks = diagnostics["checks"][-2:]
    if len(checks) != 2:
        raise ValueError("Two numerical checks are required")
    with mp.workdps(80):
        values = [mp.mpf(str(c["value"])) for c in checks]
        errors = [mp.mpf(str(c["estimated_error"])) for c in checks]
        if any(not mp.isfinite(v) for v in [*values, *errors]) or any(e < 0 for e in errors):
            raise ValueError("Nonfinite or invalid numerical checks")
        high_precision = all("digits" in c for c in checks)
        center = values[-1] if high_precision else mp.mpf(repr(value))
        floor = mp.mpf("1e-30") if high_precision else max(mp.mpf("1e-12"),mp.mpf(math.ulp(value))*32)
        radius = 4*max(*errors,*(abs(center-v) for v in values),floor)
        value_string, radius_string = mp.nstr(center,70),mp.nstr(radius,70)
    return value_string,radius_string,stable_encoding(value_string,radius_string)


class Dataset:
    def __init__(self, cache_dir, max_operators=32, timeout_seconds=5, max_attempts=100):
        self.cache_dir=Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.max_operators,self.timeout_seconds,self.max_attempts=max_operators,timeout_seconds,max_attempts
        self.stats=Counter()

    def sample(self,gamma,seed):
        identity={"version":VERSION,"gamma":float(gamma),"seed":seed,"max_operators":self.max_operators,
                  "timeout_seconds":self.timeout_seconds,"max_attempts":self.max_attempts}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=self.cache_dir/f"{key}.json" if self.cache_dir is not None else None
        if path is not None and path.exists():
            saved=json.loads(path.read_text())
            if saved["identity"]!=identity:
                raise ValueError("Dataset cache identity mismatch")
            self.stats["cache_hits"]+=1
            self.last_provenance=saved
            return Example.from_dict(saved["example"])
        generator=Generator(seed,self.max_operators)
        a,b,k,tree,retries=generator.sample(gamma)
        rejected=[]
        for attempt in range(1,self.max_attempts+1):
            try:
                with deadline(self.timeout_seconds):
                    value,method,diagnostics=evaluate(tree,a,b)
                    try:
                        value_string,radius,encoded=label_from_checks(value,diagnostics)
                    except ValueError:
                        # Re-evaluate the same AST before discarding a rounding-boundary case.
                        value,diagnostics=high_precision_pair(tree,a,b)
                        method="mp_stable_target"
                        value_string,radius,encoded=label_from_checks(value,diagnostics)
                example=Example(tuple([a,b,*tree.tokens(),"INT_X","="]),encoded.tokens,
                                value_string,radius,gamma,k,attempt)
                saved={"identity":identity,"example":example.to_dict(),"method":method,
                       "numerics":diagnostics,"rejected_candidates":rejected}
                self.last_provenance=saved
                if path is not None:
                    temporary=path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(saved,indent=2,allow_nan=False)+"\n")
                    temporary.replace(path)
                self.stats["generated"]+=1
                return example
            except (TimeoutError,ValueError,ZeroDivisionError,OverflowError) as exc:
                self.stats["rejected"]+=1
                reason=f"{type(exc).__name__}: {exc}"
                self.stats[type(exc).__name__]+=1
                rejected.append({"rpn":record(a,b,k,tree,retries)["rpn"],"reason":reason})
                if attempt < self.max_attempts:
                    a,b,k,tree,retries=generator.sample_fixed(k,a,b)
        raise RuntimeError(f"Cannot label gamma={gamma}, K={k}, bounds={a},{b}; no easier fallback")

    def examples(self,gamma,seed,count):
        return [self.sample(g,s) for g,s in example_plan(gamma,seed,count)]

    def training_examples(self, spec, count):
        if spec.gamma_sampling == "fixed":
            return self.examples(spec.gamma, spec.seed, count)
        return [self.sample(g,s) for g,s in example_plan(spec.gamma, spec.seed, count, spec.gamma_sampling)]


def batch(examples,block_size,device="cpu",pack=True):
    """A leading newline supplies context for the first bound; pad targets are -1.

    Every real record token is predicted exactly once, including its final newline.
    Packed rows always begin at a record boundary and contain complete records.
    """
    if not examples:
        raise ValueError("Empty batch")
    rows=[];current=[STOI["\n"]]
    for example in examples:
        tokens=[STOI[t] for t in example.tokens]
        if len(tokens)>block_size:
            raise ValueError("Complete expression does not fit the context")
        if len(current)>1 and (not pack or len(current)+len(tokens)-1>block_size):
            rows.append(current);current=[STOI["\n"]]
        current.extend(tokens)
    rows.append(current)
    x=torch.full((len(rows),block_size),STOI["<PAD>"],dtype=torch.long)
    y=torch.full((len(rows),block_size),IGNORE_INDEX,dtype=torch.long)
    for i,row in enumerate(rows):
        x[i,:len(row)-1]=torch.tensor(row[:-1])
        y[i,:len(row)-1]=torch.tensor(row[1:])
    return x.to(device),y.to(device)
