"""Local background survey of unquantized numerical integral values.

Each gamma fills fixed (K, bounds) slots with numerically accepted ASTs.
Rejected candidates are retained separately for diagnostics.
This is a distribution diagnostic, not a certified training-data labeler.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nanogpt-integral-matplotlib")

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import lambertw

from experiments.proposals.integral_ast_preview import (
    BOUND_TOKENS, ROUNDING_OPS, UncertainDiscontinuity, Generator,
    VERSION as GENERATOR_VERSION, checked_rounding, constant, integer_valued, mp, record,
)

VERSION = "integral-distribution-v2-accepted-slots"
NEAR_ZERO = 1e-8


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    # A report may read while the last JSONL record is being written.
    return [json.loads(line) for line in lines if line.endswith("\n")]


def real_w0(value):
    result = lambertw(value, k=0)
    if result.imag != 0 or not math.isfinite(result.real):
        raise ValueError("W0 evaluation is not finite and real")
    return float(result.real)


def compile_scalar(tree):
    """Closures avoid repeatedly parsing RPN. Fold constant subtrees for speed.

    The original AST and saved RPN remain unchanged. No numeric clamping is used.
    """
    if not tree.children:
        if tree.token == "x":
            return lambda x: x, False
        value = float(constant(tree.token, mp))
        return lambda x, value=value: value, True
    children = [compile_scalar(child) for child in tree.children]
    functions = {"NEG": lambda a: -a, "SQUARE": lambda a: a*a,
                 "SQRT": math.sqrt, "SIN": math.sin, "COS": math.cos,
                 "TAN": math.tan, "ASIN": math.asin, "ACOS": math.acos,
                 "ATAN": math.atan, "LN": math.log, "EXP": math.exp,
                 "W0": real_w0, "+": lambda a,b: a+b, "-": lambda a,b: a-b,
                 "*": lambda a,b: a*b, "/": lambda a,b: a/b,
                 "FLOOR": math.floor, "CEIL": math.ceil, "FRAC": lambda a: a-math.floor(a)}
    function = functions[tree.token]
    if tree.token in ROUNDING_OPS:
        child = children[0][0]
        exact_integer = integer_valued(tree.children[0])
        def compiled(x):
            value = child(x)
            if exact_integer or abs(value-round(value)) > 1e-12*max(1,abs(value)):
                return function(value)
            return float(checked_rounding(tree.token,tree.children[0],mp.mpf(x),mp.mpf(value),guard="1e-12"))
    elif len(children) == 1:
        child = children[0][0]
        compiled = lambda x: function(child(x))
    else:
        left, right = children[0][0], children[1][0]
        compiled = lambda x: function(left(x), right(x))
    is_constant = all(child[1] for child in children)
    if is_constant:
        value = compiled(0.0)
        if not math.isfinite(value):
            raise ValueError("Nonfinite constant subtree")
        return lambda x, value=value: value, True
    return compiled, False


@contextmanager
def deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError(f"Evaluation exceeded {seconds:g} seconds")
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def mp_integral(tree, a_token, b_token, digits, pieces):
    with mp.workdps(digits):
        a, b = constant(a_token, mp), constant(b_token, mp)
        points = [a+(b-a)*i/pieces for i in range(pieces+1)]
        value, error = mp.quad(tree.evaluate, points, method="tanh-sinh", error=True,
                               maxdegree=8)
        if mp.im(value) != 0 or not mp.isfinite(value):
            raise ValueError("Non-real or nonfinite high-precision integral")
        return {"digits": digits, "pieces": pieces, "value": mp.nstr(value, digits),
                "estimated_error": mp.nstr(error, 8)}


def high_precision_pair(tree, a, b):
    first = mp_integral(tree, a, b, 25, 4)
    second = mp_integral(tree, a, b, 40, 8)
    value = mp.mpf(second["value"])
    scale = max(mp.mpf(1), abs(value))
    difference = abs(value-mp.mpf(first["value"]))
    if (difference > mp.mpf("1e-9")*scale or
            max(mp.mpf(first["estimated_error"]), mp.mpf(second["estimated_error"])) > mp.mpf("1e-9")*scale):
        raise ValueError("High-precision quadrature did not agree/converge")
    return float(value), {"checks": [first, second], "difference": str(difference)}


def evaluate(tree, a_token, b_token, audit=False):
    a, b = float(constant(a_token, mp)), float(constant(b_token, mp))
    try:
        function, is_constant = compile_scalar(tree)
        # Independent arithmetic path checks; these are not a domain certificate.
        point_difference = 0.0
        with mp.workdps(50):
            for fraction in (mp.mpf("0.2113248654"), mp.mpf("0.7886751346")):
                x = constant(a_token, mp)+(constant(b_token, mp)-constant(a_token, mp))*fraction
                reference = tree.evaluate(x)
                if mp.im(reference) != 0 or not mp.isfinite(reference):
                    raise ValueError("High-precision point evaluation invalid")
                difference = abs(function(float(x))-float(reference))
                point_difference = max(point_difference, difference)
                if difference > 1e-9*max(1, abs(float(reference))):
                    raise ValueError("Float/MP point evaluations disagree")
        diagnostics = {"point_difference_max": point_difference}
        estimates, failed_attempts = [], []
        for pieces, absolute, relative, limit in [(1, 2e-10, 2e-9, 300), (7, 2e-11, 2e-10, 800),
                                                 (31, 4e-12, 5e-11, 2000), (63, 2e-12, 2e-11, 4000)]:
            points = [a+(b-a)*i/pieces for i in range(1, pieces)] or None
            result = quad(function, a, b, points=points, full_output=1,
                          epsabs=absolute, epsrel=relative, limit=limit)
            value, error, info = result[:3]
            if len(result) != 3 or not math.isfinite(value):
                failed_attempts.append({"pieces":pieces,"reason":str(result[3])[:180] if len(result)>3 else "nonfinite"})
                continue
            estimates.append({"value": value, "estimated_error": error,
                              "evaluations": info["neval"], "pieces": pieces})
            if len(estimates)>=2:
                value = estimates[-1]["value"]
                scale = max(1, abs(value))
                delta = abs(value-estimates[-2]["value"])
                if delta<=1e-8*scale and max(e["estimated_error"] for e in estimates[-2:])<=1e-8*scale:
                    break
        else:
            raise ValueError("QUADPACK checks disagree or have excessive estimated error")
        diagnostics.update({"checks": estimates, "difference": delta, "constant_subtree": is_constant,
                            "failed_attempts":failed_attempts})
    except UncertainDiscontinuity:
        raise
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        value, diagnostics = high_precision_pair(tree, a_token, b_token)
        diagnostics["float_failure"] = str(exc)
        return value, "mp_fallback", diagnostics
    if audit:
        reference, mp_diagnostics = high_precision_pair(tree, a_token, b_token)
        if abs(value-reference) > 1e-8*max(1, abs(reference)):
            raise ValueError("QUADPACK/high-precision integral audit disagrees")
        diagnostics["high_precision_audit"] = mp_diagnostics
        diagnostics["high_precision_difference"] = abs(value-reference)
    return value, "quad_checked", diagnostics


def run_gamma(gamma, config, directory):
    path = Path(directory) / "records" / f"gamma_{gamma:02d}.jsonl"
    rejected_path = Path(directory) / "rejections" / f"gamma_{gamma:02d}.jsonl"
    previous = read_rows(path)
    if previous:
        # Remove a partially written trailing record before appending on resume.
        path.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in previous))
    previous_rejections = read_rows(rejected_path)
    # Incomplete slots are regenerated from their own seed. Their partial reject
    # logs are replaced, avoiding duplicate attempts after a restart.
    previous_rejections = [r for r in previous_rejections if r["sample_index"] <= len(previous)]
    rejected_path.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in previous_rejections))
    rejection_count = len(previous_rejections)
    candidate_rejections = Counter()
    started = time.monotonic()
    with path.open("a", buffering=1) as output, rejected_path.open("a",buffering=1) as rejected_output:
        for index in range(len(previous),config["count"]):
            seed = int.from_bytes(hashlib.sha256(
                f"{GENERATOR_VERSION}:{config['seed']}:{gamma}:slot:{index}".encode()).digest()[:8], "big")
            generator = Generator(seed)
            a, b, k, tree, retries = generator.sample(gamma)
            for attempt in range(1,config["max_attempts_per_slot"]+1):
                item = record(a, b, k, tree, retries)
                item.update({"gamma":gamma,"sample_index":index+1,"evaluation_attempt":attempt})
                begin = time.monotonic()
                try:
                    with deadline(config["item_timeout_seconds"]):
                        value, method, diagnostics = evaluate(tree,a,b,audit=index % config["audit_every"] == 0)
                    item.update({"status":"ok","value":value,"method":method,
                                 "numerics":diagnostics,"outside_answer_range":abs(value)>32640})
                except (TimeoutError,ValueError,ZeroDivisionError,OverflowError) as exc:
                    item.update({"status":"unresolved","value":None,
                                 "reason":f"{type(exc).__name__}: {exc}"})
                item["seconds"] = time.monotonic()-begin
                if item["status"] != "ok" and not item["reason"].startswith("UncertainDiscontinuity:"):
                    from experiments.proposals.integral_distribution_refine import refine
                    item = refine(item,timeout_seconds=config["refinement_timeout_seconds"])
                if item["status"] == "ok":
                    output.write(json.dumps(item,allow_nan=False)+"\n")
                    break
                item["status"] = "rejected"
                rejected_output.write(json.dumps(item,allow_nan=False)+"\n")
                rejection_count += 1
                a, b, k, tree, retries = generator.sample_fixed(k,a,b)
            else:
                raise RuntimeError(f"Failed to fill gamma={gamma}, slot={index+1}, K={k}, bounds={a},{b}")
            candidate_rejections.update(generator.rejections)
            if (index+1) % 100 == 0 or index+1 == config["count"]:
                print(f"gamma={gamma:02d} accepted={index+1}/{config['count']} rejected={rejection_count} elapsed={time.monotonic()-started:.1f}s",flush=True)
    diagnostics = {"gamma": gamma,"candidate_rejections_this_process":dict(candidate_rejections),
                   "rejected_evaluations":rejection_count,
                   "seconds_this_process": time.monotonic()-started}
    atomic_json(path.with_suffix(".stats.json"), diagnostics)
    return gamma


def collect(directory, config):
    groups = {}
    for gamma in config["gammas"]:
        rows = read_rows(directory / "records" / f"gamma_{gamma:02d}.jsonl")
        rejected = read_rows(directory / "rejections" / f"gamma_{gamma:02d}.jsonl")
        values = np.array([row["value"] for row in rows if row["status"]=="ok"], dtype=float)
        stats = {"gamma": gamma, "requested": config["count"], "evaluated": len(rows),
                 "accepted":len(values),"unresolved":sum(r["status"]!="ok" for r in rows),
                 "rejected_evaluations":len(rejected),"candidate_evaluations":len(rows)+len(rejected)}
        if len(values):
            stats.update({"minimum": float(values.min()), "maximum": float(values.max()),
                          "mean": float(values.mean()), "median": float(np.median(values)),
                          "p01": float(np.quantile(values,.01)), "p99": float(np.quantile(values,.99)),
                          "near_zero": int(np.sum(abs(values)<=NEAR_ZERO)),
                          "negative": int(np.sum(values < -NEAR_ZERO)),
                          "positive": int(np.sum(values > NEAR_ZERO)),
                          "outside_answer_range": int(np.sum(abs(values)>32640)),
                          "high_precision_checked": sum(r.get("method")=="mp_fallback" or
                              "high_precision_audit" in r.get("numerics",{}) for r in rows)})
        groups[str(gamma)] = {"stats": stats,
            "samples": [{"index": r["sample_index"], "k":r["operators"], "value":r["value"],
                         "rpn":r["rpn"], "status":r["status"]} for r in rows]}
    return groups


def write_report(directory, config, started, state="running", errors=None, static=False):
    groups = collect(directory, config)
    summary = {"state": state, "generated_at": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic()-started, "config": config,
               "evaluated": sum(g["stats"]["evaluated"] for g in groups.values()),
               "accepted": sum(g["stats"]["accepted"] for g in groups.values()),
               "unresolved": sum(g["stats"]["unresolved"] for g in groups.values()),
               "rejected_evaluations":sum(g["stats"]["rejected_evaluations"] for g in groups.values()),
               "candidate_evaluations":sum(g["stats"]["candidate_evaluations"] for g in groups.values()),
               "worker_errors": errors or [], "groups": {k:v["stats"] for k,v in groups.items()}}
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "progress.json", {k:v for k,v in summary.items() if k!="groups"})
    template = Path(__file__).with_name("integral_distribution_template.html").read_text()
    payload = {"summary": summary, "groups": groups}
    html = template.replace("__SURVEY_DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/"))
    html = html.replace("__AUTO_REFRESH__", '<meta http-equiv="refresh" content="30">' if state=="running" else "")
    target = directory / "histograms.html"
    temporary = target.with_suffix(".html.tmp")
    temporary.write_text(html)
    temporary.replace(target)
    if static:
        static_plots(directory, groups)
        with (directory / "summary.csv").open("w", newline="") as output:
            fields = sorted(set().union(*(g["stats"] for g in groups.values())))
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(g["stats"] for g in groups.values())
        with (directory / "values.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["gamma","sample_index","operators","status","value","rpn"])
            for gamma,group in groups.items():
                for sample in group["samples"]:
                    writer.writerow([gamma,sample["index"],sample["k"],sample["status"],sample["value"],sample["rpn"]])
    return summary


def signed_log(values):
    return np.sign(values)*np.log10(1+np.abs(values))


def static_plots(directory, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    values = {int(k): np.array([s["value"] for s in g["samples"] if s["status"]=="ok"])
              for k,g in groups.items()}
    pooled = np.concatenate(list(values.values()))
    if not len(pooled):
        return
    extent = max(1, float(np.max(np.abs(signed_log(pooled)))))
    edges = np.linspace(-extent, extent, 102)
    possible_ticks = np.array([-1e6,-1e4,-100,-10,-1,0,1,10,100,1e4,1e6])
    tick_locations = signed_log(possible_ticks)
    use = abs(tick_locations) <= extent
    tick_values, tick_locations = possible_ticks[use], tick_locations[use]
    # Thin labels if the observed range is wide; keep zero and symmetric major ticks.
    if len(tick_values)>9:
        use = np.isin(tick_values, [-1e6,-1e4,-100,0,100,1e4,1e6])
        tick_values, tick_locations = tick_values[use], tick_locations[use]
    labels = [f"{v:g}" for v in tick_values]
    matrices = [np.histogram(signed_log(v), bins=edges)[0] for v in values.values()]
    ymax = max(int(hist.max()) for hist in matrices)
    nrows = math.ceil(len(values)/4)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(nrows,4,figsize=(18,3*nrows+1),sharex=True,sharey=True,squeeze=False)
    for ax, (gamma, vals), counts in zip(axes.flat,values.items(),matrices):
        ax.bar((edges[:-1]+edges[1:])/2, counts, width=np.diff(edges)*.94,color="#257c96",linewidth=0)
        ax.set_title(f"Gamma {gamma}  |  n={len(vals):,}",loc="left",weight="bold")
        ax.set_xlim(-extent,extent); ax.set_ylim(0,ymax*1.08)
        ax.set_xticks(tick_locations,labels,rotation=35,ha="right")
        ax.grid(axis="y",alpha=.16)
        near = 100*np.mean(abs(vals)<=NEAR_ZERO) if len(vals) else 0
        ax.text(.03,.88,f"Near zero: {near:.1f}%",transform=ax.transAxes,color="#536471",fontsize=9)
    for ax in list(axes.flat)[len(values):]: ax.set_visible(False)
    fig.suptitle("Numerical integral values across AST complexity",fontsize=22,x=.06,y=.99,ha="left",weight="bold")
    fig.supxlabel("Evaluated integral value · signed logarithmic scale (linear near zero)",fontsize=12)
    fig.supylabel("Expressions per bin",fontsize=12)
    fig.text(.06,.99-.40/(3*nrows+1),"Shared bins and axes • unquantized results • near zero means |value| ≤ 10⁻⁸",fontsize=11,va="top",color="#536471")
    fig.tight_layout(rect=(.02,.025,1,1-.9/(3*nrows+1)))
    fig.savefig(directory/"histograms.png",dpi=160)
    fig.savefig(directory/"histograms.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14,max(6,len(values)*.35)))
    matrix = np.array(matrices,dtype=float)
    percentages = 100*matrix/np.maximum(1,matrix.sum(axis=1,keepdims=True))
    image = ax.imshow(np.ma.masked_where(percentages==0,percentages),origin="lower",aspect="auto",
                      extent=(-extent,extent,.5,len(values)+.5),cmap="viridis",
                      norm=LogNorm(vmin=max(.01,100/max(1,max(map(len,values.values())))),vmax=100))
    ax.set_yticks(range(1,len(values)+1),list(values))
    ax.set_xticks(tick_locations,labels)
    ax.set_xlabel("Evaluated integral value · signed logarithmic scale")
    ax.set_ylabel("Gamma")
    ax.set_title("How the distribution of integral values changes with gamma",loc="left",weight="bold",fontsize=16,pad=15)
    fig.colorbar(image,ax=ax,label="Percent of evaluated values per bin (log color scale)")
    fig.tight_layout()
    fig.savefig(directory/"heatmap.png",dpi=180)
    fig.savefig(directory/"heatmap.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("out/integral-value-survey-bounds2-rounding"))
    parser.add_argument("--count",type=int,default=1000)
    parser.add_argument("--gammas",type=int,nargs="+",default=list(range(1,21)))
    parser.add_argument("--seed",type=int,default=20261024)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--item-timeout",type=float,default=20)
    parser.add_argument("--audit-every",type=int,default=200)
    parser.add_argument("--refinement-timeout",type=float,default=30)
    parser.add_argument("--max-attempts-per-slot",type=int,default=100)
    parser.add_argument("--report-only",action="store_true")
    args = parser.parse_args()
    if (args.count<1 or args.workers<1 or args.audit_every<1 or args.item_timeout<=0
            or args.refinement_timeout<=0 or args.max_attempts_per_slot<1):
        parser.error("counts, workers, audit interval, and timeout must be positive")
    directory = args.output.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"records").mkdir(exist_ok=True)
    (directory/"rejections").mkdir(exist_ok=True)
    config = {"version":VERSION,"generator_version":GENERATOR_VERSION,"count":args.count,
              "gammas":args.gammas,"seed":args.seed,"workers":args.workers,
              "item_timeout_seconds":args.item_timeout,"audit_every":args.audit_every,
              "scipy_version":scipy.__version__,"numpy_version":np.__version__,
              "near_zero_threshold":NEAR_ZERO,"preview_node_magnitude_limit":1e6,
              "bounds_grid":list(BOUND_TOKENS),"resample_failed_evaluations":True,
              "sampling_scheme":"independent-per-slot-v1",
              "max_attempts_per_slot":args.max_attempts_per_slot,
              "refinement_timeout_seconds":args.refinement_timeout,
              "output_range_filter":False,"quantized":False}
    config_path = directory/"config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if args.report_only:
            config = saved
        elif saved != config:
            raise ValueError("Output directory contains a different configuration")
    else:
        atomic_json(config_path,config)
    started = time.monotonic()
    if args.report_only:
        previous_path = directory / "summary.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        started -= previous.get("elapsed_seconds",0)
        write_report(directory,config,started,state=previous.get("state","report"),
                     errors=previous.get("worker_errors",[]),static=True)
        return
    atomic_json(directory/"process.json",{"pid":os.getpid(),"started":datetime.now(timezone.utc).isoformat()})
    write_report(directory,config,started)
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_gamma,gamma,config,str(directory)):gamma for gamma in args.gammas}
        while pending:
            done,_ = wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                gamma = pending.pop(future)
                try: future.result()
                except Exception as exc:
                    errors.append({"gamma":gamma,"error":repr(exc)})
                    print(f"WORKER ERROR gamma={gamma}: {exc!r}",flush=True)
            summary = write_report(directory,config,started,errors=errors)
            print(f"TOTAL evaluated={summary['evaluated']} accepted={summary['accepted']} unresolved={summary['unresolved']}",flush=True)
    complete = summary["accepted"] == args.count*len(args.gammas) and not errors
    summary=write_report(directory,config,started,state="complete" if complete else "incomplete",errors=errors,static=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("groups","config")}),flush=True)


if __name__=="__main__":
    main()
