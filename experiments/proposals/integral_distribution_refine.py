"""Refine unresolved survey integrals with vectorized composite Gauss-Legendre.

Independent node orders and mesh refinement are numerical consistency checks,
not rigorous error bounds. Original failed evaluations are retained in records.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.special import roots_legendre, lambertw

from experiments.proposals.integral_ast_preview import ARITY, ROUNDING_OPS, Node, constant, integer_valued, interval_apply, iv, mp
from experiments.proposals.integral_distribution import compile_scalar, deadline, read_rows, write_report


def tree_from_rpn(rpn):
    tokens = rpn.split()
    assert tokens[-2:] == ["INT_X", "="]
    domain=iv.mpf([constant(tokens[0],iv).a,constant(tokens[1],iv).b])
    stack=[]
    for token in tokens[2:-2]:
        n=ARITY.get(token,0)
        children=tuple(stack[-n:]) if n else ()
        if n: del stack[-n:]
        interval=(interval_apply(token,[c.interval for c in children]) if children
                  else domain if token=="x" else constant(token,iv))
        stack.append(Node(token,children,interval))
    assert len(stack)==1
    return stack[0]


def array_w0(x):
    y=lambertw(x,0)
    if np.any(y.imag!=0) or not np.all(np.isfinite(y.real)):
        raise ValueError("Invalid vector Lambert W result")
    return y.real


def compile_array(node):
    if not node.children:
        if node.token=="x":return lambda x:x,False
        value=float(constant(node.token,mp))
        return lambda x:value,True
    children=[compile_array(child) for child in node.children]
    functions={"+":np.add,"-":np.subtract,"*":np.multiply,"/":np.divide,
               "NEG":np.negative,"SQUARE":np.square,"SQRT":np.sqrt,
               "SIN":np.sin,"COS":np.cos,"TAN":np.tan,"ASIN":np.arcsin,
               "ACOS":np.arccos,"ATAN":np.arctan,"LN":np.log,"EXP":np.exp,"W0":array_w0,
               "FLOOR":np.floor,"CEIL":np.ceil,"FRAC":lambda x:x-np.floor(x)}
    fn=functions[node.token]
    if node.token in ROUNDING_OPS and not integer_valued(node.children[0]):
        child=children[0][0]
        precise_scalar,_=compile_scalar(node)
        def compiled(x):
            values=child(x)
            near=np.abs(values-np.rint(values)) <= 1e-12*np.maximum(1,np.abs(values))
            result=fn(values)
            if not np.any(near):
                return result
            # Underflow and rounded transcendental values can change the side of
            # a jump. Use the same interval-backed decision as the scalar path.
            if np.ndim(x)==0:
                return precise_scalar(float(x))
            result=np.broadcast_to(result,np.shape(x)).copy()
            mask=np.broadcast_to(near,np.shape(x))
            for index in zip(*np.nonzero(mask)):
                result[index]=precise_scalar(float(x[index]))
            return result
    elif len(children)==1:
        child=children[0][0]
        compiled=lambda x:fn(child(x))
    else:
        left,right=children[0][0],children[1][0]
        compiled=lambda x:fn(left(x),right(x))
    is_constant=all(c[1] for c in children)
    if is_constant:
        value=float(node.evaluate(mp.mpf(0))) if node.token in ROUNDING_OPS else compiled(0.)
        return lambda x:value,True
    return compiled,False


def composite(function,a,b,pieces,order):
    nodes,weights=roots_legendre(order)
    total=np.longdouble(0)
    # The cosine substitution smooths common endpoint square-root behavior.
    # Chunking caps memory regardless of mesh size.
    for begin in range(0,pieces,1024):
        segment=np.arange(begin,min(pieces,begin+1024))[:,None]
        t=(segment+.5+nodes[None,:]/2)/pieces
        x=a+(b-a)*np.sin(np.pi*t/2)**2
        jac=(b-a)*np.pi/2*np.sin(np.pi*t)
        with np.errstate(invalid="raise",divide="raise",over="raise",under="ignore"):
            values=function(x)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite vector integrand")
        total+=np.sum(values*jac*weights[None,:],dtype=np.longdouble)/(2*pieces)
    return float(total)


def refine(row,timeout_seconds=90):
    started=time.monotonic()
    try:
        with deadline(timeout_seconds):
            tree=tree_from_rpn(row["rpn"])
            function,_=compile_array(tree)
            a,b=(float(constant(t,mp)) for t in row["bounds"])
            checks=[]
            for pieces in (16,64,256,1024,4096,16384,65536,262144):
                lo=composite(function,a,b,pieces,32)
                hi=composite(function,a,b,pieces,48)
                checks.append({"pieces":pieces,"order32":lo,"order48":hi})
                tolerance=1e-8*max(1,abs(hi))
                if len(checks)>=2 and max(abs(hi-lo),abs(hi-checks[-2]["order48"]))<=tolerance:
                    # A third, distinct order on a doubled mesh must also agree.
                    final=composite(function,a,b,pieces*2,40)
                    if abs(final-hi)<=tolerance:
                        return {**row,"initial_failure":{"reason":row["reason"],"seconds":row["seconds"]},
                                "status":"ok","value":final,"method":"composite_checked",
                                "outside_answer_range":abs(final)>32640,
                                "numerics":{"composite_checks":checks,"final_order":40,
                                            "final_pieces":pieces*2,"final_value":final,
                                            "absolute_tolerance":tolerance},
                                "refinement_seconds":time.monotonic()-started}
            raise ValueError("Composite refinements did not agree")
    except (ValueError,TimeoutError,FloatingPointError,OverflowError,ZeroDivisionError) as exc:
        return {**row,"refinement_failure":str(exc),"refinement_seconds":time.monotonic()-started}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("out/integral-value-survey"))
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    directory=args.output.resolve()
    config=json.loads((directory/"config.json").read_text())
    process=json.loads((directory/"process.json").read_text())
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(process["started"])).total_seconds()
    started=time.monotonic()-elapsed
    by_gamma={g:read_rows(directory/"records"/f"gamma_{g:02d}.jsonl") for g in config["gammas"]}
    unresolved=[r for rows in by_gamma.values() for r in rows if r["status"]!="ok"]
    print(f"Refining {len(unresolved)} unchanged ASTs",flush=True)
    write_report(directory,config,started,state="running")
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(refine,row) for row in unresolved]
        for future in as_completed(futures):
            row=future.result()
            gamma=row["gamma"]
            by_gamma[gamma][row["sample_index"]-1]=row
            destination=directory/"records"/f"gamma_{gamma:02d}.jsonl"
            temporary=destination.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in by_gamma[gamma]))
            temporary.replace(destination)
            print(f"gamma={gamma} sample={row['sample_index']} status={row['status']} elapsed={row['refinement_seconds']:.2f}s",flush=True)
            write_report(directory,config,started,state="running")
    result=write_report(directory,config,started,state="complete",static=True)
    print({k:result[k] for k in ('evaluated','accepted','unresolved')},flush=True)


if __name__=="__main__":main()
