"""Numerically evaluate the five displayed AST examples; not a dataset labeler.

Run from the repository root:
  .venv/bin/python -m experiments.proposals.integral_answer_preview
"""

import json
from pathlib import Path

from experiments.proposals.integral_ast_preview import constant, evaluate_rpn, mp

DIRECTORY = Path(__file__).resolve().parent
SELECTION = {"1": 4, "3": 7, "6": 1, "12": 5, "20": 6}


def normal_bit_preview(value):
    """Nearest-value encoding for these normal, nonzero examples only.

    This is not the proposed dataset quantizer: subnormal/zero handling and
    reliable decisions for arbitrary integral values remain future work.
    """
    if not mp.isfinite(value) or value == 0:
        raise ValueError("This preview only handles finite, nonzero normal values")
    fraction, exponent = mp.frexp(abs(value))
    mantissa = int(mp.nint(fraction * 256))
    if mantissa == 256:
        mantissa = 128
        exponent += 1
    if not (-16 <= exponent <= 15 and 128 <= mantissa <= 255):
        raise ValueError("Selected value is outside the preview's normal range")
    sign = int(value < 0)
    raw_exponent = exponent + 16
    decoded = (-1)**sign * mp.ldexp(mp.mpf(mantissa)/256, exponent)
    return {"sign": sign, "exponent_bits": f"{raw_exponent:05b}",
            "mantissa_bits": f"{mantissa:08b}",
            "compact_display": f"{sign} | {raw_exponent:05b} | {mantissa:08b}",
            "decoded_decimal": mp.nstr(decoded, 30),
            "absolute_rounding_error": mp.nstr(abs(decoded-value), 15)}


def independent_checks():
    """Direct formulas, separate from the AST/RPN evaluation path.

    Gamma 1 has an analytic antiderivative. Gamma 12 substitutes x=-1+t^2
    and rewrites acos(-1+t^2)=pi-2 asin(t/sqrt(2)) to smooth the endpoint.
    Other checks use Gauss-Legendre instead of tanh-sinh. Identities W0(e)=1
    and cancellation of +x and -x simplify the checks for gamma 6 and 20.
    """
    with mp.workdps(50):
        def gamma12(t):
            x = -1+t*t
            acos_x = mp.pi-2*mp.asin(t/mp.sqrt(2))
            return -2*t*(mp.tan(mp.sin(x*x-1))
                         + mp.sin(mp.sin(mp.e**2+x/2-acos_x)))

        w3 = mp.lambertw(3)
        formulas = {
            "3": (lambda x: mp.log(x*mp.atan(x)-x), [-1, -mp.mpf(1)/2]),
            "6": (lambda x: mp.tan(1/(mp.e-x))/mp.cos(x), [mp.mpf(1)/2, 1]),
            "12": (gamma12, [0, mp.sqrt(mp.mpf(1)/2)]),
            "20": (lambda x: mp.atan(3/(3-x)+2)
                    * (mp.atan(mp.e)-(9/mp.pi)**2+mp.exp(x/w3+mp.cos(3))),
                    [0, mp.mpf(1)/2]),
        }
        values = {"1": {"method": "analytic: 5*pi/12 - sqrt(3)/2",
                         "value": mp.nstr(5*mp.pi/12-mp.sqrt(3)/2, 50)}}
        for gamma, (function, bounds) in formulas.items():
            value, error = mp.quad(function, bounds, method="gauss-legendre", error=True)
            values[gamma] = {"method": "independent formula, 50-digit Gauss-Legendre",
                             "value": mp.nstr(value, 50), "estimated_error": mp.nstr(error, 6)}
        return values


def main():
    source = json.loads((DIRECTORY / "integral_ast_samples.json").read_text())
    checks = independent_checks()
    rows = []
    for gamma, sample_index in SELECTION.items():
        sample = source["gammas"][gamma]["samples"][sample_index-1]
        input_tokens = sample["rpn"].split()
        assert input_tokens[:2] == sample["bounds"]
        assert input_tokens[-2:] == ["INT_X", "="]
        expression_tokens = input_tokens[2:-2]
        estimates = []
        for digits, pieces in [(40, 4), (70, 8)]:
            with mp.workdps(digits):
                a, b = (constant(token, mp) for token in sample["bounds"])
                points = [a+(b-a)*i/pieces for i in range(pieces+1)]
                value, error = mp.quad(lambda x: evaluate_rpn(expression_tokens, x), points,
                                       method="tanh-sinh", error=True)
                estimates.append({"digits": digits, "subintervals": pieces,
                                  "value": mp.nstr(value, digits),
                                  "estimated_error": mp.nstr(error, 6)})
        low_precision, answer = (mp.mpf(item["value"]) for item in estimates)
        precision_delta = abs(answer-low_precision)
        independent_delta = abs(answer-mp.mpf(checks[gamma]["value"]))
        tolerance = mp.mpf("1e-30")*max(1, abs(answer))
        assert precision_delta < tolerance
        assert independent_delta < tolerance
        bits = normal_bit_preview(answer)
        for alternative in (low_precision, mp.mpf(checks[gamma]["value"])):
            assert normal_bit_preview(alternative)["compact_display"] == bits["compact_display"]
        display = mp.nstr(answer, 12)
        rows.append({"gamma": gamma, "sample_index": sample_index, **sample,
                     "quadrature_checks": estimates, "independent_check": checks[gamma],
                     "precision_difference": mp.nstr(precision_delta, 6),
                     "independent_difference": mp.nstr(independent_delta, 6),
                     "answer_decimal": estimates[-1]["value"],
                     "answer_display_approx": display,
                     "rpn_with_decimal_approx": sample["rpn"]+" "+display,
                     "normal_encoding_preview": bits})
        print(f"gamma={gamma}: {display}; bits={bits['compact_display']}; "
              f"decoded={bits['decoded_decimal']}", flush=True)
    result = {"source": "integral_ast_samples.json", "selected_examples_only": True,
              "agreement_tolerance": "1e-30 * max(1, abs(answer))",
              "note": "Numerical agreement checks, not formally certified integration bounds. "
                      "Decimal displays are approximate. This is not a general dataset labeler.",
              "samples": rows}
    destination = DIRECTORY / "integral_answer_samples.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
