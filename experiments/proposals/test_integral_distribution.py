"""Independent numerical references and histogram-transform checks."""

import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

from experiments.proposals.integral_distribution import evaluate, signed_log, run_gamma, read_rows
from experiments.proposals.integral_ast_preview import UncertainDiscontinuity
from experiments.proposals.integral_distribution_refine import compile_array, composite, tree_from_rpn


class NumericalDistributionTests(unittest.TestCase):
    def test_integrals_of_discontinuous_operators(self):
        for token,answer in [("FLOOR",-2),("CEIL",2),("FRAC",2)]:
            with self.subTest(token=token):
                value,_,_=evaluate(tree_from_rpn(f"-2 2 x {token} INT_X ="),"-2","2")
                self.assertAlmostEqual(value,answer,places=9)

    def test_uncertain_jump_side_is_rejected(self):
        # sin(pi) is exactly zero; its rounded approximation must not choose -1.
        with self.assertRaises(UncertainDiscontinuity):
            evaluate(tree_from_rpn("0 1 PI SIN FLOOR INT_X ="),"0","1")

    def test_fractional_part_negative_constant(self):
        value,_,_=evaluate(tree_from_rpn("0 1 THIRD NEG FRAC INT_X ="),"0","1")
        self.assertAlmostEqual(value,2/3,places=12)

    def test_vector_ceiling_preserves_positive_underflow(self):
        # exp(-6561*x^2) is positive, even where float arithmetic gives zero.
        tree=tree_from_rpn("1 2 x SQUARE 3 SQUARE SQUARE SQUARE * NEG EXP CEIL INT_X =")
        function,_=compile_array(tree)
        np.testing.assert_array_equal(function(np.array([1.,1.5,2.])),[1,1,1])

    def test_previous_high_precision_answers(self):
        path=Path(__file__).with_name("integral_answer_samples.json")
        for row in json.loads(path.read_text())["samples"]:
            with self.subTest(gamma=row["gamma"]):
                tree=tree_from_rpn(row["rpn"])
                value,_,_=evaluate(tree,*row["bounds"])
                self.assertAlmostEqual(value,float(row["answer_decimal"]),places=9)

    def test_zero_odd_integral(self):
        value,_,_=evaluate(tree_from_rpn("-1 1 x SIN INT_X ="),"-1","1")
        self.assertLess(abs(value),1e-12)

    def test_negative_integral(self):
        value,_,_=evaluate(tree_from_rpn("0 1 x SQUARE NEG INT_X ="),"0","1")
        self.assertAlmostEqual(value,-1/3,places=12)

    def test_endpoint_substitution(self):
        function,_=compile_array(tree_from_rpn("0 1 x SQRT INT_X ="))
        self.assertAlmostEqual(composite(function,0,1,16,32),2/3,places=12)

    def test_constant_vector_broadcast(self):
        function,_=compile_array(tree_from_rpn("-1 1 PI INT_X ="))
        self.assertAlmostEqual(composite(function,-1,1,16,32),2*np.pi,places=12)

    def test_histogram_keeps_sign_zero_and_tails(self):
        values=np.array([-278222.7722057061,-1,0,1,287233.12261127424])
        transformed=signed_log(values)
        self.assertTrue(np.all(np.diff(transformed)>0))
        self.assertEqual(transformed[2],0)
        extent=max(abs(transformed))
        counts,_=np.histogram(transformed,bins=np.linspace(-extent,extent,102))
        self.assertEqual(sum(counts),len(values))
        self.assertEqual(counts[len(counts)//2],1)

    def test_replacements_preserve_slot_and_future_draws(self):
        config={"seed":19,"count":2,"item_timeout_seconds":2,"audit_every":200,
                "max_attempts_per_slot":10,"refinement_timeout_seconds":2}
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            for directory in (first,second):
                (Path(directory)/"records").mkdir();(Path(directory)/"rejections").mkdir()
            with patch("experiments.proposals.integral_distribution.evaluate",
                       side_effect=[UncertainDiscontinuity("test jump"),(0.,"test",{}),(0.,"test",{})]):
                run_gamma(4,config,first)
            accepted=read_rows(Path(first)/"records/gamma_04.jsonl")
            rejected=read_rows(Path(first)/"rejections/gamma_04.jsonl")
            self.assertEqual(len(accepted),2);self.assertEqual(len(rejected),1)
            self.assertEqual(accepted[0]["bounds"],rejected[0]["bounds"])
            self.assertEqual(accepted[0]["operators"],rejected[0]["operators"])
            self.assertEqual(accepted[0]["evaluation_attempt"],2)
            with patch("experiments.proposals.integral_distribution.evaluate",return_value=(0.,"test",{})):
                run_gamma(4,config,second)
            baseline=read_rows(Path(second)/"records/gamma_04.jsonl")
            self.assertEqual(accepted[1]["rpn"],baseline[1]["rpn"])
            with patch("experiments.proposals.integral_distribution.evaluate") as evaluate_mock:
                run_gamma(4,config,first)
                evaluate_mock.assert_not_called()
            self.assertEqual(accepted,read_rows(Path(first)/"records/gamma_04.jsonl"))
            self.assertEqual(rejected,read_rows(Path(first)/"rejections/gamma_04.jsonl"))


if __name__=="__main__":unittest.main()
