"""Domain, extrema, rounding, and serialization checks for the local preview."""

import unittest

from experiments.proposals.integral_ast_preview import (
    ARITY, BOUND_TOKENS, Generator, Ineligible, audit_tree, constant, high, interval_apply,
    iv, low, mp, record,
)


class RangeTests(unittest.TestCase):
    def assert_encloses(self, interval, *values):
        for value in values:
            self.assertLessEqual(low(interval), value)
            self.assertGreaterEqual(high(interval), value)

    def test_interior_extrema(self):
        sine = interval_apply("SIN", [iv.mpf([0, iv.pi.b])])
        cosine = interval_apply("COS", [iv.mpf([-iv.pi.b, iv.pi.b])])
        square = interval_apply("SQUARE", [iv.mpf([-2, 1])])
        self.assert_encloses(sine, 0, 1)
        self.assert_encloses(cosine, -1, 1)
        self.assertEqual(low(square), 0)
        self.assertEqual(high(square), 4)

    def test_tangent_interior_pole(self):
        with self.assertRaises(Ineligible):
            interval_apply("TAN", [iv.mpf([1, 2])])
        self.assert_encloses(interval_apply("TAN", [iv.mpf([-1, 1])]),
                             mp.tan(-1), mp.tan(1))

    def test_division_signs(self):
        result = interval_apply("/", [iv.mpf([-2, 3]), iv.mpf([-4, -2])])
        self.assertEqual(low(result), mp.mpf("-1.5"))
        self.assertEqual(high(result), 1)
        for bounds in [(-1, 1), (0, 1), (-1, 0)]:
            with self.assertRaises(Ineligible):
                interval_apply("/", [iv.mpf(1), iv.mpf(bounds)])

    def test_domains(self):
        for token, bounds in [("LN", [0, 1]), ("SQRT", [-1, 1]),
                              ("ASIN", [-2, 0]), ("ACOS", [0, 2]),
                              ("W0", [-1, 0])]:
            with self.subTest(token=token), self.assertRaises(Ineligible):
                interval_apply(token, [iv.mpf(bounds)])
        self.assertEqual(low(interval_apply("LN", [iv.mpf(1)])), 0)
        self.assertEqual(low(interval_apply("SQRT", [iv.mpf(0)])), 0)

    def test_atan_rounding_regression(self):
        for value in ["-1000", "-1", "-0.5", "0", "0.5", "1", "1000"]:
            self.assert_encloses(interval_apply("ATAN", [iv.mpf(value)]),
                                 mp.atan(mp.mpf(value)))

    def test_inverse_trig_endpoints_and_monotonicity(self):
        for token, function in [("ASIN", mp.asin), ("ACOS", mp.acos)]:
            for bounds in [(-1, 1), (-1, 0), (0, 1), ("0.2", "0.8")]:
                result = interval_apply(token, [iv.mpf(bounds)])
                self.assert_encloses(result, function(mp.mpf(bounds[0])),
                                     function(mp.mpf(bounds[1])))

    def test_lambert_principal_branch(self):
        for bounds in [("-0.36", "-0.1"), ("-0.1", "1"), ("1", "100")]:
            result = interval_apply("W0", [iv.mpf(bounds)])
            self.assert_encloses(result, mp.lambertw(mp.mpf(bounds[0])),
                                 mp.lambertw(mp.mpf(bounds[1])))
            self.assertGreaterEqual(low(result), -1)
        self.assertEqual(low(interval_apply("W0", [iv.mpf(0)])), 0)
        near_branch = -1/mp.e + mp.mpf("1e-20")
        self.assert_encloses(interval_apply("W0", [iv.mpf(near_branch)]), mp.lambertw(near_branch))

    def test_constant_enclosures(self):
        for token in ["PI", "E", "HALF", "THIRD", "NEG_HALF"]:
            self.assert_encloses(constant(token, iv), constant(token, mp))

    def test_preview_guard_is_explicit(self):
        with self.assertRaisesRegex(Ineligible, "preview magnitude guard"):
            interval_apply("EXP", [iv.mpf(100)])

    def test_floor_and_ceiling_ranges(self):
        u=iv.mpf(["-1.2","1.2"])
        floor=interval_apply("FLOOR",[u]);ceil=interval_apply("CEIL",[u])
        self.assertEqual((low(floor),high(floor)),(-2,1))
        self.assertEqual((low(ceil),high(ceil)),(-1,2))

    def test_fractional_part_wraps_and_handles_negatives(self):
        wrapped=interval_apply("FRAC",[iv.mpf(["-.2",".2"])])
        self.assertEqual((low(wrapped),high(wrapped)),(0,1))
        negative=interval_apply("FRAC",[iv.mpf(["-1.3","-1.1"])])
        self.assert_encloses(negative,mp.mpf(".7"),mp.mpf(".9"))
        exact=interval_apply("FRAC",[iv.mpf(-2)])
        self.assertEqual((low(exact),high(exact)),(0,0))


class GeneratorTests(unittest.TestCase):
    def test_expanded_bounds_and_fixed_slot_generation(self):
        self.assertEqual(BOUND_TOKENS,("-2","-1","NEG_HALF","0","HALF","1","2"))
        generator=Generator(41)
        pairs={generator.sample(0)[:2] for _ in range(1000)}
        self.assertEqual(len(pairs),21)
        for _ in range(20):
            a,b,k,tree,_=generator.sample_fixed(12,"-2","2")
            self.assertEqual((a,b,k),("-2","2",12))
            self.assertEqual(sum(t in ARITY for t in tree.tokens()),12)

    def test_seed_and_rpn(self):
        first, second = Generator(37), Generator(37)
        for gamma in [0, 1, 3, 6, 12, 20]:
            a, b, k, tree, retries = first.sample(gamma)
            self.assertEqual(record(a, b, k, tree, retries), record(*second.sample(gamma)))
            row = record(a, b, k, tree, retries)
            tokens = row["rpn"].split()
            self.assertEqual(tokens[:2], [a, b])
            self.assertEqual(tokens[-2:], ["INT_X", "="])
            self.assertEqual(sum(t in ARITY for t in tokens), k)
            audit_tree(a, b, k, tree, points=33)


if __name__ == "__main__":
    unittest.main()
