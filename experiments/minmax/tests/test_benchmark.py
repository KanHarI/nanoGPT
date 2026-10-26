"""The chosen batch must retain memory headroom and avoid unnecessary size."""

import unittest

from experiments.minmax.benchmark import choose_candidate


class BenchmarkTests(unittest.TestCase):
    def test_choose_smaller_near_best_and_reject_unsafe_candidates(self):
        rows = [dict(status="ok", batch_size=size, answers_per_second=rate,
                     memory_fraction=memory) for size, rate, memory in
                [(64, 600, 0.1), (128, 970, 0.2), (256, 1000, 0.4), (512, 1500, 0.95)]]
        rows.append(dict(status="oom", batch_size=1024))
        self.assertEqual(choose_candidate(rows)["batch_size"], 128)
        with self.assertRaises(RuntimeError):
            choose_candidate(rows[-2:])


if __name__ == "__main__":
    unittest.main()
