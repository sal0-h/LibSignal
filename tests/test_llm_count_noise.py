"""Cluster Traffic-R1 noisy-lane_count adapter."""

import pathlib
import unittest


def _load_rescale():
    import math
    from typing import List, Sequence, Tuple

    text = pathlib.Path("agent/llm_tsc.py").read_text(encoding="utf-8")
    start = text.index("def _rescale_observation_bins")
    end = text.index("@Registry.register_model(\"traffic_r1\")")
    namespace = {
        "math": math,
        "List": List,
        "Sequence": Sequence,
        "Tuple": Tuple,
    }
    exec(compile(text[start:end], "agent/llm_tsc.py", "exec"), namespace)
    return namespace["_rescale_observation_bins"]


class RescaleObservationBinsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rescale = _load_rescale()

    def test_matching_total_is_a_no_op(self):
        rescale = type(self).rescale
        self.assertEqual(rescale(3, [1, 2, 4], 10), (3, [1, 2, 4]))

    def test_sum_matches_noisy_total(self):
        rescale = type(self).rescale
        early, segments = rescale(3, [1, 2, 4], 7)
        self.assertEqual(early + sum(segments), 7)
        self.assertTrue(early >= 0 and all(value >= 0 for value in segments))

    def test_largest_remainder_is_deterministic(self):
        rescale = type(self).rescale
        first = rescale(1, [1, 1], 4)
        second = rescale(1, [1, 1], 4)
        self.assertEqual(first, second)
        self.assertEqual(first, (2, [1, 1]))

    def test_empty_lane_false_positive_goes_to_last_segment(self):
        rescale = type(self).rescale
        self.assertEqual(rescale(0, [0, 0, 0], 3), (0, [0, 0, 3]))

    def test_target_zero_clears_bins(self):
        rescale = type(self).rescale
        self.assertEqual(rescale(2, [1, 0, 3], 0), (0, [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
