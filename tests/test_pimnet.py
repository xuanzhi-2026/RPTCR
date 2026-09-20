import unittest
import numpy as np
from rptcr import pimnet


class RecordingSession(object):
    """Return distinct per-view proposals and record the exact feedback input."""
    def __init__(self):
        self.views = []
        self.feature_objects = []

    def run(self, fetch, feed):
        view = feed['tokens']
        self.views.append(view.copy())
        self.feature_objects.append(feed['features'])
        residue = len(self.views) - 1
        result = np.zeros((1, 25, 38), dtype=np.float32)
        result[:, :, residue + 1] = 8.0
        return result


GRAPH = {'read_logits': 'read', 'tokens': 'tokens', 'supplied': 'features'}


class PIMNetTests(unittest.TestCase):
    def test_current_configuration(self):
        self.assertEqual((pimnet.T, pimnet.K, pimnet.TAU),
                         (5, 5, 1.404470682144165))

    def test_first_eos_plus_one_and_original_unchanged(self):
        base = np.array([[1, 37, 3, 37, 5]], dtype=np.int32)
        original = base.copy()
        prepared, active = pimnet.prepare(base)
        np.testing.assert_array_equal(prepared, [[1, 37, 36, 36, 36]])
        np.testing.assert_array_equal(active, [[True, True, True, False, False]])
        np.testing.assert_array_equal(base, original)

    def test_no_eos_preserves_sequence(self):
        base = np.array([[1, 2, 3, 4]], dtype=np.int32)
        prepared, active = pimnet.prepare(base)
        np.testing.assert_array_equal(prepared, base)
        self.assertTrue(active.all())

    def test_last_eos_does_not_extend_past_array(self):
        prepared, active = pimnet.prepare(np.array([[1, 2, 37]], dtype=np.int32))
        np.testing.assert_array_equal(prepared, [[1, 2, 37]])
        self.assertTrue(active.all())

    def test_strict_threshold_and_original_fallback(self):
        base = np.array([[0, 0, 0]], dtype=np.int32)
        active = np.array([[True, True, False]])
        logits = np.array([[[0, 3], [0, 1], [0, 8]]], dtype=np.float32)
        result = pimnet.apply_arm(base, active, logits, tau=2)
        np.testing.assert_array_equal(result['output_ids'], [[1, 0, 0]])
        # Exactly equal support must KEEP; a strictly lower threshold must EDIT.
        tie = float(result['gain'][0, 0])
        self.assertFalse(pimnet.apply_arm(base, active, logits, tau=tie)['edits'][0, 0])
        lower = np.nextafter(np.float32(tie), np.float32(-np.inf))
        self.assertTrue(pimnet.apply_arm(base, active, logits, tau=lower)['edits'][0, 0])

    def test_candidate_incumbent_tie_keeps(self):
        base = np.array([[1]], dtype=np.int32)
        result = pimnet.apply_arm(base, np.array([[True]]), np.zeros((1, 1, 3)), tau=0)
        self.assertEqual(int(result['output_ids'][0, 0]), 1)

    def test_independent_views_assigned_readout_and_one_update(self):
        session = RecordingSession()
        base = np.zeros((1, 25), dtype=np.int32)
        features = object()
        result = pimnet.revise(session, GRAPH, base, features)
        self.assertEqual(len(session.views), 5)
        for r, view in enumerate(session.views):
            expected = base.copy()
            expected[:, r::5] = 36
            np.testing.assert_array_equal(view, expected)
            self.assertIs(session.feature_objects[r], features)
        expected = (np.arange(25) % 5 + 1)[None, :]
        np.testing.assert_array_equal(result['output_ids'], expected)
        np.testing.assert_array_equal(base, np.zeros((1, 25), dtype=np.int32))

    def test_nan_rejected(self):
        logits = np.array([[[0.0, np.nan]]], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            pimnet.apply_arm(np.array([[0]]), np.array([[True]]), logits)

    def test_logits_shift_invariance(self):
        logits = np.array([[[1.0, 4.0, 2.0]]], dtype=np.float32)
        result = pimnet.apply_arm(np.array([[0]]), np.array([[True]]), logits)
        shifted = pimnet.apply_arm(np.array([[0]]), np.array([[True]]), logits+100)
        np.testing.assert_array_equal(result['output_ids'], shifted['output_ids'])
        np.testing.assert_array_equal(result['gain'], shifted['gain'])


if __name__ == '__main__':
    unittest.main()
