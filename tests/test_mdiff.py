import unittest
import hashlib
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


class SourceLockTest(unittest.TestCase):
    def test_exact_frozen_mdiff_core(self):
        source = Path(__file__).resolve().parents[1] / 'rptcr' / 'mdiff.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         '0d6ebe9c74cf64559a172c0e4b2ef87cd8542e2d2af6614c075122ca8b3604b4')


@unittest.skipIf(torch is None, 'PyTorch not available; no tensor execution claimed')
class MDiffTests(unittest.TestCase):
    def setUp(self):
        from rptcr import mdiff
        self.mdiff = mdiff

    def decoder(self):
        class Decoder(object):
            mask_token_id = 95
            eos = 0
            temperature = 1.0

            def get_masked_indice_after_eos(self, tokens):
                positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
                eos = tokens.eq(0)
                first = eos.to(torch.int64).argmax(1)[:, None]
                return (positions > first) | ~eos.any(1)[:, None]

            def forward_decoding(self, memory, views, step_i=0):
                self.views = views.clone()
                result = torch.zeros((views.shape[0], 26, 95), dtype=memory.dtype)
                result[:, :, 2] = 3.0
                return result

            def tgt_word_prj(self, hidden):
                return hidden
        return Decoder()

    def test_no_eos_masks_all_feedback(self):
        prepared, active, _ = self.mdiff.active_state(self.decoder(), torch.ones((1, 26), dtype=torch.long))
        self.assertTrue(prepared.eq(95).all().item())
        self.assertTrue(active.all().item())

    def test_eos_plus_one_and_independent_k4_views(self):
        base = torch.ones((1, 26), dtype=torch.long)
        base[0, 3] = 0
        views, active, positions, masks, _ = self.mdiff.build_views(self.decoder(), base, False)
        self.assertEqual(int(active.sum()), 5)
        self.assertEqual(len(views), 4)
        prepared, _, _ = self.mdiff.active_state(self.decoder(), base)
        for r, view in enumerate(views):
            expected = prepared.clone()
            expected[active & positions.remainder(4).eq(r)] = 95
            self.assertTrue(torch.equal(expected, view))
        self.assertEqual(int(base[0, 4]), 1)

    def test_selective_update_keeps_inactive_original(self):
        decoder = self.decoder()
        factual = torch.zeros((1, 26, 95))
        factual[:, :, 1] = 1
        factual[0, 3, 1] = 0
        factual[0, 3, 0] = 1
        original = factual.clone()
        with torch.inference_mode():
            result = self.mdiff.readout(decoder, torch.zeros((1, 2, 3)), factual, 'K4')
        self.assertTrue(result.output_ids[0, :5].eq(2).all().item())
        self.assertTrue(result.output_ids[0, 5:].eq(1).all().item())
        self.assertTrue(torch.equal(factual, original))
        self.assertEqual(tuple(decoder.views.shape), (4, 26))


if __name__ == '__main__':
    unittest.main()
