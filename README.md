# RPTCR

Residue-Partitioned Terminal Cloze Revision for scene text recognition.

This repository contains the RPTCR postprocessing code extracted from our frozen MDiff4STR and PIMNet experiments, plus explicit model interfaces. It does not contain third-party recognizers, weights, datasets, predictions, experiment logs or a model downloader. It does not train a model.

RPTCR creates independent masked residue views of one terminal sequence, selects each position's prediction from its assigned view, and accepts a replacement only when its same-view log-probability margin is strictly greater than a threshold. Rejected positions retain their original terminal token. All decisions are applied together once.

## Frozen configurations

| Model | Native steps | Views | Threshold | Input to revision |
| --- | --- | --- | --- | --- |
| MDiff4STR-B | up to 5, preserving native early exit | 4 | 0.8817490935325623 | Last actually executed round's complete probability tensor and cached visual memory |
| PIMNet | 5 | 5 | 1.404470682144165 | Terminal token IDs and the visual features cached in the same native call |

The machine-readable configurations are in `configs/`. The illustrated K=3, tau=2 example sometimes used in figures is not either frozen model configuration.

## Environments and external models

The two evaluated backends used separate environments:

| Backend | Evaluated environment | External implementation |
| --- | --- | --- |
| MDiff4STR | Python 3.8.20, PyTorch 2.2.0+cu118, CUDA 11.8 | [OpenOCR](https://github.com/Topdu/OpenOCR), commit `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97` |
| PIMNet | Python 3.6.13, TensorFlow 1.12.0, NumPy 1.16.6, CPU | [PIMNet](https://github.com/Pay20Y/PIMNet), commit `d4b1e39670b1cd7679e6f5a5364a340575d9bc4b` |

Follow each upstream project's installation, preprocessing, vocabulary, checkpoint and license instructions. Supply checkpoints yourself. No third-party model implementation or derivative native decoding loop is bundled. No rights to third-party assets are granted by this repository. No open-source license for this repository has been selected.

Use this source tree on `PYTHONPATH` or run Python from its root. The package initializer has no backend import. `rptcr.pimnet` needs NumPy and remains Python-3.6 compatible; `rptcr.mdiff` uses the separate MDiff Python/PyTorch environment. TensorFlow is supplied explicitly to `rptcr.pimnet_graph`; importing the package never changes CUDA/CPU settings or opens files.

## MDiff4STR interface

`rptcr/mdiff.py` is a byte-identical copy of the verified readout core. Its `K4` mode is RPTCR; `F` and `C4` are the original unmasked/collateral experimental controls retained to avoid changing that source. The public call is:

```python
import torch
from rptcr.mdiff import readout

# Obtain these from your external MDiff adapter:
# decoder: evaluated frozen OpenOCR MDiff decoder
# memory: cached visual tensor [batch, visual_tokens, channels]
# terminal_probabilities: [batch, 26, 95], float32
with torch.inference_mode():
    result = readout(decoder, memory, terminal_probabilities, mode='K4')
output_ids = result.output_ids
```

The input must be the complete probabilities from the final native round that actually executed at a requested T=5. Upstream MDiff's merged output can retain probabilities from earlier rounds; it is not interchangeable with this terminal tensor. This repository does not provide a newly rewritten or hook-based native trajectory bridge and does not claim an end-to-end parity test for such a bridge. It exposes the exact input boundary of the tested RPTCR readout. Keep the original model in evaluation mode, parameters frozen, full output vocabulary, temperature and preprocessing unchanged.

The four RPTCR views are batched together in the original core. The decoder receives four copies of the cached visual memory; this is four verification views in one batched decoder invocation. No F output is required before K4.

## PIMNet interface

For an already constructed source-compatible graph, provide its `tokens`, `supplied` and `read_logits` handles together with native T5 state:

```python
from rptcr.pimnet import revise

# base: integer IDs [1,25] from the native T5 model
# features: cached visual features from that same native call
# graph: mapping with tokens / supplied / read_logits
result = revise(session, graph, base, features)
output_ids = result['output_ids']
```

`rptcr.pimnet_graph.build_graph(tf, model_factory)` also extracts our evaluated graph adapter. `model_factory(5)` must construct the original PIMNet `Model` with its official LOWERCASE vocabulary/configuration, `seq_len=25`, and `is_training=False`, inside the graph context supplied by the helper. The helper adds shared-weight terminal rereading and checks that the variable list is unchanged. `open_session(tf, graph, checkpoint)` restores an explicitly supplied checkpoint using the evaluated CPU session configuration. This packaging of the factory boundary has not been rerun with a real PIMNet checkpoint in the local packaging environment.

Use upstream image preprocessing and text normalization/decoding. `apply_arm` exposes token IDs and diagnostics instead of importing the upstream string decoder. The packaged PIMNet inference wrapper supports the evaluated batch size of one.

## Numerical and EOS behavior

- The acceptance condition is `active & (candidate != base) & (gain > tau)`.
- MDiff uses softmax probabilities, clamps them to 1e-12, then subtracts logarithms. PIMNet uses the original NumPy float32 log-softmax. These numerical paths are deliberately separate.
- The active region includes the first EOS and one subsequent slot, bounded by sequence length. Without EOS, all slots are active.
- MDiff delegates feedback preparation to its original decoder method, including that method's all-MASK behavior when no EOS occurs. PIMNet leaves a no-EOS sequence unchanged before residue masking.
- Candidates use the complete output vocabulary. Special output tokens are not silently removed.
- Every view starts from the same prepared terminal sequence and shares the cached visual features. One view's proposed edits never enter another view.

## Validation and reproduction status

```bash
python -m unittest discover -s tests -v
python -m examples.synthetic_gate
```

The tests cover threshold ties, incumbent preservation, EOS/no-EOS behavior, independent masking, assigned readout and input immutability. The NumPy tests can run without a model. PyTorch tests skip explicitly when PyTorch is unavailable; no checkpoint is required for them. The packaged graph adapter is syntax checked but requires the external TensorFlow/PIMNet environment for a real-model check.

The complete 14-test suite passed with no skips in a separate Python 3.9.6 / NumPy 1.26.4 / PyTorch 2.2.0 environment on macOS ARM CPU. This includes the actual PyTorch synthetic tensor tests, not a CUDA or real-checkpoint reproduction.

`PROVENANCE.md` records source versions, hashes and extraction changes. `VALIDATION.md` records what was actually checked during packaging. This release is an extraction of verified algorithm code, not a claim that the new package has rerun all benchmark experiments or proved parity for a new end-to-end model loader.
