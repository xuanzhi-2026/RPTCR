# RPTCR

**Residue-Partitioned Terminal Cloze Revision for scene text recognition.**

RPTCR checks and revises a recognizer's output after decoding, without additional training. This repository provides the revision code used with MDiff4STR and PIMNet, model-specific configurations, and unit tests.

## Method

RPTCR has two components:

1. **Residue-partitioned cloze verification.** Positions are grouped by their index modulo `K`. Each group is masked in a separate copy of the final prediction, and the recognizer predicts those positions using cached visual features and the remaining characters.
2. **Same-view selective revision.** The candidate and original token are compared under the same verification distribution. A replacement is accepted when `log p(candidate) - log p(original) > tau`.

Each position is read from its assigned masked view. All views use the original sequence, and accepted replacements are applied together.

## Setup

Clone the repository and run the examples from its root:

```bash
git clone https://github.com/xuanzhi-2026/RPTCR.git
cd RPTCR
```

The experiments used separate environments for the two recognizers:

| Recognizer | Environment | Upstream version |
| --- | --- | --- |
| MDiff4STR | Python 3.8.20, PyTorch 2.2.0+cu118, CUDA 11.8 | [OpenOCR](https://github.com/Topdu/OpenOCR), `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97` |
| PIMNet | Python 3.6.13, TensorFlow 1.12.0, NumPy 1.16.6, CPU | [PIMNet](https://github.com/Pay20Y/PIMNet), `d4b1e39670b1cd7679e6f5a5364a340575d9bc4b` |

Install the recognizer and obtain its checkpoint following the upstream instructions. Keep its preprocessing, vocabulary, temperature, and text decoding settings unchanged. The recognizer must be in evaluation mode with frozen weights.

This package takes the recognizer's terminal state as input. Model code, checkpoints, datasets, and a complete image-to-text evaluation runner are not included. To import `rptcr` from another project, add this repository to `PYTHONPATH`.

## Configuration

| Recognizer | Native decoding steps | Verification views | Edit threshold |
| --- | --- | --- | --- |
| MDiff4STR-B | Up to 5, with native early exit | 4 | `0.8817490935325623` |
| PIMNet | 5 | 5 | `1.404470682144165` |

The settings are recorded in [configs/mdiff_t5_k4.json](configs/mdiff_t5_k4.json) and [configs/pimnet_t5_k5.json](configs/pimnet_t5_k5.json). These JSON files document the settings; the functions use the matching constants in their Python modules.

## MDiff4STR

Pass the cached visual memory and the **complete probability tensor from the last executed native decoding round** to `readout`:

```python
import torch
from rptcr.mdiff import readout

# decoder: frozen OpenOCR MDiff decoder
# memory: visual features, shape [batch, visual_tokens, channels]
# terminal_probabilities: float32 probabilities, shape [batch, 26, 95]
with torch.inference_mode():
    result = readout(decoder, memory, terminal_probabilities, mode="K4")

output_ids = result.output_ids
```

OpenOCR's merged output may retain probabilities from earlier rounds, so it cannot replace the last-round tensor in this call. The caller must capture that tensor during native decoding.

`K4` is the RPTCR mode. It batches four masked views into one decoder call and reuses the cached visual memory. The module also retains the experimental controls `F` (unmasked rereading) and `C4` (collateral masking). Run `K4` directly on the native terminal state; it does not require an `F` pass first.

## PIMNet

Pass the final token IDs and visual features from the same native decoding call:

```python
from rptcr.pimnet import revise

# base: integer token IDs, shape [1, 25], after five decoding steps
# features: visual features cached during that call
# graph: handles for "tokens", "supplied", and "read_logits"
result = revise(session, graph, base, features)

output_ids = result["output_ids"]
```

`revise` uses five masked views and supports a batch size of one. Its result also includes candidate IDs, log-probability gains, active positions, and accepted edits. Convert the output IDs to text with the upstream decoder.

The optional TensorFlow adapter is in `rptcr.pimnet_graph`:

- `build_graph(tf, model_factory)` creates the native decoding path and a verification path with shared weights. The factory is called inside the new graph as `model_factory(5)` and must return the upstream `Model` with the official `LOWERCASE` vocabulary, `seq_len=25`, and `is_training=False`.
- `open_session(tf, graph, checkpoint)` restores the EMA checkpoint in a CPU session.

## Implementation details

The active region includes the first EOS and one following slot, capped at the sequence length. All slots are active when EOS is absent. MDiff uses the upstream decoder's feedback preparation, which masks all feedback when no EOS is found; PIMNet preserves the sequence before applying residue masks.

Candidates are selected from the full output vocabulary, including special tokens. MDiff clamps probabilities to `1e-12` before taking logarithms; PIMNet computes log-softmax in NumPy float32. Rejected candidates leave the original token unchanged, including when the score difference equals the threshold.

## Tests

For the NumPy tests and the synthetic example:

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python -m examples.synthetic_gate
```

The suite checks masking, assigned-view readout, threshold ties, EOS handling, and input preservation. Three tests require PyTorch and are skipped if it is not installed. No checkpoint is needed for the tests.

See [VALIDATION.md](VALIDATION.md) for test results and model integration checks, and [PROVENANCE.md](PROVENANCE.md) for source versions and extraction details.

## License

A license for this repository has not yet been selected. Obtain the external recognizers and their assets under their respective upstream licenses.
