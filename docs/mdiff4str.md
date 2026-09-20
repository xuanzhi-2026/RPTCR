# MDiff4STR integration

[Project overview](../README.md) · [Evaluation](evaluation.md) · [Validation](validation.md)

Use `rptcr.mdiff.readout` after native MDiff4STR-B decoding. The integration point is the decoder's cached visual memory and the complete posterior from its last executed round. Your OpenOCR caller must capture these tensors; this repository provides the terminal revision core.

## Source and environment

The experiments used [OpenOCR at `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`](https://github.com/Topdu/OpenOCR/tree/0d522801ec6dc1df852c6b6d4ed6a08f5127ed97), with Python 3.8.20, PyTorch 2.2.0+cu118, and CUDA 11.8. These versions record the experiment environment. Install OpenOCR and obtain the matching MDiff4STR-B checkpoint through the upstream project.

In that environment, run the following from the RPTCR repository root before starting your OpenOCR script:

```bash
export RPTCR_ROOT="$(pwd)"
export PYTHONPATH="${RPTCR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "from rptcr.mdiff import readout"
```

This keeps RPTCR importable when you change into the OpenOCR directory. Keep the upstream preprocessing, vocabulary, temperature, and text decoding unchanged, with the model in evaluation mode and its weights frozen.

## Inputs and frozen settings

The [configuration record](../configs/mdiff_t5_k4.json) specifies up to five native decoding steps, native early exit, four verification views, and `tau = 0.8817490935325623`. The JSON records these settings; `readout` uses the constants in [the implementation](../rptcr/mdiff.py).

| Argument | Required value |
| --- | --- |
| `decoder` | The matching OpenOCR MDiff decoder, exposing `get_masked_indice_after_eos`, `mask_token_id`, `eos`, `forward_decoding`, `tgt_word_prj`, and `temperature`. |
| `memory` | Cached visual features from the same native call, shape `[batch, visual_tokens, channels]`. |
| `factual` | Complete float32 probabilities from the last native round actually executed, shape `[batch, 26, 95]`. |
| `mode` | `"K4"` for RPTCR. |

The tensors must be on the decoder's device, with matching batch order. The recorded vocabulary uses EOS ID `0` and feedback mask ID `95`; the output distribution has 95 classes.

Preserve the native decoding trajectory, including early exit. Capture the complete last-round probability tensor before any merge with earlier rounds. OpenOCR's accumulated output can retain probabilities from previous rounds, so it is not interchangeable with `factual`. RPTCR derives its original token IDs as `factual.argmax(dim=-1)`.

## Call RPTCR

This is an integration snippet. It assumes your caller has already loaded the model and captured `decoder`, `memory`, and `terminal_probabilities` as described above.

```python
import torch
from rptcr.mdiff import readout

with torch.inference_mode():
    result = readout(
        decoder,
        memory,
        terminal_probabilities,
        mode="K4",
    )

output_ids = result.output_ids
accepted = output_ids.ne(terminal_probabilities.argmax(dim=-1))
```

Convert `output_ids` to text with the same upstream vocabulary and text decoder used for the native output.

`readout` returns a `Readout` object containing detached tensors:

| Field | Shape in `K4` mode | Meaning |
| --- | --- | --- |
| `output_ids` | `[batch, 26]` | Revised token IDs. |
| `candidate_ids` | `[batch, 26]` | Assigned-view candidates; inactive positions retain the original IDs. |
| `gain` | `[batch, 26]` | Candidate minus original log-probability; `-inf` at inactive positions. |
| `active` | `[batch, 26]` | Positions eligible for revision. |
| `masked_counts` | `[4, batch]` | Positions selected for masking in each view. |
| `target_counts` | `[4, batch]` | Assigned active positions in each view. |
| `decoy_counts` | `[4, batch]` | Collateral-mask counts; zero in `K4` mode. |

## Readout behavior

**Residue-partitioned cloze verification** groups positions by their index modulo four. Each view starts from the same prepared sequence; all four views are batched into one decoder call using the cached visual memory. A position is read only from its assigned view.

**Same-view selective revision** compares the candidate and original token under that assigned distribution. An edit requires a different candidate and `log p(candidate) - log p(original) > tau`. The implementation clamps probabilities to `1e-12` before taking logarithms. Threshold equality keeps the original token, and all accepted edits are applied together.

The active region includes the first EOS and one following slot, capped at 26 positions. Without EOS, every position is active and the upstream feedback preparation masks the whole sequence. Candidates come from the full output vocabulary, including EOS; do not restrict candidate selection to visible characters.

The core also retains `F` for unmasked rereading and `C4` for collateral masking as experimental controls. Run `K4` directly on the native terminal state; an `F` pass is not a prerequisite.

For a new model integration, retain both native outputs and the captured terminal tensors so that capture behavior and revision can be checked separately. See [evaluation](evaluation.md) for comparison requirements and [validation](validation.md) for the checks already completed.
