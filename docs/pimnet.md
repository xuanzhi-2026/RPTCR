# PIMNet integration

[Project overview](../README.md) · [Evaluation](evaluation.md) · [Validation](validation.md)

Use `rptcr.pimnet.revise` with PIMNet's final token IDs and the visual features cached during the same native decoding call. The NumPy revision core accepts an existing TensorFlow session; `rptcr.pimnet_graph` can construct the native and verification paths with shared weights.

## Source and environment

The experiments used [PIMNet at `d4b1e39670b1cd7679e6f5a5364a340575d9bc4b`](https://github.com/Pay20Y/PIMNet/tree/d4b1e39670b1cd7679e6f5a5364a340575d9bc4b), with Python 3.6.13, TensorFlow 1.12.0, NumPy 1.16.6, and CPU execution. These versions record the experiment environment, which is separate from the MDiff4STR environment. Obtain the model source and checkpoint through the upstream project.

In the PIMNet environment, run the following from the RPTCR repository root before starting your PIMNet script:

```bash
export RPTCR_ROOT="$(pwd)"
export PYTHONPATH="${RPTCR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "from rptcr.pimnet import revise"
```

This keeps RPTCR importable when you change into the PIMNet directory. Keep the official `LOWERCASE` vocabulary, preprocessing, and text decoding settings. The model must use `seq_len=25`, `is_training=False`, and frozen weights.

## Inputs and frozen settings

The [configuration record](../configs/pimnet_t5_k5.json) specifies five native decoding steps, five verification views, and `tau = 1.404470682144165`. The JSON records these settings; `revise` uses the constants in [the implementation](../rptcr/pimnet.py).

| Argument | Required value |
| --- | --- |
| `sess` | A TensorFlow session containing the restored PIMNet model and verification path. |
| `graph` | A mapping with `tokens`, `supplied`, and `read_logits` handles. |
| `base` | Integer token IDs of shape `[1, 25]` after five native decoding steps. |
| `features` | Cached visual features from that same call, matching the `supplied` placeholder. |
| `tau` | Omit to use the frozen threshold. |

The verification path accepts int32 tokens of shape `[1, 25]` and returns logits of shape `[1, 25, 38]`. The official vocabulary uses mask ID `36` and EOS ID `37`. The public `revise` entry point supports batch size one.

## Call RPTCR in an existing session

This integration snippet assumes your caller has constructed the shared-weight verification path, restored the checkpoint, and obtained `base` and `features` from one native call.

```python
from rptcr.pimnet import revise

result = revise(session, graph, base, features)
output_ids = result["output_ids"]
accepted = result["edits"]
```

Convert `output_ids` to text with the upstream decoder. All returned arrays have shape `[1, 25]`:

| Field | Meaning |
| --- | --- |
| `output_ids` | Revised token IDs. |
| `candidate_ids` | Candidates selected from the assigned-view logits. |
| `gain` | Candidate minus original log-probability. |
| `active` | Positions eligible for revision. |
| `edits` | Accepted replacements. |

Candidate IDs and gains are also computed at inactive positions; use `active` and `edits` to interpret them. Inactive and rejected positions retain the original IDs in `output_ids`.

## Optional graph adapter

[The graph adapter](../rptcr/pimnet_graph.py) calls your `model_factory(5)` inside a new TensorFlow graph. The factory must construct the pinned upstream `Model` with the configuration above. It must not return a model constructed in a different graph.

The following integration snippet assumes you supply `model_factory`, an EMA `checkpoint` path, and the upstream inference inputs: `input_images` as float32 `[1, 64, 256, 1]` and `input_labels` as int32 `[1, 25]`. Prepare these inputs according to the upstream inference code.

```python
import tensorflow as tf
from rptcr.pimnet import revise
from rptcr.pimnet_graph import build_graph, open_session

graph = build_graph(tf, model_factory)
session = open_session(tf, graph, checkpoint)
try:
    base, features = session.run(
        [graph["native"], graph["features"]],
        feed_dict={
            graph["images"]: input_images,
            graph["labels"]: input_labels,
        },
    )
    result = revise(session, graph, base, features)
    output_ids = result["output_ids"]
finally:
    session.close()
```

`build_graph` reuses the model's verification weights and checks that constructing the verification path introduces no trainable variables. `open_session` restores the EMA checkpoint into a CPU session. Fetching `native` and `features` together keeps the terminal IDs and cached image features tied to the same native call.

## Revision behavior

**Residue-partitioned cloze verification** groups positions by their index modulo five. Each view starts from the same prepared sequence and masks its assigned active positions. The implementation makes five verification calls with the cached features and reads each position from its assigned view.

**Same-view selective revision** compares the candidate and original token under the same verification distribution, computed with NumPy float32 log-softmax. An edit requires a different candidate and `log p(candidate) - log p(original) > tau`. Threshold equality keeps the original token, and all accepted edits are applied together.

The active region includes the first EOS and one following slot, capped at 25 positions. Feedback after the first EOS is masked. If EOS is absent, the sequence is preserved before applying residue masks and all positions are active. Candidates come from all 38 output classes, including special tokens; do not filter them to visible characters before revision.

See [evaluation](evaluation.md) for comparison requirements and [validation](validation.md) for completed checks, including the current graph-adapter verification status.
