# Evaluation notes

[Back to README](../README.md)

RPTCR operates on the terminal state of a frozen recognizer. These notes describe how to compare the native prediction with its revised output using the packaged code. Model loading, image preprocessing, and dataset iteration belong to the external recognizer; this repository currently provides the revision functions and a synthetic example.

## Keep the native and revised runs paired

Use the same checkpoint, images, preprocessing, vocabulary, and text normalization for both outputs. Complete native decoding once, retain its output and visual features, and pass the required terminal state to RPTCR. Decode the revised IDs with the same upstream text decoder, including its EOS handling.

For MDiff4STR, request at most five native steps and preserve the upstream early-exit behavior. Capture the complete probability tensor from the last round actually executed, together with the cached visual memory. The decoder's merged output may contain probabilities from earlier rounds and cannot substitute for this tensor. The [MDiff4STR guide](mdiff4str.md) describes the readout interface.

For PIMNet, use five native steps and fetch the terminal token IDs and visual features from the same native call. The [PIMNet guide](pimnet.md) describes the shared-weight graph helper and the batch-size-one revision interface.

The four MDiff4STR views and five PIMNet views are additional verification work after native decoding. Record native steps and verification views separately.

## Configuration and token handling

The settings are recorded in [mdiff_t5_k4.json](../configs/mdiff_t5_k4.json) and [pimnet_t5_k5.json](../configs/pimnet_t5_k5.json). The modules use matching Python constants; editing a JSON file does not change an API call.

| Setting | MDiff4STR | PIMNet |
| --- | --- | --- |
| Sequence slots | 26 | 25 |
| Output classes | 95 | 38 |
| Mask token ID | 95 | 36 |
| EOS token ID | 0 | 37 |
| Verification views | 4 | 5 |
| Edit threshold | `0.8817490935325623` | `1.404470682144165` |

The active region extends through the first EOS and one following slot, capped at the sequence length. If EOS is absent, all positions are active. MDiff4STR uses the upstream feedback preparation, which masks all feedback in the no-EOS case; PIMNet retains that sequence before applying residue masks.

Candidates are selected from the full output vocabulary, including special tokens. MDiff4STR clamps probabilities to `1e-12` before taking logarithms; PIMNet computes log-softmax in NumPy float32. A replacement must change the original token and exceed the edit threshold strictly. Equality keeps the original token. All accepted edits are applied together to the original sequence.

## Report recognition and correction results

Measure word accuracy after applying the same text normalization to both predictions and the ground truth. Retain each sample's identifier, ground truth, native prediction, and revised prediction so the comparison can be checked per image.

- **Rescue:** an incorrect native word becomes correct.
- **Harm:** a correct native word becomes incorrect.
- **Net:** rescues minus harms.

For `N` paired samples, the word-accuracy change in percentage points is `100 * (rescue - harm) / N`. A changed prediction that remains incorrect contributes to neither rescue nor harm. These are word-level counts; they differ from the number of accepted token replacements returned by the code.

Report each benchmark's accuracy and sample count. When combining benchmarks, identify whether the result is a mean of benchmark accuracies or accuracy pooled across all samples, since those summaries weight the datasets differently. Keep dataset versions and exclusions explicit when comparing with published results.

## Check an integration

Run the native recognizer with and without terminal-state capture on the same inputs and compare its predictions before revision. Capture should preserve the original decoding trajectory, including early exit. Then check that the verification path reuses the loaded weights and visual features and that every active position is read from its assigned masked view.

The packaged [unit tests](../tests) exercise the revision rules with synthetic inputs. The [validation record](validation.md) separates those checks from real-model and dataset evaluation.
