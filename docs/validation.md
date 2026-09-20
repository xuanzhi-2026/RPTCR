# Packaging validation

[Back to README](../README.md)

Validation covers source extraction and synthetic software behavior. No benchmark was run.

## Original checks: 2026-09-07

| Check | Outcome |
| --- | --- |
| MDiff readout source lock | `rptcr/mdiff.py` matched the complete frozen source byte for byte; SHA-256 `0d6ebe9c74cf64559a172c0e4b2ef87cd8542e2d2af6614c075122ca8b3604b4` |
| Complete synthetic unit suite | 14 tests passed, no skips, using Python 3.9.6, NumPy 1.26.4 and PyTorch 2.2.0 on macOS ARM CPU |
| PIMNet source arithmetic equivalence | 256 synthetic terminal states, 1,024 threshold comparisons: preparation, active region, all view inputs, assigned logits, float32 log-probabilities, revised IDs and edit counts were bit-exact to extracted original functions; zero mismatches |
| Independent PIMNet review | An independent review checked 100 preparation cases, 100 view/readout cases and 400 threshold cases against the frozen source; zero mismatches |
| Import behavior | Importing the package and PIM modules did not load model backends, modify the environment or create files in the working directory |
| Language syntax | PIM package and helper code parsed as Python 3.6 syntax; the separate MDiff module parsed as Python 3.8 syntax |
| Illustrative example | Synthetic logits with threshold 2 revised `[[0,0]]` to `[[1,0]]` |

## Local checks: 2026-09-20

| Check | Outcome |
| --- | --- |
| Synthetic unit suite | 11 tests passed; 3 PyTorch-dependent tests were skipped because PyTorch was not installed |
| Illustrative example | Passed using synthetic logits |
| Documentation reorganization | All 44 relative links resolve; the three Python API snippets parse successfully |
| Source preservation after reorganization | Code, configurations, examples, and tests are byte-identical to the preceding revision |

The three skipped tests were included in the complete 2026-09-07 run above.

## Coverage and limits

The suite covers first EOS, repeated EOS, EOS at the last slot, no EOS, strict threshold equality, equal-score candidates, inactive-slot retention, independent residue masking, assigned readout, one simultaneous update, input immutability, common logit-shift invariance, non-finite rejection and the frozen defaults. MDiff tests use PyTorch tensors and a small fake decoder, without loading OpenOCR or a checkpoint.

The arithmetic comparison ran original PIM function definitions extracted through Python AST, bypassing the experiment modules' import-time environment changes and file handling. It checked extraction consistency on synthetic inputs in one NumPy runtime. Numerical equality across NumPy or TensorFlow versions was not tested.

The TensorFlow/PIMNet graph factory was checked statically against the source graph construction. TensorFlow 1.12.0 and real checkpoints were not loaded. The package omits the MDiff native trajectory bridge, and real-model, target-CUDA and complete-dataset parity remain untested. Integration requires the external model and input interfaces documented in the [MDiff4STR guide](mdiff4str.md) and [PIMNet guide](pimnet.md).
