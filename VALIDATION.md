# Packaging validation

Checked on 2026-09-07. These are code-extraction and synthetic software checks, not a new benchmark run.

| Check | Outcome |
| --- | --- |
| MDiff readout source lock | Packaged `rptcr/mdiff.py` matches the entire frozen original source byte for byte; SHA-256 `0d6ebe9c74cf64559a172c0e4b2ef87cd8542e2d2af6614c075122ca8b3604b4` |
| Complete synthetic unit suite | 14 tests passed, no skips, using Python 3.9.6, NumPy 1.26.4 and PyTorch 2.2.0 on macOS ARM CPU |
| PIMNet source arithmetic equivalence | 256 synthetic terminal states, 1,024 threshold comparisons: preparation, active region, all view inputs, assigned logits, float32 log-probabilities, revised IDs and edit counts were bit-exact to extracted original functions; zero mismatches |
| Independent PIMNet review | Another reviewer checked 100 preparation cases, 100 view/readout cases and 400 threshold cases against the frozen source; zero mismatches |
| Import behavior | Package/PIM modules neither imported model backends nor modified the environment or created files in the working directory |
| Language syntax | PIM package and helper code parse as Python 3.6 syntax; the separate MDiff module parses as Python 3.8 syntax |
| Illustrative example | Constructed logits with threshold 2 revise `[[0,0]]` to `[[1,0]]`; no real sample is embedded |

The suite covers first EOS, repeated EOS, EOS at the last slot, no EOS, strict threshold equality, equal-score candidates, inactive-slot retention, independent residue masking, assigned readout, one simultaneous update, input immutability, common logit-shift invariance, non-finite rejection and the current frozen defaults. MDiff synthetic tests use actual PyTorch tensors and a deliberately small fake decoder; they do not load OpenOCR or a checkpoint.

The arithmetic comparison executed the original PIM function definitions extracted through Python AST, without executing the old experiment modules' import-time environment changes or file handling. This check establishes extraction consistency on synthetic inputs in one NumPy runtime, not numerical equality across different NumPy/TensorFlow versions.

The TensorFlow/PIMNet graph-factory wrapper is statically checked and reviewed against the source graph construction. TensorFlow 1.12.0 and real checkpoints were not loaded in the packaging environment. No new MDiff native-trajectory bridge is supplied. Real-model, target-CUDA and complete-dataset parity of this newly packaged interface remain outside these checks. Use the exact documented external model/input boundary; do not present these tests as benchmark reproduction.
