# Source provenance

The paths below identify the original sources; they are not required at runtime. The original experiment files are not bundled.

| Packaged file | Original source | SHA-256 of original source | Extraction |
| --- | --- | --- | --- |
| `rptcr/mdiff.py` | `method_sources_v1/mdiff_confirmation_core.py` | `0d6ebe9c74cf64559a172c0e4b2ef87cd8542e2d2af6614c075122ca8b3604b4` | The complete 131-line readout core, unchanged byte for byte, including the F/K4/C4 modes. |
| `rptcr/pimnet.py` | `method_sources_v1/pimnet_gate1_reference.py` | `8eefe69f3c058a7117706b066c336a42342094066d6a839d417884c2f32fc1ba` | `prepare`, `log_prob` and `selected_for_k`: token IDs are parameters with defaults, replacing the global vocabulary and experiment dependencies. Arithmetic and K-view readout are unchanged. |
| `rptcr/pimnet.py` | `method_sources_v1/pimnet_evaluate_dataset_shard.py` | `7e009876d18b7b882e46526a8ee086ae685937bec97f919253176b77f76b30ea` | `apply_arm`: retains the frozen tau and arithmetic; returns IDs and diagnostics in place of a decoded string and scalar edit count. |
| `rptcr/pimnet_graph.py` | `method_sources_v1/pimnet_gate1_reference.py` | `8eefe69f3c058a7117706b066c336a42342094066d6a839d417884c2f32fc1ba` | Extracts capture, native decoding, reread and EMA restore logic. Callers supply TensorFlow and model construction. Checkpoint discovery, import-time environment changes, and dataset/result handling are omitted. |

The two original source locks were independently verified during the method audit. The MDiff decoder was inspected to check its interface but is not bundled. The native trajectory bridge and older `sc_pact` loader are also excluded; they are separate from the terminal RPTCR algorithm.

External implementation versions: OpenOCR `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`; PIMNet `d4b1e39670b1cd7679e6f5a5364a340575d9bc4b`. Obtain their source and checkpoints separately under the upstream licenses; this package grants no additional third-party rights.

The API wrappers and synthetic tests were organized for this package. Packaging did not include training, real-image inference or a new model-checkpoint parity run. The 2026-09-20 documentation edits preserve the MDiff file byte for byte and leave PIMNet arithmetic unchanged.
