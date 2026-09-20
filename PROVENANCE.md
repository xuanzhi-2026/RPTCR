# Source provenance

All references below are source identifiers, not required runtime filesystem paths. Original experimental material is not bundled.

| Packaged file | Original source | SHA-256 of original source | Extraction |
| --- | --- | --- | --- |
| `rptcr/mdiff.py` | `method_sources_v1/mdiff_confirmation_core.py` | `0d6ebe9c74cf64559a172c0e4b2ef87cd8542e2d2af6614c075122ca8b3604b4` | Byte-identical 131-line readout core, including original F/K4/C4 mode definitions |
| `rptcr/pimnet.py` | `method_sources_v1/pimnet_gate1_reference.py` | `8eefe69f3c058a7117706b066c336a42342094066d6a839d417884c2f32fc1ba` | `prepare`, `log_prob`, `selected_for_k`: removed global vocabulary and experiment coupling; token IDs supplied as parameters/defaults; arithmetic and K-view readout preserved |
| `rptcr/pimnet.py` | `method_sources_v1/pimnet_evaluate_dataset_shard.py` | `7e009876d18b7b882e46526a8ee086ae685937bec97f919253176b77f76b30ea` | `apply_arm`: current frozen tau; returns IDs/diagnostics instead of upstream decoded string and scalar edit count; arithmetic preserved |
| `rptcr/pimnet_graph.py` | `method_sources_v1/pimnet_gate1_reference.py` | `8eefe69f3c058a7117706b066c336a42342094066d6a839d417884c2f32fc1ba` | Capture/native/reread/EMA restore logic extracted; TensorFlow and model construction supplied explicitly; checkpoint path discovery, import-time environment changes, results and dataset handling removed |

The two original source locks were independently verified during the method audit. The MDiff decoder itself was read to check its interface, but is not included here. The native trajectory bridge and the older `sc_pact` loader are also not included: they are distinct from the terminal RPTCR algorithm and should not be mistaken for newly validated package components.

External implementation versions: OpenOCR `0d522801ec6dc1df852c6b6d4ed6a08f5127ed97`; PIMNet `d4b1e39670b1cd7679e6f5a5364a340575d9bc4b`. Third-party model licensing remains with its authors; their source and checkpoints must be obtained separately under the upstream terms.

The API glue and synthetic tests are newly organized for this package. No training, real-image inference or new model-checkpoint parity run was performed while packaging them. The provenance does not grant any additional third-party rights.
