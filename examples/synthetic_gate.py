"""Show selective revision for two positions using synthetic logits."""
import numpy as np
from rptcr.pimnet import apply_arm

# Only the first position exceeds the edit threshold.
base = np.array([[0, 0]], dtype=np.int32)
active = np.array([[True, True]])
logits = np.array([[[0.0, 3.0], [0.0, 0.5]]], dtype=np.float32)
result = apply_arm(base, active, logits, tau=2.0)
print('Constructed input:', base.tolist())
print('Revised token IDs:', result['output_ids'].tolist())
print('Accepted positions:', result['edits'].tolist())
