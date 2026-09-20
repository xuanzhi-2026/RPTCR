"""A tiny acceptance example. No image, model, checkpoint or dataset needed."""
import numpy as np
from rptcr.pimnet import apply_arm

# Constructed logits: slot 0 supports replacement; slot 1 does not.
base = np.array([[0, 0]], dtype=np.int32)
active = np.array([[True, True]])
logits = np.array([[[0.0, 3.0], [0.0, 0.5]]], dtype=np.float32)
result = apply_arm(base, active, logits, tau=2.0)
print('Constructed input:', base.tolist())
print('Revised token IDs:', result['output_ids'].tolist())
print('Accepted positions:', result['edits'].tolist())
