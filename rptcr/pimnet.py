"""RPTCR revision for PIMNet.

Uses NumPy and supports Python 3.6. The caller supplies the TensorFlow session
and PIMNet graph handles.
"""
import numpy as np

T = 5
K = 5
TAU = 1.404470682144165
SLOTS = 25
MASK_ID = 36
EOS_ID = 37


def prepare(base, mask_id=MASK_ID, eos_id=EOS_ID):
    """Prepare feedback and active slots, including EOS plus one extra slot."""
    base = np.asarray(base)
    eos = base == eos_id
    first = np.where(eos.any(1), eos.argmax(1), base.shape[1] - 1)
    positions = np.arange(base.shape[1])[None, :]
    active = positions <= np.minimum(first + 1, base.shape[1] - 1)[:, None]
    prepared = np.where(positions > first[:, None], mask_id, base)
    return prepared.astype(np.int32), active


def log_prob(logits):
    """Compute log-softmax in float32."""
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def selected_for_k(sess, graph, prepared, active, features, k=K,
                   mask_id=MASK_ID):
    """Run independent cloze views and read each position from its own view."""
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('k must be a positive integer')
    selected = None
    positions = np.arange(prepared.shape[1])[None, :]
    for residue in range(k):
        view = prepared.copy()
        view[active & (positions % k == residue)] = mask_id
        logits = sess.run(graph['read_logits'], {
            graph['tokens']: view,
            graph['supplied']: features,
        })
        if selected is None:
            selected = np.empty_like(logits)
        selected[:, residue::k, :] = logits[:, residue::k, :]
    return selected


def apply_arm(base, active, logits, tau=TAU):
    """Accept candidates whose log-probability gain exceeds tau.

    Rejected candidates retain the original token from base, not the masked
    feedback token. Return revised IDs and per-position diagnostics.
    """
    selected = log_prob(logits)
    candidate = selected.argmax(-1)
    incumbent = np.take_along_axis(selected, base[..., None], axis=-1)[..., 0]
    gain = selected.max(-1) - incumbent
    active_indices = np.where(active[0])[0]
    active_gain = gain[0, active_indices]
    if not np.isfinite(active_gain).all():
        raise RuntimeError('Non-finite gain')
    edits = active & (candidate != base) & (gain > tau)
    revised = np.where(edits, candidate, base)
    return {'output_ids': revised, 'candidate_ids': candidate,
            'gain': gain, 'active': active, 'edits': edits}


def revise(sess, graph, base, features, tau=TAU):
    """Revise the final PIMNet sequence using five masked views.

    base contains integer IDs of shape [1, 25] after five native decoding
    steps. features is the visual tensor cached during the same call.
    """
    base = np.asarray(base)
    if base.shape != (1, SLOTS) or base.dtype.kind not in 'iu':
        raise ValueError('PIMNet base must be integer token IDs of shape [1,25]')
    if not np.isfinite(tau) or tau < 0:
        raise ValueError('tau must be a finite nonnegative scalar')
    prepared, active = prepare(base)
    selected = selected_for_k(sess, graph, prepared, active, features, K)
    if not np.isfinite(selected).all():
        raise RuntimeError('Non-finite K5 logits')
    return apply_arm(base, active, selected, tau)
