#!/usr/bin/env python3
"""Frozen T5-K4 and compute-matched C4 readout primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


SLOTS = 26
CLASSES = 95
F_TAU = 0.3170628249645233
RPTCR_TAU = 0.8817490935325623


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass
class Readout:
    output_ids: Any
    candidate_ids: Any
    gain: Any
    active: Any
    masked_counts: Any
    target_counts: Any
    decoy_counts: Any


def active_state(decoder: Any, base_ids: Any) -> tuple[Any, Any, Any]:
    import torch

    prepared = base_ids.clone()
    prepared[decoder.get_masked_indice_after_eos(prepared)] = int(decoder.mask_token_id)
    eos_mask = base_ids.eq(int(decoder.eos))
    eos_positions = eos_mask.to(torch.int64).argmax(dim=1)
    fallback = torch.full_like(eos_positions, int(base_ids.shape[1]) - 1)
    eos_positions = torch.where(eos_mask.any(dim=1), eos_positions, fallback)
    positions = torch.arange(base_ids.shape[1], device=base_ids.device).unsqueeze(0).expand_as(base_ids)
    active = positions.le((eos_positions + 1).clamp_max(base_ids.shape[1] - 1).unsqueeze(1))
    return prepared, active, positions


def build_views(decoder: Any, base_ids: Any, collateral: bool) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    prepared, active, positions = active_state(decoder, base_ids)
    residue = positions.remainder(4)
    views = []
    masks = []
    targets = []
    decoys = []
    for value in range(4):
        target = active & residue.eq(value)
        if collateral:
            translated = (positions - int(value)).remainder(8)
            decoy = active & (translated.eq(1) | translated.eq(3))
        else:
            decoy = torch.zeros_like(active)
        require(bool((target & decoy).sum().eq(0).item()), "decoy overlaps assigned target")
        selected = target | decoy
        view = prepared.clone()
        view[selected] = int(decoder.mask_token_id)
        views.append(view)
        masks.append(selected)
        targets.append(target)
        decoys.append(decoy)
    assigned_self_mask = torch.zeros_like(active)
    for value, mask in enumerate(masks):
        assigned_self_mask |= mask & residue.eq(value)
    require(bool(torch.where(active, assigned_self_mask, torch.ones_like(active)).all().item()),
            "assigned token not self-masked")
    return tuple(views), active, positions, torch.stack(masks), torch.stack(decoys)


def readout(decoder: Any, memory: Any, factual: Any, mode: str) -> Readout:
    import torch
    import torch.nn.functional as F

    require(mode in ("F", "K4", "C4"), "unknown readout mode")
    base_ids = factual.argmax(dim=-1)
    prepared, active, positions = active_state(decoder, base_ids)
    if mode == "F":
        views = (prepared,)
        masks = torch.zeros((1,) + tuple(active.shape), dtype=torch.bool, device=active.device)
        decoys = masks.clone()
    else:
        views, active, positions, masks, decoys = build_views(decoder, base_ids, collateral=(mode == "C4"))
    count = len(views)
    hidden = decoder.forward_decoding(
        torch.cat([memory for _ in range(count)], dim=0), torch.cat(list(views), dim=0), step_i=0
    )
    probabilities = F.softmax(
        decoder.tgt_word_prj(hidden) / float(decoder.temperature), dim=-1
    ).reshape(count, int(memory.shape[0]), SLOTS, CLASSES)
    if mode == "F":
        selected = probabilities[0]
    else:
        selected = factual.clone()
        residue = positions.remainder(4)
        for value in range(4):
            assigned = active & residue.eq(value)
            selected[assigned] = probabilities[value][assigned]
    require(bool(selected.isfinite().all().item()), "non-finite readout posterior")
    candidate_probability, candidate_ids = selected.max(dim=-1)
    incumbent_probability = selected.gather(-1, base_ids.unsqueeze(-1)).squeeze(-1)
    gain = candidate_probability.clamp_min(1e-12).log() - incumbent_probability.clamp_min(1e-12).log()
    candidate_ids = torch.where(active, candidate_ids, base_ids)
    gain = torch.where(active, gain, torch.full_like(gain, -float("inf")))
    tau = F_TAU if mode == "F" else RPTCR_TAU
    changed = active & candidate_ids.ne(base_ids) & gain.gt(float(tau))
    output = torch.where(changed, candidate_ids, base_ids)
    target_counts = []
    residue = positions.remainder(4)
    for value in range(count):
        target_counts.append((active & residue.eq(value)).sum(dim=1, dtype=torch.int64))
    if mode == "F":
        target_counts = torch.zeros((1, int(memory.shape[0])), dtype=torch.int64, device=memory.device)
    else:
        target_counts = torch.stack(target_counts)
    return Readout(
        output_ids=output.detach(),
        candidate_ids=candidate_ids.detach(),
        gain=gain.detach(),
        active=active.detach(),
        masked_counts=masks.sum(dim=2, dtype=torch.int64).detach(),
        target_counts=target_counts.detach(),
        decoy_counts=decoys.sum(dim=2, dtype=torch.int64).detach(),
    )
