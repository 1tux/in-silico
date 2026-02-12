from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import nnsight
import torch
import tqdm


def get_activations_at_pos(model, prompts_with_pos: Iterable[Tuple[str, int]]) -> torch.Tensor:
    """Return activations [num_prompts, num_layers, hidden] on CPU."""
    activations: List[List[torch.Tensor]] = [[] for _ in range(len(model.model.layers))]

    for prompt, position in tqdm.tqdm(list(prompts_with_pos), desc="Tracing prompts"):
        with torch.no_grad():
            with model.trace(prompt):
                activations_for_prompt = nnsight.list().save()
                for layer_idx, layer in enumerate(model.model.layers):
                    activations_for_prompt.append(
                        layer.mlp.down_proj.input[0][position].cpu().save()
                    )

            for layer_idx, layer_activations in enumerate(activations_for_prompt):
                activations[layer_idx].append(layer_activations)

        torch.cuda.empty_cache()

    return torch.stack([torch.stack(acts) for acts in activations], dim=1)


def get_activations(model, prompts: Sequence[str]) -> torch.Tensor:
    return get_activations_at_pos(model, ((p, -1) for p in prompts))


def compute_metrics(acts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return acts.mean(dim=0), acts.std(dim=0, unbiased=False)


def compute_stability_score(acts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean, std = compute_metrics(acts)
    score = mean ** 2 / (std + eps)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def z_score_normalize(
    acts: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    normalized = (acts - mean) / (std + eps)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def rank_neurons(scores: torch.Tensor) -> torch.Tensor:
    scores_ = scores.view(-1)
    return torch.stack(
        torch.unravel_index(scores_.topk(k=len(scores_)).indices, scores.shape), dim=1
    )
