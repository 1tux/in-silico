from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from nnsight import LanguageModel

from activations import compute_metrics, compute_stability_score, get_activations, get_activations_at_pos, rank_neurons, z_score_normalize
from f2_neuron_localization import find_entity_token_pos, load_entities_from_file, prompts_with_entity_positions
from model_load import language_model_kwargs
from plot_style import set_paper_style
from prompts import entity_questions, load_generic_prompts


def last_word(entity: str) -> str:
    parts = [part for part in entity.strip().split() if part]
    if not parts:
        return entity.strip()
    return parts[-1]


def localize_entity(
    model,
    base_mean: torch.Tensor,
    base_std: torch.Tensor,
    entity_text: str,
    *,
    prompt_k: int,
    seed: int,
) -> Dict[str, float | int | str]:
    rng = random.Random(seed)
    prompts = entity_questions(entity_text)
    rng.shuffle(prompts)
    prompts = prompts[:prompt_k]
    prompts_with_pos = prompts_with_entity_positions(model.tokenizer, prompts, entity_text)
    if not prompts_with_pos:
        raise RuntimeError(f"No entity positions found for: {entity_text}")
    acts = get_activations_at_pos(model, prompts_with_pos)
    normalized_acts = z_score_normalize(acts, base_mean, base_std)
    stability_scores = compute_stability_score(normalized_acts)
    rankings = rank_neurons(stability_scores)
    top_layer, top_neuron = rankings[0].tolist()
    return {
        "top_layer": int(top_layer),
        "top_neuron": int(top_neuron),
        "n_prompts": len(prompts_with_pos),
        "surface": entity_text,
    }


def save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=220)


def plot_layer_hists(full_layers: List[int], token_layers: List[int], out_base: Path) -> None:
    import matplotlib.pyplot as plt

    set_paper_style()
    max_layer = max(max(full_layers), max(token_layers))
    bins = np.arange(-0.5, max_layer + 1.5, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    specs = [
        (axes[0], full_layers, "Full Entity"),
        (axes[1], token_layers, "Last Token Only"),
    ]
    for ax, layers, title in specs:
        ax.hist(layers, bins=bins, color="#4C78A8", edgecolor="white")
        early = 100.0 * sum(1 for layer in layers if layer <= 5) / max(len(layers), 1)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.text(
            0.98,
            0.95,
            f"L0-5: {early:.1f}%",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )
        ax.set_xticks(list(range(0, max_layer + 1, 5)))
    axes[0].set_ylabel("Entities")
    fig.suptitle("Localization Depth: Full Entity vs Last Token", y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, out_base)


def plot_layer_scatter(full_layers: List[int], token_layers: List[int], out_base: Path) -> None:
    import matplotlib.pyplot as plt

    set_paper_style()
    limit = max(max(full_layers), max(token_layers))
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.scatter(full_layers, token_layers, alpha=0.75, color="#F58518", s=22)
    ax.plot([0, limit], [0, limit], linestyle="--", color="black", alpha=0.5, linewidth=1.0)
    ax.set_xlabel("Full-entity layer")
    ax.set_ylabel("Last-token layer")
    ax.set_title("Per-entity layer comparison")
    ax.set_xlim(-0.5, limit + 0.5)
    ax.set_ylim(-0.5, limit + 0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, out_base)


def plot_match_bars(summary: Dict[str, float | int], out_base: Path) -> None:
    import matplotlib.pyplot as plt

    set_paper_style()
    labels = ["Exact neuron", "Same layer", "Both early"]
    values = [
        100.0 * float(summary["exact_match_rate"]),
        100.0 * float(summary["same_layer_rate"]),
        100.0 * float(summary["both_early_rate"]),
    ]
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Match Rate (%)")
    ax.set_title("Full Entity vs Last Token")
    for idx, value in enumerate(values):
        ax.text(idx, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--entities-file", default=str(Path(__file__).resolve().parents[1] / "configs" / "entities_popqa_popular_200_minq2.txt"))
    parser.add_argument("--generic-prompts", default=str(Path(__file__).resolve().parents[1] / "data" / "generic_prompts.txt"))
    parser.add_argument("--prompt-k", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of multi-token entities.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "diagnostics" / "last_token_localization"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entities = load_entities_from_file(args.entities_file)
    multi_token_entities = [entity for entity in entities if len(entity.strip().split()) > 1]
    if args.limit > 0:
        multi_token_entities = multi_token_entities[: args.limit]
    if not multi_token_entities:
        raise RuntimeError("No multi-token entities available for diagnostic.")

    model = LanguageModel(args.model, **language_model_kwargs(args.model))
    generic_prompts = load_generic_prompts(args.generic_prompts)
    baseline_acts = get_activations(model, generic_prompts)
    base_mean, base_std = compute_metrics(baseline_acts)

    rows = []
    for idx, entity in enumerate(multi_token_entities):
        full = localize_entity(model, base_mean, base_std, entity, prompt_k=args.prompt_k, seed=args.seed + idx)
        token = localize_entity(model, base_mean, base_std, last_word(entity), prompt_k=args.prompt_k, seed=args.seed + idx)
        rows.append(
            {
                "entity": entity,
                "last_token": last_word(entity),
                "full": full,
                "last_token_only": token,
                "exact_match": bool(full["top_layer"] == token["top_layer"] and full["top_neuron"] == token["top_neuron"]),
                "same_layer": bool(full["top_layer"] == token["top_layer"]),
                "both_early": bool(int(full["top_layer"]) <= 5 and int(token["top_layer"]) <= 5),
            }
        )

    full_layers = [int(row["full"]["top_layer"]) for row in rows]
    token_layers = [int(row["last_token_only"]["top_layer"]) for row in rows]

    summary = {
        "model": args.model,
        "n_multi_token_entities": len(rows),
        "exact_match_rate": sum(1 for row in rows if row["exact_match"]) / max(len(rows), 1),
        "same_layer_rate": sum(1 for row in rows if row["same_layer"]) / max(len(rows), 1),
        "both_early_rate": sum(1 for row in rows if row["both_early"]) / max(len(rows), 1),
        "full_early_rate": sum(1 for layer in full_layers if layer <= 5) / max(len(full_layers), 1),
        "last_token_early_rate": sum(1 for layer in token_layers if layer <= 5) / max(len(token_layers), 1),
        "examples": {
            row["entity"]: {
                "last_token": row["last_token"],
                "full": row["full"],
                "last_token_only": row["last_token_only"],
                "exact_match": row["exact_match"],
                "same_layer": row["same_layer"],
            }
            for row in rows
            if row["entity"] in {"Barack Obama", "Donald Trump", "New York City", "Muhammad Ali"}
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    header = ["entity", "last_token", "full_layer", "full_neuron", "token_layer", "token_neuron", "exact_match", "same_layer"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row["entity"],
                    row["last_token"],
                    str(row["full"]["top_layer"]),
                    str(row["full"]["top_neuron"]),
                    str(row["last_token_only"]["top_layer"]),
                    str(row["last_token_only"]["top_neuron"]),
                    str(int(bool(row["exact_match"]))),
                    str(int(bool(row["same_layer"]))),
                ]
            )
        )
    (out_dir / "rows.tsv").write_text("\n".join(lines) + "\n")

    plot_layer_hists(full_layers, token_layers, out_dir / "layer_hist_comparison")
    plot_layer_scatter(full_layers, token_layers, out_dir / "layer_scatter")
    plot_match_bars(summary, out_dir / "match_rates")


if __name__ == "__main__":
    main()
