from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from nnsight import LanguageModel

from activations import (
    compute_metrics,
    compute_stability_score,
    get_activations,
    rank_neurons,
    z_score_normalize,
)
from plot_style import set_paper_style
from prompts import entity_questions, load_generic_prompts


def _safe_bidi(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def load_entities(path: Path) -> List[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def get_baseline_stats(model, generic_prompts: List[str], cache_path: Path | None):
    if cache_path and cache_path.exists():
        data = torch.load(cache_path)
        return data["mean"], data["std"]

    baseline_acts = get_activations(model, generic_prompts)
    base_mean, base_std = compute_metrics(baseline_acts)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": base_mean, "std": base_std}, cache_path)

    return base_mean, base_std


def get_top_neurons_for_entity(
    model,
    entity: str,
    base_mean: torch.Tensor,
    base_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    questions = entity_questions(entity)
    acts = get_activations(model, questions)
    normalized_acts = z_score_normalize(acts, base_mean, base_std)
    stability_scores = compute_stability_score(normalized_acts)
    rankings = rank_neurons(stability_scores)
    return rankings, stability_scores


def plot_entity_ranking(entity: str, scores: torch.Tensor, best_neurons: torch.Tensor, out_path: Path):
    import matplotlib.pyplot as plt

    set_paper_style()

    best_neurons = best_neurons[:6]
    labels = [f"L{n[0]}-{n[1]}" for n in best_neurons]
    raw_scores = np.array([scores[n[0], n[1]].item() for n in best_neurons])

    top_score = raw_scores.max()
    rel_scores = raw_scores / top_score
    top_idx = int(np.argmax(rel_scores))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y_pos = np.arange(len(best_neurons))

    base_color = "#B0C4DE"
    highlight_color = "#2E598B"
    colors = [highlight_color if i == top_idx else base_color for i in range(len(best_neurons))]

    bars = ax.barh(y_pos, rel_scores, color=colors, height=0.7, edgecolor="none")
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Stability Relative to Top Neuron")

    for i, (bar, value) in enumerate(zip(bars, rel_scores)):
        is_top = (i == top_idx)
        if value > 0.85:
            x_pos = bar.get_width() - 0.02
            ha = "right"
            color = "white" if is_top else "#333333"
        else:
            x_pos = bar.get_width() + 0.02
            ha = "left"
            color = highlight_color if is_top else "#555555"

        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=7,
            color=color,
            fontweight="bold" if is_top else "normal",
        )

    ax.set_title(_safe_bidi(entity), pad=6, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_variant_grid(
    entities: List[str],
    per_entity_scores: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    out_path: Path,
    *,
    ncols: int | None = None,
    titles: List[str] | None = None,
):
    import matplotlib.pyplot as plt

    set_paper_style()
    n = len(entities)
    if n == 0:
        raise ValueError("entities must be non-empty")

    if ncols is None:
        ncols = n
    ncols = max(1, min(int(ncols), n))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    if titles is not None and len(titles) != len(entities):
        raise ValueError("titles must be None or the same length as entities")

    for idx, (ax, ent) in enumerate(zip(axes, entities)):
        rankings, stability_scores = per_entity_scores[ent]
        best_neurons = rankings[:6]
        labels = [f"L{n[0]}-{n[1]}" for n in best_neurons]
        raw_scores = np.array([stability_scores[n[0], n[1]].item() for n in best_neurons])

        top_score = raw_scores.max()
        rel_scores = raw_scores / top_score
        top_idx = int(np.argmax(rel_scores))

        y_pos = np.arange(len(best_neurons))
        base_color = "#B0C4DE"
        highlight_color = "#2E598B"
        colors = [highlight_color if i == top_idx else base_color for i in range(len(best_neurons))]

        bars = ax.barh(y_pos, rel_scores, color=colors, height=0.7, edgecolor="none")
        ax.invert_yaxis()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Rel. Stability")

        for i, (bar, value) in enumerate(zip(bars, rel_scores)):
            is_top = (i == top_idx)
            if value > 0.85:
                x_pos = bar.get_width() - 0.03
                ha = "right"
                color = "white" if is_top else "#333333"
            else:
                x_pos = bar.get_width() + 0.03
                ha = "left"
                color = highlight_color if is_top else "#555555"

            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha=ha,
                fontsize=6.5,
                color=color,
                fontweight="bold" if is_top else "normal",
            )

        title = titles[idx] if titles is not None else ent
        ax.set_title(_safe_bidi(title), pad=4, fontweight="bold")

    for ax in axes[len(entities) :]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_layer_hist(layer_ids: List[int], out_path: Path):
    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(layer_ids, bins=range(min(layer_ids), max(layer_ids) + 2), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Layer (Top Neuron)")
    ax.set_ylabel("Entity Count")
    ax.set_title("Top-Neuron Layer Distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--entities", default=str(Path(__file__).resolve().parents[1] / "configs" / "entities_default.txt"))
    parser.add_argument("--generic-prompts", default=str(Path(__file__).resolve().parents[1] / "data" / "generic_prompts.txt"))
    parser.add_argument("--cache-baseline", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "f1_f3_localization.json"))
    parser.add_argument("--fig-dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    parser.add_argument("--plot-entities", nargs="*", default=["Donald Trump", "Jennifer Aniston", "FBI", "Paris"])
    parser.add_argument("--variant-entities", nargs="*", default=["Barack Obama", "Obaama", "Brock Obma", "Bark Obamna"])
    parser.add_argument(
        "--acronym-entities",
        nargs="*",
        default=["Federal Bureau of Investigation", "FBI"],
        help="Two surface forms that should refer to the same entity (acronym robustness probe).",
    )
    parser.add_argument(
        "--multilingual-entities",
        nargs="*",
        # Keep defaults ASCII-only; pass literal scripts via CLI if desired.
        default=["Paris", "\u05e4\u05e8\u05d9\u05d6", "\u5df4\u9ece", "\u0628\u0627\u0631\u064a\u0633"],
        help="Multiple surface forms in different scripts (multilingual robustness probe).",
    )
    parser.add_argument(
        "--multilingual-labels",
        nargs="*",
        default=["Paris (Latin)", "Paris (Hebrew)", "Paris (Chinese)", "Paris (Arabic)"],
        help="Display labels for --multilingual-entities (use ASCII to avoid font issues).",
    )
    args = parser.parse_args()

    entities = load_entities(Path(args.entities))
    generic_prompts = load_generic_prompts(args.generic_prompts)

    model = LanguageModel(args.model, device_map="auto")

    cache_path = None
    if args.cache_baseline:
        model_key = args.model.replace("/", "_").replace(":", "_")
        cache_path = Path(__file__).resolve().parents[1] / "data" / f"baseline_{model_key}.pt"

    base_mean, base_std = get_baseline_stats(model, generic_prompts, cache_path)

    results = {}
    layer_ids = []

    per_entity_scores: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    needed = set(entities) | set(args.plot_entities) | set(args.variant_entities) | set(args.acronym_entities) | set(args.multilingual_entities)

    for ent in sorted(needed):
        rankings, stability_scores = get_top_neurons_for_entity(model, ent, base_mean, base_std)
        top = rankings[0].tolist()
        if ent in entities:
            layer_ids.append(int(top[0]))
        results[ent] = {
            "top_layer": int(top[0]),
            "top_neuron": int(top[1]),
        }

        if ent in args.plot_entities or ent in args.variant_entities or ent in args.acronym_entities or ent in args.multilingual_entities:
            per_entity_scores[ent] = (rankings, stability_scores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    fig_dir = Path(args.fig_dir)

    for ent in args.plot_entities:
        if ent in per_entity_scores:
            rankings, stability_scores = per_entity_scores[ent]
            plot_entity_ranking(ent, stability_scores, rankings, fig_dir / f"f1_top_neuron_{ent.replace(' ', '_')}" )

    if args.variant_entities and all(e in per_entity_scores for e in args.variant_entities):
        plot_variant_grid(args.variant_entities, per_entity_scores, fig_dir / "f3_variants_grid_2x2", ncols=2)

    if len(args.acronym_entities) >= 2 and all(e in per_entity_scores for e in args.acronym_entities[:2]):
        plot_variant_grid(args.acronym_entities[:2], per_entity_scores, fig_dir / "f3_acronym_grid", ncols=2)

    if len(args.multilingual_entities) >= 2 and all(e in per_entity_scores for e in args.multilingual_entities):
        titles = None
        if len(args.multilingual_labels) == len(args.multilingual_entities):
            titles = args.multilingual_labels
        plot_variant_grid(
            args.multilingual_entities,
            per_entity_scores,
            fig_dir / "f3_multilingual_grid_2x2",
            ncols=2,
            titles=titles,
        )

    if layer_ids:
        plot_layer_hist(layer_ids, fig_dir / "f1_layer_hist")


if __name__ == "__main__":
    main()
