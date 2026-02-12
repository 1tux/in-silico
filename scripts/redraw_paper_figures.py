from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Keep Matplotlib fully headless and avoid writing to $HOME.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

from plot_style import set_paper_style  # noqa: E402


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def plot_layer_hist_from_localization(localization_path: Path, out_base: Path) -> None:
    payload = _load_json(localization_path)
    results = payload.get("results", {})
    entities = payload.get("entities", list(results.keys()))

    layers: List[int] = []
    for ent in entities:
        rec = results.get(ent)
        if not isinstance(rec, dict):
            continue
        layer = rec.get("top_layer", None)
        if layer is None:
            continue
        try:
            layers.append(int(layer))
        except Exception:
            continue

    if not layers:
        raise RuntimeError(f"No layer ids found in {localization_path}")

    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = list(range(min(layers), max(layers) + 2))
    ax.hist(layers, bins=bins, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Layer (Top Neuron)")
    ax.set_ylabel("Entity Count")
    ax.set_title("Top-Neuron Layer Distribution")
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def _extract_f2_stats(localization_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = _load_json(localization_path)
    results = payload.get("results", {})
    entities = payload.get("entities", list(results.keys()))

    dominance: List[float] = []
    topk_rel: List[float] = []
    rand_rel: List[float] = []

    for ent in entities:
        rec = results.get(ent)
        if not isinstance(rec, dict):
            continue
        top1 = float(rec.get("top1", 0.0))
        topk_mean = float(rec.get("topk_mean", 0.0))
        randk_mean = float(rec.get("randk_mean", 0.0))
        if not math.isfinite(top1) or top1 <= 0:
            continue
        dominance.append(top1 / max(topk_mean, 1e-12))
        topk_rel.append(topk_mean / max(top1, 1e-12))
        rand_rel.append(randk_mean / max(top1, 1e-12))

    if not dominance:
        raise RuntimeError(f"No usable dominance values found in {localization_path}")

    return np.asarray(dominance), np.asarray(topk_rel), np.asarray(rand_rel)


def plot_f2_dominance_hist(localization_path: Path, out_base: Path) -> None:
    dominance, _topk_rel, _rand_rel = _extract_f2_stats(localization_path)

    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.hist(dominance, bins=20, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Top-1 / Mean Top-5 (Same Layer)")
    ax.set_ylabel("Entity Count")
    ax.set_title("Neuron Dominance")
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def plot_f2_topk_comparison(localization_path: Path, out_base: Path) -> None:
    _dominance, topk_rel, rand_rel = _extract_f2_stats(localization_path)
    top1_rel = np.ones_like(topk_rel)

    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    means = [float(np.mean(top1_rel)), float(np.mean(topk_rel)), float(np.mean(rand_rel))]
    errs = [
        float(np.std(top1_rel) / np.sqrt(len(top1_rel))),
        float(np.std(topk_rel) / np.sqrt(len(topk_rel))),
        float(np.std(rand_rel) / np.sqrt(len(rand_rel))),
    ]
    labels = ["Top-1", "Top-5", "Random-5"]
    ax.bar(labels, means, yerr=errs, color=["#4C78A8", "#F58518", "#54A24B"], capsize=3)
    ax.set_ylabel("Mean Relative Score (Top-1=1.0)")
    ax.set_title("Top-k vs Random Controls")
    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)


def plot_unlearning_curve_from_results(results_path: Path, out_base: Path) -> None:
    payload = _load_json(results_path)
    x = np.asarray(payload["x_steps"], dtype=float)
    target_mean = np.asarray(payload.get("target_mean", []), dtype=float)
    target_err = np.asarray(payload.get("target_stderr", []), dtype=float)
    control_mean = np.asarray(payload.get("control_mean", []), dtype=float)
    control_err = np.asarray(payload.get("control_stderr", []), dtype=float)

    if target_mean.size == 0:
        raise RuntimeError(f"Missing target curve in {results_path}")

    entity = payload.get("entity", "Target")
    control = payload.get("control", None)
    normalize = payload.get("normalize", "unknown")
    layer = payload.get("layer", None)
    neuron = payload.get("neuron", None)

    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    ax.plot(x, target_mean, label=f"Target ({entity})", color="crimson")
    if target_err.size == target_mean.size:
        ax.fill_between(x, target_mean - target_err, target_mean + target_err, color="crimson", alpha=0.2)

    if control is not None and control_mean.size == target_mean.size:
        ax.plot(x, control_mean, label=f"Control ({control})", color="royalblue")
        if control_err.size == control_mean.size:
            ax.fill_between(x, control_mean - control_err, control_mean + control_err, color="royalblue", alpha=0.2)

    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="Original Model")
    if normalize == "unknown":
        ax.axhline(0.0, color="gray", ls=":", alpha=0.7, label="Unknown Entity Baseline")

    if layer is not None and neuron is not None:
        ax.set_xlabel(f"Multiplier for Neuron {neuron} (Layer {layer})")
    else:
        ax.set_xlabel("Ablation Multiplier")
    ax.set_ylabel("Relative Knowledge Score")
    ax.set_title("Selective Unlearning via Neuron Ablation")
    ax.legend(frameon=True)
    ax.invert_xaxis()
    fig.tight_layout()

    _save(fig, out_base)
    plt.close(fig)


def plot_injection_curve_from_results(results_path: Path, out_base: Path) -> None:
    payload = _load_json(results_path)
    multipliers = np.asarray(payload["multipliers"], dtype=float)
    probs = np.asarray(payload["probs"], dtype=float)

    entity = payload.get("entity", "")
    layer = payload.get("layer", None)
    neuron = payload.get("neuron", None)

    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(multipliers, probs, marker="o", color="#4C78A8")
    ax.set_xlabel("Injected Neuron Value")
    ax.set_ylabel("Answer Token Probability")
    if layer is not None and neuron is not None and entity:
        ax.set_title(f"Injection: {entity} (L{layer}-N{neuron})")
    else:
        ax.set_title("Injection Dose-Response")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    _save(fig, out_base)
    plt.close(fig)


def plot_edit_vs_preserve_from_latent_results(results_path: Path, out_base: Path, meta_out: Path | None = None) -> None:
    payload = _load_json(results_path)
    attack_eval = payload.get("attack_eval", {})
    preserve_eval = payload.get("preserve_eval", {})

    attack_keys = list(attack_eval.keys())
    preserve_keys = list(preserve_eval.keys())

    attack_base = [float(attack_eval[k]["base_prob"]) for k in attack_keys]
    attack_steered = [float(attack_eval[k]["steered_prob"]) for k in attack_keys]
    preserve_ratios = [float(preserve_eval[k]["ratio"]) for k in preserve_keys]

    import matplotlib.pyplot as plt

    set_paper_style()
    fig = plt.figure(figsize=(6.8, 3.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(attack_keys))
    width = 0.38
    ax1.bar(x - width / 2, attack_base, width=width, label="Base", color="#B0C4DE", edgecolor="none")
    ax1.bar(x + width / 2, attack_steered, width=width, label="Steered", color="#2E598B", edgecolor="none")
    ax1.set_yscale("log")
    ax1.set_ylabel("P(target token)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"A{i+1}" for i in range(len(attack_keys))])
    ax1.set_title("Attack prompts (target relation)")
    ax1.legend(frameon=True, ncol=2, loc="upper left")

    ax2 = fig.add_subplot(gs[1])
    x2 = np.arange(len(preserve_keys))
    ax2.bar(x2, preserve_ratios, color="#4C78A8", edgecolor="none")
    ax2.axhline(1.0, color="black", ls="--", alpha=0.6, linewidth=1.0)
    ax2.set_ylabel("Steered/Base ratio")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"P{i+1}" for i in range(len(preserve_keys))])
    ymax = max(2.6, float(np.max(preserve_ratios)) * 1.15) if preserve_ratios else 2.6
    ax2.set_ylim(0.0, ymax)
    ax2.set_title("Preservation prompts (non-target facts)")

    fig.tight_layout()
    _save(fig, out_base)
    plt.close(fig)

    if meta_out is not None:
        meta = {
            "attack_prompt_keys": attack_keys,
            "preserve_prompt_keys": preserve_keys,
            "attack_base_mean": float(np.mean(attack_base)) if attack_base else None,
            "attack_steered_mean": float(np.mean(attack_steered)) if attack_steered else None,
            "preserve_ratio_median": float(np.median(preserve_ratios)) if preserve_ratios else None,
            "preserve_ratio_mean": float(np.mean(preserve_ratios)) if preserve_ratios else None,
        }
        meta_out.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    figures = repo / "figures"
    results = repo / "results"

    # Derived solely from JSON artifacts to avoid re-running GPU-heavy jobs.
    plot_layer_hist_from_localization(
        localization_path=results / "f2_popqa_popular_200_minq2.json",
        out_base=figures / "f1_layer_hist_popular",
    )
    plot_f2_dominance_hist(
        localization_path=results / "f2_popqa_popular_200_minq2.json",
        out_base=figures / "f2_dominance_hist_popular",
    )
    plot_f2_topk_comparison(
        localization_path=results / "f2_popqa_popular_200_minq2.json",
        out_base=figures / "f2_topk_comparison_popular",
    )
    plot_unlearning_curve_from_results(
        results_path=figures / "f6_unlearning_obama_trump_results.json",
        out_base=figures / "f6_unlearning_obama_trump",
    )
    plot_injection_curve_from_results(
        results_path=figures / "f5_injection_obama_anchor.json",
        out_base=figures / "f5_injection_obama_anchor",
    )
    plot_edit_vs_preserve_from_latent_results(
        results_path=figures / "f7_latent_steering_obama_wife.json",
        out_base=figures / "f7_edit_vs_preserve",
        meta_out=figures / "f7_edit_vs_preserve_meta.json",
    )


if __name__ == "__main__":
    main()
