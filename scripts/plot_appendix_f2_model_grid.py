from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

from plot_style import set_paper_style  # noqa: E402


MODEL_META = {
    "qwen_qwen2_5_7b_instruct": ("Qwen-based", "qwen2.5-instruct"),
    "deepseek_ai_deepseek_r1_distill_qwen_7b": ("Qwen-based", "DeepSeek-R1-Distill-Qwen-7B"),
    "qwen_qwen3_8b": ("Qwen-based", "qwen3-8b"),
    "allenai_olmo_7b_0724_hf": ("Other Models", "OLMo-7B-0724"),
    "meta_llama_llama_3_1_8b_instruct": ("Other Models", "Llama-3.1-8B-Instruct"),
    "google_gemma_2_9b_it": ("Other Models", "Gemma-2-9B-it"),
    "mistralai_mistral_7b_v0_3": ("Other Models", "Mistral-7B-v0.3"),
    "openlm_research_open_llama_7b": ("Other Models", "OpenLLaMA-7B"),
    "01_ai_yi_6b": ("Other Models", "Yi-6B"),
    "thudm_chatglm3_6b": ("Other Models", "ChatGLM3-6B"),
}

MODEL_ORDER = list(MODEL_META.keys())


def load_layers(path: Path) -> Tuple[str, List[int]]:
    payload = json.loads(path.read_text())
    results = payload.get("results", {})
    entities = payload.get("entities", list(results.keys()))
    layers: List[int] = []
    for entity in entities:
        rec = results.get(entity)
        if not isinstance(rec, dict):
            continue
        layer = rec.get("top_layer")
        if layer is None:
            continue
        layers.append(int(layer))
    tag = path.stem.removeprefix("f2_popqa_popular_200_")
    return tag, layers


def sort_key(tag: str) -> Tuple[int, int, str]:
    group, label = MODEL_META.get(tag, ("Other Models", tag))
    group_rank = 0 if group == "Qwen-based" else 1
    try:
        idx = MODEL_ORDER.index(tag)
    except ValueError:
        idx = math.inf
    return group_rank, idx, label


def save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input F2 localization JSON files.")
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--ncols", type=int, default=3)
    args = parser.parse_args()

    records = []
    max_layer = 0
    for raw_path in args.inputs:
        path = Path(raw_path)
        tag, layers = load_layers(path)
        if not layers:
            continue
        max_layer = max(max_layer, max(layers))
        group, label = MODEL_META.get(tag, ("Other Models", tag.replace("_", "-")))
        early = sum(1 for layer in layers if layer <= 5)
        records.append(
            {
                "tag": tag,
                "group": group,
                "label": label,
                "layers": layers,
                "early": early,
                "n": len(layers),
            }
        )

    if not records:
        raise RuntimeError("No usable localization files supplied.")

    records.sort(key=lambda rec: sort_key(str(rec["tag"])))
    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(records) / ncols)

    import matplotlib.pyplot as plt

    set_paper_style()
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
        }
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.15 * ncols, 2.65 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    bins = np.arange(-0.5, max_layer + 1.5, 1.0)

    for idx, rec in enumerate(records):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        layers = np.asarray(rec["layers"], dtype=np.int64)
        ax.hist(layers, bins=bins, color="#4C78A8", edgecolor="white")
        ax.set_title(str(rec["label"]), fontsize=12.5, pad=5)
        early_pct = 100.0 * float(rec["early"]) / max(int(rec["n"]), 1)
        ax.text(
            0.98,
            0.95,
            f"L0-5: {early_pct:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.25,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.2},
        )
        if row == nrows - 1:
            ax.set_xlabel("Layer")
        if col == 0:
            ax.set_ylabel("Entities")
        ax.set_xlim(-0.5, max_layer + 0.5)
        ax.set_xticks(list(range(0, max_layer + 1, 5)))

    for idx in range(len(records), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.suptitle("Extended Cross-Model Localization Depth", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, Path(args.out_base))


if __name__ == "__main__":
    main()
