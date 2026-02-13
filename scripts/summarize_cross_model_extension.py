from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

from plot_style import set_paper_style  # noqa: E402


MODEL_ORDER = [
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-7B-0724-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-v0.3",
    "openlm-research/open_llama_7b",
]


MODEL_LABEL = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Inst",
    "allenai/OLMo-7B-0724-hf": "OLMo-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst",
    "google/gemma-2-9b-it": "Gemma-2-9B-it",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B-v0.3",
    "openlm-research/open_llama_7b": "OpenLLaMA-7B",
}


LOCALIZATION_RESULT_FILE = {
    "Qwen/Qwen2.5-7B-Instruct": "results/f2_popqa_popular_200_qwen2_5_7b_instruct.json",
    "allenai/OLMo-7B-0724-hf": "results/f2_popqa_popular_200_olmo_7b_0724.json",
    "meta-llama/Llama-3.1-8B-Instruct": "results/f2_popqa_popular_200_llama3_1_8b_instruct.json",
    "google/gemma-2-9b-it": "results/f2_popqa_popular_200_gemma2_9b_it.json",
    "mistralai/Mistral-7B-v0.3": "results/f2_popqa_popular_200_mistral_7b_v03.json",
    "openlm-research/open_llama_7b": "results/f2_popqa_popular_200_open_llama_7b.json",
}


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def sanitize_model(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def get_variant_file(results_dir: Path, model: str) -> Path:
    model_tag = sanitize_model(model)
    candidates = sorted(results_dir.glob(f"variant_robustness_{model_tag}.json"))
    if not candidates:
        raise FileNotFoundError(f"Missing variant robustness JSON for model: {model}")
    return candidates[0]


def get_f4_file(results_dir: Path, model: str) -> Path:
    model_tag = sanitize_model(model).lower()
    # f4 files are normalized to short tags; use glob on model tail tokens.
    candidates = sorted(glob.glob(str(results_dir / "f4_activation_causality_generalize_*_results.json")))
    for cand in candidates:
        payload = read_json(Path(cand))
        if payload.get("model") == model:
            return Path(cand)
    raise FileNotFoundError(f"Missing f4 generalization results for model: {model}")


def write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = parser.parse_args()

    repo = Path(args.repo_root)
    results_dir = repo / "results"
    figures_dir = repo / "figures"

    # Localization depth summary from the full PopQA-200 run.
    loc_rows: Dict[str, Dict] = {}
    for model in MODEL_ORDER:
        path = repo / LOCALIZATION_RESULT_FILE[model]
        payload = read_json(path)
        entities = payload["entities"]
        rows = [payload["results"][e] for e in entities if e in payload["results"]]
        layers = np.asarray([int(r["top_layer"]) for r in rows], dtype=np.int64)
        loc_rows[model] = {
            "n": int(len(rows)),
            "pct_l6": float(np.mean(layers <= 6)),
            "median_layer": float(np.median(layers)),
            "pct_l3": float(np.mean(layers <= 3)),
        }

    # Causal validation summary from per-model F4 runs on found cells.
    f4_rows: Dict[str, Dict] = {}
    for model in MODEL_ORDER:
        path = get_f4_file(results_dir, model)
        payload = read_json(path)
        rel = payload.get("means", {}).get("relprob", [None, None, None, None, None])
        succ = payload.get("success_summary", {})
        n_total = int(succ.get("n_trustworthy_entities", 0))
        k_succ = int(succ.get("k_success_topk", 0))
        f4_rows[model] = {
            "n_entities": int(payload.get("n_entities_used", 0)),
            "n_examples": int(payload.get("n_examples", 0)),
            "rel_noinj": float(rel[1]),
            "rel_mean": float(rel[2]),
            "rel_correct": float(rel[3]),
            "rel_wrong": float(rel[4]),
            "delta_correct_minus_wrong": float(rel[3] - rel[4]),
            "success_rate": (float(k_succ) / max(n_total, 1)) if n_total > 0 else 0.0,
            "k_success": k_succ,
            "n_total": n_total,
            "k_topk_needed": int(succ.get("k_topk_needed", 0)),
        }

    # Variant robustness summary.
    variant_rows: Dict[str, Dict] = {}
    for model in MODEL_ORDER:
        path = get_variant_file(results_dir, model)
        payload = read_json(path)
        groups = payload["groups"]
        row: Dict[str, float | int] = {}
        total_k = 0
        total_n = 0
        for key in ["person_typos_obama", "acronym_fbi", "multilingual_paris"]:
            rec = groups[key]
            k = int(rec["k_match"])
            n = int(rec["n_variants"])
            row[f"{key}_k"] = k
            row[f"{key}_n"] = n
            row[f"{key}_rate"] = float(k) / max(n, 1)
            total_k += k
            total_n += n
            row[f"{key}_cell"] = f"L{rec['canonical_layer']}-N{rec['canonical_neuron']}"
        row["total_k"] = total_k
        row["total_n"] = total_n
        row["total_rate"] = float(total_k) / max(total_n, 1)
        variant_rows[model] = row

    combined_rows: List[List[str]] = []
    for model in MODEL_ORDER:
        loc = loc_rows[model]
        f4 = f4_rows[model]
        var = variant_rows[model]
        combined_rows.append(
            [
                MODEL_LABEL[model],
                str(loc["n"]),
                f"{100.0 * loc['pct_l6']:.1f}",
                f"{loc['median_layer']:.1f}",
                str(f4["n_entities"]),
                str(f4["n_examples"]),
                f"{f4['rel_noinj']:.3f}",
                f"{f4['rel_mean']:.3f}",
                f"{f4['rel_correct']:.3f}",
                f"{f4['rel_wrong']:.3f}",
                f"{f4['delta_correct_minus_wrong']:.3f}",
                f"{100.0 * f4['success_rate']:.1f}",
                f"{f4['k_success']}/{f4['n_total']}",
                f"{var['total_k']}/{var['total_n']}",
                f"{100.0 * var['total_rate']:.1f}",
            ]
        )

    write_tsv(
        results_dir / "cross_model_extension_summary.tsv",
        [
            "model",
            "f2_n",
            "f2_pct_layer_le_6",
            "f2_median_layer",
            "f4_n_entities",
            "f4_n_examples",
            "f4_rel_noinj",
            "f4_rel_mean",
            "f4_rel_correct",
            "f4_rel_wrong",
            "f4_delta_correct_minus_wrong",
            "f4_success_pct",
            "f4_success_k_over_n",
            "variant_match_k_over_n",
            "variant_match_pct",
        ],
        combined_rows,
    )

    variant_table_rows: List[List[str]] = []
    for model in MODEL_ORDER:
        var = variant_rows[model]
        variant_table_rows.append(
            [
                MODEL_LABEL[model],
                var["person_typos_obama_cell"],
                f"{var['person_typos_obama_k']}/{var['person_typos_obama_n']}",
                var["acronym_fbi_cell"],
                f"{var['acronym_fbi_k']}/{var['acronym_fbi_n']}",
                var["multilingual_paris_cell"],
                f"{var['multilingual_paris_k']}/{var['multilingual_paris_n']}",
            ]
        )
    write_tsv(
        results_dir / "cross_model_variant_cells.tsv",
        [
            "model",
            "obama_cell",
            "obama_matches",
            "fbi_cell",
            "fbi_matches",
            "paris_cell",
            "paris_matches",
        ],
        variant_table_rows,
    )

    # Figure A: causal separation + success rate summary.
    import matplotlib.pyplot as plt

    set_paper_style()
    labels = [MODEL_LABEL[m] for m in MODEL_ORDER]
    x = np.arange(len(labels), dtype=float)
    delta = np.asarray([f4_rows[m]["delta_correct_minus_wrong"] for m in MODEL_ORDER], dtype=float)
    success_pct = 100.0 * np.asarray([f4_rows[m]["success_rate"] for m in MODEL_ORDER], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax1.bar(x, delta, color="#F58518")
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_ylabel(r"$\Delta$ RelProb (Correct $-$ Wrong)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Causal separation")

    ax2.bar(x, success_pct, color="#4C78A8")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_title("Entity-level success")

    fig.tight_layout()
    fig.savefig(figures_dir / "fx_cross_model_f4_delta_success.pdf")
    fig.savefig(figures_dir / "fx_cross_model_f4_delta_success.png", dpi=220)
    plt.close(fig)

    # Figure B: Variant-match heatmap.
    probes = ["person_typos_obama", "acronym_fbi", "multilingual_paris"]
    probe_labels = ["Obama typos", "FBI acronym", "Paris multilingual"]
    mat = np.asarray([[variant_rows[m][f"{p}_rate"] for p in probes] for m in MODEL_ORDER], dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(probes)))
    ax.set_xticklabels(probe_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Top-cell robustness across surface forms")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Match rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "fx_cross_model_variant_heatmap.pdf")
    fig.savefig(figures_dir / "fx_cross_model_variant_heatmap.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
