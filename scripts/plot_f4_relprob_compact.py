from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

from plot_style import set_paper_style  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        default=str(
            Path(__file__).resolve().parents[1]
            / "figures"
            / "f4_activation_causality_popular200_trustworthy_meaninit_topk5_alphasearch_poplist_results.json"
        ),
    )
    parser.add_argument(
        "--output-base",
        default=str(
            Path(__file__).resolve().parents[1]
            / "figures"
            / "f4_activation_causality_popular200_trustworthy_meaninit_topk5_alphasearch_poplist_relprob"
        ),
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.results_json).read_text())
    per_entity = payload.get("per_entity", {})
    if not per_entity:
        raise RuntimeError("Missing per-entity records in results JSON.")

    rows = list(per_entity.values())
    rel_means_full = payload.get("means", {}).get("relprob", None)
    if not isinstance(rel_means_full, list) or len(rel_means_full) < 5:
        raise RuntimeError("Missing aggregated RelProb means in results JSON.")

    # Main-text compact view: drop "No Injection" because it is effectively identical to
    # mean-entity initialization in this run and clutters the figure.
    labels = ["Entity Present", "Mean Entity", "Correct Cell", "Wrong Cell"]
    means = [float(rel_means_full[0]), float(rel_means_full[2]), float(rel_means_full[3]), float(rel_means_full[4])]
    import matplotlib.pyplot as plt

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    colors = ["#6C7A89", "#9C755F", "#F58518", "#54A24B"]
    ax.bar(labels, means, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Relative Answer Probability")
    ax.set_title(f"Activation Causality (Normalized)\nModel: {payload.get('model', 'Unknown')}")
    fig.tight_layout()

    out_base = Path(args.output_base)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)

    note = {
        "n_entities": int(len(rows)),
        "mean_no_injection_relprob": float(rel_means_full[1]),
        "mean_mean_entity_relprob": float(rel_means_full[2]),
        "mean_correct_relprob": float(rel_means_full[3]),
        "mean_wrong_relprob": float(rel_means_full[4]),
    }
    out_base.with_suffix(".meta.json").write_text(json.dumps(note, indent=2) + "\n")


if __name__ == "__main__":
    main()
