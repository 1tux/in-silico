from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


DEFAULT_ENTITIES = [
    "Barack Obama",
    "Donald Trump",
    "Paris",
    "White House",
    "European Union",
]


def load_result(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def dominance(record: Dict) -> float:
    top1 = float(record.get("top1", 0.0))
    topk = float(record.get("topk_mean", 0.0))
    return top1 / max(topk, 1e-12)


def summarize_model(payload: Dict, layer_cap: int, dom_threshold: float) -> Dict:
    entities = payload["entities"]
    results = payload["results"]
    rows = [results[e] for e in entities if e in results]
    layers = [int(r["top_layer"]) for r in rows]
    doms = [dominance(r) for r in rows]
    n = len(rows)
    k_early = sum(1 for layer in layers if layer <= layer_cap)
    k_dom = sum(1 for dom in doms if dom >= dom_threshold)
    k_both = sum(1 for layer, dom in zip(layers, doms) if layer <= layer_cap and dom >= dom_threshold)
    return {
        "n_entities": n,
        "k_layer_le_cap": k_early,
        "pct_layer_le_cap": (k_early / n) if n else 0.0,
        "k_dom_ge_threshold": k_dom,
        "pct_dom_ge_threshold": (k_dom / n) if n else 0.0,
        "k_joint": k_both,
        "pct_joint": (k_both / n) if n else 0.0,
        "median_layer": statistics.median(layers) if layers else None,
        "median_dominance": statistics.median(doms) if doms else None,
    }


def collect_cells(payload: Dict, entities: List[str]) -> Dict[str, Dict[str, float | int | None]]:
    results = payload["results"]
    out = {}
    for entity in entities:
        rec = results.get(entity)
        if rec is None:
            out[entity] = {"layer": None, "neuron": None, "dominance": None}
            continue
        out[entity] = {
            "layer": int(rec["top_layer"]),
            "neuron": int(rec["top_neuron"]),
            "dominance": float(dominance(rec)),
        }
    return out


def short_model_name(path: Path) -> str:
    stem = path.stem
    prefix = "f2_popqa_popular_200_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem.replace("_", "-")


def write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input f2 localization JSON files (one per model).",
    )
    parser.add_argument("--layer-cap", type=int, default=6, help="Early-layer cap for summary statistics.")
    parser.add_argument(
        "--dom-threshold",
        type=float,
        default=10.0,
        help="Dominance threshold for localization strength summary.",
    )
    parser.add_argument(
        "--entities",
        nargs="*",
        default=DEFAULT_ENTITIES,
        help="Entities to include in the cross-model neuron table.",
    )
    parser.add_argument(
        "--out-json",
        default="results/f2_cross_model_localization_summary.json",
    )
    parser.add_argument(
        "--out-summary-tsv",
        default="results/f2_cross_model_localization_summary.tsv",
    )
    parser.add_argument(
        "--out-cells-tsv",
        default="results/f2_cross_model_reference_cells.tsv",
    )
    args = parser.parse_args()

    model_payloads = {}
    for input_path in args.inputs:
        path = Path(input_path)
        model_name = short_model_name(path)
        model_payloads[model_name] = load_result(path)

    summary = {}
    cells = {}
    for model_name, payload in model_payloads.items():
        summary[model_name] = summarize_model(payload, args.layer_cap, args.dom_threshold)
        cells[model_name] = collect_cells(payload, args.entities)

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(
        json.dumps(
            {
                "layer_cap": args.layer_cap,
                "dom_threshold": args.dom_threshold,
                "entities": args.entities,
                "summary": summary,
                "cells": cells,
            },
            indent=2,
        )
    )

    summary_rows = []
    for model_name in sorted(summary):
        rec = summary[model_name]
        summary_rows.append(
            [
                model_name,
                str(rec["n_entities"]),
                str(rec["k_layer_le_cap"]),
                f"{rec['pct_layer_le_cap']:.3f}",
                str(rec["k_dom_ge_threshold"]),
                f"{rec['pct_dom_ge_threshold']:.3f}",
                str(rec["k_joint"]),
                f"{rec['pct_joint']:.3f}",
                f"{rec['median_layer']:.1f}" if rec["median_layer"] is not None else "NA",
                f"{rec['median_dominance']:.2f}" if rec["median_dominance"] is not None else "NA",
            ]
        )
    write_tsv(
        Path(args.out_summary_tsv),
        [
            "model",
            "n_entities",
            f"k_layer_le_{args.layer_cap}",
            f"pct_layer_le_{args.layer_cap}",
            f"k_dom_ge_{args.dom_threshold:g}",
            f"pct_dom_ge_{args.dom_threshold:g}",
            "k_joint",
            "pct_joint",
            "median_layer",
            "median_dominance",
        ],
        summary_rows,
    )

    cell_rows = []
    for model_name in sorted(cells):
        for entity in args.entities:
            row = cells[model_name][entity]
            cell_rows.append(
                [
                    model_name,
                    entity,
                    "NA" if row["layer"] is None else str(row["layer"]),
                    "NA" if row["neuron"] is None else str(row["neuron"]),
                    "NA" if row["dominance"] is None else f"{row['dominance']:.2f}",
                ]
            )
    write_tsv(
        Path(args.out_cells_tsv),
        ["model", "entity", "layer", "neuron", "dominance"],
        cell_rows,
    )


if __name__ == "__main__":
    main()
