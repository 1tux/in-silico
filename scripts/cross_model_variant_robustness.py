from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from nnsight import LanguageModel

from activations import (
    compute_metrics,
    compute_stability_score,
    get_activations,
    rank_neurons,
    z_score_normalize,
)
from prompts import entity_questions, load_generic_prompts


DEFAULT_GROUPS = {
    "person_typos_obama": [
        "Barack Obama",
        "Obaama",
        "Brock Obma",
        "Bark Obamna",
    ],
    "acronym_fbi": [
        "Federal Bureau of Investigation",
        "FBI",
    ],
    "multilingual_paris": [
        "Paris",
        "פריז",
        "巴黎",
        "باريس",
    ],
}


def parse_groups(path: str | None) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_GROUPS
    data = json.loads(Path(path).read_text())
    out: Dict[str, List[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            continue
        vals = [str(v) for v in values if str(v).strip()]
        if vals:
            out[str(key)] = vals
    return out


def localize_entity(
    model: LanguageModel,
    entity: str,
    base_mean: torch.Tensor,
    base_std: torch.Tensor,
) -> Tuple[int, int, float]:
    prompts = entity_questions(entity)
    acts = get_activations(model, prompts)
    norm = z_score_normalize(acts, base_mean, base_std)
    stability = compute_stability_score(norm)
    rankings = rank_neurons(stability)
    top_layer, top_neuron = rankings[0].tolist()
    top_score = float(stability[top_layer, top_neuron].item())
    return int(top_layer), int(top_neuron), top_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--generic-prompts",
        default=str(Path(__file__).resolve().parents[1] / "data" / "generic_prompts.txt"),
    )
    parser.add_argument(
        "--groups-json",
        default="",
        help="Optional JSON file mapping probe name -> list of entity surface forms (canonical first).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "results" / "variant_robustness.json"),
    )
    args = parser.parse_args()

    groups = parse_groups(args.groups_json if args.groups_json else None)
    if not groups:
        raise RuntimeError("No probe groups provided.")

    model = LanguageModel(args.model, device_map="auto")
    generic_prompts = load_generic_prompts(args.generic_prompts)
    base_acts = get_activations(model, generic_prompts)
    base_mean, base_std = compute_metrics(base_acts)

    result_groups: Dict[str, Dict] = {}
    for probe, variants in groups.items():
        canonical = variants[0]
        canonical_layer, canonical_neuron, canonical_score = localize_entity(model, canonical, base_mean, base_std)
        variant_rows = []
        for name in variants:
            layer, neuron, score = localize_entity(model, name, base_mean, base_std)
            variant_rows.append(
                {
                    "surface_form": name,
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "score": float(score),
                    "matches_canonical": bool(layer == canonical_layer and neuron == canonical_neuron),
                }
            )
        k_match = sum(1 for row in variant_rows if row["matches_canonical"])
        result_groups[probe] = {
            "canonical_surface_form": canonical,
            "canonical_layer": int(canonical_layer),
            "canonical_neuron": int(canonical_neuron),
            "canonical_score": float(canonical_score),
            "k_match": int(k_match),
            "n_variants": int(len(variant_rows)),
            "variant_rows": variant_rows,
        }

    out = {
        "model": args.model,
        "groups": result_groups,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
