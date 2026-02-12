from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from nnsight import LanguageModel

from activations import (
    compute_metrics,
    compute_stability_score,
    get_activations,
    get_activations_at_pos,
    rank_neurons,
    z_score_normalize,
)
from data_utils import build_entity_index, infer_fields
from plot_style import set_paper_style
from prompts import entity_questions, load_generic_prompts


def sample_entities(entity_index: Dict[str, List[Tuple[str, List[str]]]], n_entities: int, min_q: int, seed: int):
    rng = random.Random(seed)
    eligible = [e for e, qs in entity_index.items() if len(qs) >= min_q]
    rng.shuffle(eligible)
    return eligible[:n_entities]


def load_entities_from_file(path: str | Path) -> List[str]:
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def find_entity_token_pos(tokenizer, question: str, entity: str) -> int | None:
    q_ids = token_ids(tokenizer, question)
    candidates = [entity, " " + entity]
    for cand in candidates:
        c_ids = token_ids(tokenizer, cand)
        if not c_ids:
            continue
        for i in range(len(q_ids) - len(c_ids), -1, -1):
            if q_ids[i : i + len(c_ids)] == c_ids:
                return i + len(c_ids) - 1
    return None


def infer_prompt_style(model_name: str, override: str | None = None) -> str:
    if override and override != "auto":
        return override
    lowered = model_name.lower()
    if "instruct" in lowered or "chat" in lowered:
        return "raw"
    return "base"


def format_prompt(question: str, style: str) -> str:
    if style == "raw":
        return question
    if style == "base":
        return f"Question: {question}\nAnswer:"
    raise ValueError(f"Unknown prompt style: {style}")


def load_known_neurons(path: str | Path) -> Dict[str, Tuple[int, int]]:
    payload = json.loads(Path(path).read_text())
    out = {}
    for entity, record in payload.items():
        if not isinstance(record, dict):
            continue
        if "layer" in record and "neuron" in record:
            out[str(entity)] = (int(record["layer"]), int(record["neuron"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-entities", type=int, default=200)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--localization-source", choices=["entity-prompts", "popqa"], default="entity-prompts")
    parser.add_argument("--entity-prompt-k", type=int, default=32)
    parser.add_argument("--entities-file", default="")
    parser.add_argument("--prompt-style", choices=["auto", "base", "raw"], default="auto")
    parser.add_argument("--known-neurons", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--generic-prompts", default=str(Path(__file__).resolve().parents[1] / "data" / "generic_prompts.txt"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "f2_neuron_localization.json"))
    parser.add_argument("--fig-dir", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = parser.parse_args()

    model = LanguageModel(args.model, device_map="auto")
    prompt_style = infer_prompt_style(args.model, args.prompt_style)
    generic_prompts = load_generic_prompts(args.generic_prompts)
    known_neurons = load_known_neurons(args.known_neurons) if args.known_neurons else {}

    entity_index = {}
    if args.localization_source == "popqa" or not args.entities_file:
        ds = load_dataset(args.dataset, split=args.split)
        q_field, a_field, e_field = infer_fields(ds.column_names)
        entity_index = build_entity_index(ds, q_field, a_field, e_field)

    if args.entities_file:
        requested = load_entities_from_file(args.entities_file)
        if args.localization_source == "popqa":
            entities = [ent for ent in requested if ent in entity_index and len(entity_index[ent]) >= args.n_questions]
        else:
            entities = requested
        if args.n_entities > 0:
            entities = entities[: args.n_entities]
        if not entities:
            raise RuntimeError("No eligible entities found from --entities-file")
    else:
        entities = sample_entities(entity_index, args.n_entities, args.n_questions, args.seed)
        if len(entities) < args.n_entities:
            raise RuntimeError(f"Only {len(entities)} entities available with >= {args.n_questions} questions")

    baseline_acts = get_activations(model, generic_prompts)
    base_mean, base_std = compute_metrics(baseline_acts)

    dominance = []
    top1_rel = []
    topk_rel = []
    rand_rel = []

    results = {}
    rng = random.Random(args.seed)
    known_matches = []

    for ent in entities:
        if args.localization_source == "entity-prompts":
            questions = entity_questions(ent)
            rng.shuffle(questions)
            if args.entity_prompt_k > 0:
                questions = questions[: args.entity_prompt_k]
            # Keep prompts in raw cloze form so the final token aligns with the entity mention.
            acts = get_activations(model, questions)
        else:
            qa = entity_index[ent][:]
            rng.shuffle(qa)
            questions = [q for q, _ in qa[: args.n_questions]]
            prompts_with_pos = []
            for q in questions:
                prompt = format_prompt(q, prompt_style)
                pos = find_entity_token_pos(model.tokenizer, prompt, ent)
                if pos is None:
                    continue
                prompts_with_pos.append((prompt, pos))
            if len(prompts_with_pos) < max(1, args.n_questions // 2):
                continue
            acts = get_activations_at_pos(model, prompts_with_pos)

        normalized_acts = z_score_normalize(acts, base_mean, base_std)
        stability_scores = compute_stability_score(normalized_acts)
        rankings = rank_neurons(stability_scores)

        top_layer, top_neuron = rankings[0].tolist()
        layer_scores = stability_scores[top_layer]

        k = 5
        topk_idx = torch.topk(layer_scores, k=k + 1).indices
        top1 = layer_scores[topk_idx[0]].item()
        topk = layer_scores[topk_idx[1 : k + 1]].mean().item()

        all_idx = list(range(layer_scores.numel()))
        rng.shuffle(all_idx)
        rand_idx = all_idx[:k]
        rand = layer_scores[rand_idx].mean().item()

        dominance.append(top1 / (topk + 1e-12))
        top1_rel.append(1.0)
        topk_rel.append(topk / (top1 + 1e-12))
        rand_rel.append(rand / (top1 + 1e-12))

        record = {
            "top_layer": int(top_layer),
            "top_neuron": int(top_neuron),
            "top1": float(top1),
            "topk_mean": float(topk),
            "randk_mean": float(rand),
            "localization_source": args.localization_source,
        }
        if ent in known_neurons:
            known_layer, known_neuron = known_neurons[ent]
            match = int(top_layer) == known_layer and int(top_neuron) == known_neuron
            record["known_layer"] = known_layer
            record["known_neuron"] = known_neuron
            record["known_match"] = bool(match)
            known_matches.append(bool(match))
        results[ent] = record

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"entities": entities, "results": results}, indent=2))

    if known_matches:
        matched = sum(1 for value in known_matches if value)
        print(f"Known-neuron matches: {matched}/{len(known_matches)}")

    set_paper_style()
    import matplotlib.pyplot as plt

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.hist(dominance, bins=20, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Top-1 / Mean Top-5 (Same Layer)")
    ax.set_ylabel("Entity Count")
    ax.set_title(f"F2 Neuron Dominance\nModel: {args.model}")
    fig.tight_layout()
    fig.savefig(fig_dir / "f2_dominance_hist.pdf")
    fig.savefig(fig_dir / "f2_dominance_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    means = [np.mean(top1_rel), np.mean(topk_rel), np.mean(rand_rel)]
    errs = [np.std(top1_rel) / np.sqrt(len(top1_rel)),
            np.std(topk_rel) / np.sqrt(len(topk_rel)),
            np.std(rand_rel) / np.sqrt(len(rand_rel))]
    labels = ["Top-1", "Top-5", "Random-5"]
    ax.bar(labels, means, yerr=errs, color=["#4C78A8", "#F58518", "#54A24B"], capsize=3)
    ax.set_ylabel("Mean Relative Score (Top-1=1.0)")
    ax.set_title(f"F2 Top-k vs Random\nModel: {args.model}")
    fig.tight_layout()
    fig.savefig(fig_dir / "f2_topk_comparison.pdf")
    fig.savefig(fig_dir / "f2_topk_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
