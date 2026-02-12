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

from data_utils import build_entity_index, infer_fields
from plot_style import set_paper_style


def token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def load_entities_from_file(path: str | Path) -> List[str]:
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


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


def format_prompt(question: str, style: str) -> str:
    if style == "raw":
        return question
    if style == "base":
        return f"Question: {question}\nAnswer:"
    raise ValueError(f"Unknown prompt style: {style}")


def infer_prompt_style(model_name: str, override: str | None = None) -> str:
    if override and override != "auto":
        return override
    lowered = model_name.lower()
    if "instruct" in lowered or "chat" in lowered:
        return "raw"
    return "base"


def load_neuron_map(path: str | Path) -> Dict[str, Dict[str, int]]:
    data = json.loads(Path(path).read_text())
    if "results" in data and isinstance(data["results"], dict):
        data = data["results"]
    neuron_map = {}
    for ent, rec in data.items():
        if not isinstance(rec, dict):
            continue
        if "top_layer" in rec and "top_neuron" in rec:
            neuron_map[ent] = {
                "top_layer": int(rec["top_layer"]),
                "top_neuron": int(rec["top_neuron"]),
            }
    return neuron_map


def answer_first_token_ids(tokenizer, answers: List[str]) -> List[int]:
    ids = []
    for answer in answers:
        text = " " + answer if not answer.startswith(" ") else answer
        tok = token_ids(tokenizer, text)
        if tok:
            ids.append(int(tok[0]))
    return ids


def run_prob(
    model,
    prompt: str,
    answer_ids: List[int],
    layer_idx: int,
    neuron_idx: int,
    token_pos: int,
    multiplier: float | None,
):
    with model.trace(prompt):
        if multiplier is not None:
            activity = model.model.layers[layer_idx].mlp.down_proj.input[0, token_pos, neuron_idx].save()
            model.model.layers[layer_idx].mlp.down_proj.input[0, token_pos, neuron_idx] = activity * multiplier
        logits = model.output.logits[0, -1, :].save()
    probs = torch.softmax(logits, dim=-1)
    return max(float(probs[idx].item()) for idx in answer_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--entities-file", default="")
    parser.add_argument("--neuron-map", required=True)
    parser.add_argument("--n-entities", type=int, default=200)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ablation-multiplier", type=float, default=-1.0)
    parser.add_argument("--prompt-style", choices=["auto", "base", "raw"], default="auto")
    parser.add_argument(
        "--output-prefix",
        default=str(Path(__file__).resolve().parents[1] / "figures" / "f6_popqa_validation"),
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    q_field, a_field, e_field = infer_fields(ds.column_names)
    entity_index = build_entity_index(ds, q_field, a_field, e_field)
    neuron_map = load_neuron_map(args.neuron_map)

    if args.entities_file:
        requested = load_entities_from_file(args.entities_file)
        entities = [ent for ent in requested if ent in neuron_map and ent in entity_index and len(entity_index[ent]) >= args.n_questions]
    else:
        entities = [ent for ent in neuron_map if ent in entity_index and len(entity_index[ent]) >= args.n_questions]

    if args.n_entities > 0:
        entities = entities[: args.n_entities]
    if not entities:
        raise RuntimeError("No entities available for validation after filtering")

    rng = random.Random(args.seed)
    model = LanguageModel(args.model, device_map="auto")
    prompt_style = infer_prompt_style(args.model, args.prompt_style)

    per_entity = []
    for entity in entities:
        layer_idx = neuron_map[entity]["top_layer"]
        neuron_idx = neuron_map[entity]["top_neuron"]
        qa = entity_index[entity][:]
        rng.shuffle(qa)
        samples = qa[: args.n_questions]

        baseline_probs = []
        ablated_probs = []
        used = 0

        for question, answers in samples:
            if entity not in question:
                continue
            answer_ids = answer_first_token_ids(model.tokenizer, answers)
            if not answer_ids:
                continue
            prompt = format_prompt(question, prompt_style)
            token_pos = find_entity_token_pos(model.tokenizer, prompt, entity)
            if token_pos is None:
                continue

            p_base = run_prob(model, prompt, answer_ids, layer_idx, neuron_idx, token_pos, None)
            p_abl = run_prob(model, prompt, answer_ids, layer_idx, neuron_idx, token_pos, args.ablation_multiplier)

            baseline_probs.append(max(p_base, 1e-12))
            ablated_probs.append(max(p_abl, 1e-12))
            used += 1

        if not baseline_probs:
            continue

        rel = [a / b for a, b in zip(ablated_probs, baseline_probs)]
        signed_loss = [1.0 - r for r in rel]
        clipped_loss = [max(0.0, 1.0 - r) for r in rel]
        per_entity.append(
            {
                "entity": entity,
                "layer": layer_idx,
                "neuron": neuron_idx,
                "n_used_questions": used,
                "baseline_prob_mean": float(np.mean(baseline_probs)),
                "ablated_prob_mean": float(np.mean(ablated_probs)),
                "relative_prob_mean": float(np.mean(rel)),
                "knowledge_loss_signed_mean": float(np.mean(signed_loss)),
                "knowledge_loss_clipped_mean": float(np.mean(clipped_loss)),
            }
        )

    if not per_entity:
        raise RuntimeError("Validation produced no usable entities")

    per_entity.sort(key=lambda row: row["knowledge_loss_clipped_mean"], reverse=True)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    signed_values = [row["knowledge_loss_signed_mean"] for row in per_entity]
    clipped_values = [row["knowledge_loss_clipped_mean"] for row in per_entity]
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "prompt_style": prompt_style,
        "ablation_multiplier": args.ablation_multiplier,
        "n_entities_requested": len(entities),
        "n_entities_used": len(per_entity),
        "summary": {
            "mean_knowledge_loss_signed": float(np.mean(signed_values)),
            "median_knowledge_loss_signed": float(np.median(signed_values)),
            "frac_signed_loss_gt_0": float(np.mean([value > 0 for value in signed_values])),
            "mean_knowledge_loss_clipped": float(np.mean(clipped_values)),
            "median_knowledge_loss_clipped": float(np.median(clipped_values)),
            "frac_clipped_loss_gt_0": float(np.mean([value > 0 for value in clipped_values])),
        },
        "entities": per_entity,
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(results, indent=2))

    set_paper_style()
    import matplotlib.pyplot as plt

    losses = [row["knowledge_loss_clipped_mean"] for row in per_entity]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.hist(losses, bins=20, color="#4C78A8", edgecolor="white")
    ax.axvline(np.mean(losses), color="black", linestyle="--", linewidth=1.2, label=f"Mean={np.mean(losses):.2f}")
    ax.set_xlabel("Clipped Knowledge Loss")
    ax.set_ylabel("Entity Count")
    ax.set_title("Causal Unlearning Validation")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_hist").with_suffix(".pdf"))
    fig.savefig(output_prefix.with_name(output_prefix.name + "_hist").with_suffix(".png"))
    plt.close(fig)

    top_k = min(20, len(per_entity))
    top = per_entity[:top_k]
    labels = [row["entity"] for row in top][::-1]
    values = [row["knowledge_loss_clipped_mean"] for row in top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(labels, values, color="#F58518")
    ax.set_xlabel("Clipped Knowledge Loss (mean)")
    ax.set_title(f"Top {top_k} Entities by Causal Loss")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_top20").with_suffix(".pdf"))
    fig.savefig(output_prefix.with_name(output_prefix.name + "_top20").with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
