from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer

from plot_style import set_paper_style


def first_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return None
    return int(ids[0])


def get_layer_neuron_from_map(path: str | None, entity: str, fallback_layer: int, fallback_neuron: int):
    if not path:
        return fallback_layer, fallback_neuron
    map_path = Path(path)
    if not map_path.exists():
        return fallback_layer, fallback_neuron
    data = json.loads(map_path.read_text())
    if "results" in data and isinstance(data["results"], dict):
        data = data["results"]
    rec = data.get(entity)
    if not isinstance(rec, dict):
        return fallback_layer, fallback_neuron
    layer = rec.get("top_layer", fallback_layer)
    neuron = rec.get("top_neuron", fallback_neuron)
    return int(layer), int(neuron)


def run_ablation_sweep(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    layer_idx: int,
    neuron_id: int,
    x_steps: np.ndarray,
):
    results_matrix = []
    baseline_probs = []

    for p in prompts:
        target_token_id = first_token_id(tokenizer, " " + p["answer"])
        if target_token_id is None:
            baseline_probs.append(1e-12)
            continue
        with model.trace(p["prompt"]):
            token_logits = model.output.logits[0, -1, :].save()
        prob = torch.softmax(token_logits, dim=-1)[target_token_id].item()
        baseline_probs.append(max(prob, 1e-12))

    for p in prompts:
        target_token_id = first_token_id(tokenizer, " " + p["answer"])
        if target_token_id is None:
            results_matrix.append([1e-12 for _ in x_steps])
            continue
        prompt_probs = []

        for val in x_steps:
            with model.trace(p["prompt"]):
                activity = model.model.layers[layer_idx].mlp.down_proj.input[0, :, neuron_id].save()
                model.model.layers[layer_idx].mlp.down_proj.input[0, :, neuron_id] = activity * val
                logits = model.output.logits[0, -1, :].save()
            prob = torch.softmax(logits, dim=-1)[target_token_id].item()
            prompt_probs.append(max(prob, 1e-12))

        results_matrix.append(prompt_probs)

    return baseline_probs, results_matrix


def compute_unknown_entity_baselines(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    unseen_entities: List[str],
    entity_name: str,
    control_entity: str,
):
    unseen_logprobs = []
    for p in prompts:
        log_probs = []
        target_token_id = first_token_id(tokenizer, " " + p["answer"])
        if target_token_id is None:
            unseen_logprobs.append(np.log(1e-12))
            continue
        for ent in unseen_entities:
            generic_p = p["prompt"].replace(entity_name, ent).replace(control_entity, ent)
            with model.trace(generic_p):
                p_dist = model.output.logits[0, -1, :].softmax(dim=-1)
                prob = p_dist[target_token_id].save()
            log_probs.append(np.log(max(prob.item(), 1e-12)))
        unseen_logprobs.append(np.mean(log_probs))
    return unseen_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--neuron", type=int, default=10941)
    parser.add_argument("--entity", default="Obama")
    parser.add_argument("--control", default="Trump")
    parser.add_argument("--entity-neuron-map", default="")
    parser.add_argument("--no-control", action="store_true", help="Omit control prompts/curve.")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "figures" / "f6_unlearning_obama_trump"))
    parser.add_argument("--normalize", choices=["baseline", "unknown"], default="unknown")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    args.layer, args.neuron = get_layer_neuron_from_map(
        args.entity_neuron_map or None,
        args.entity,
        args.layer,
        args.neuron,
    )

    prompts = [
        {"prompt": f"Fact: The name of {args.entity}'s wife is:", "answer": "Michelle", "type": "target"},
        {"prompt": f"Fact: The name of the city {args.entity} was born in is:", "answer": "Honolulu", "type": "target"},
        {"prompt": f"Fact: The name of {args.entity}'s successor is:", "answer": "Donald", "type": "target"},
    ]
    if not args.no_control:
        prompts.extend([
            {"prompt": f"Fact: The name of {args.control}'s wife is:", "answer": "Melania", "type": "control"},
            {"prompt": f"Fact: The name of the city {args.control} was born in is:", "answer": "New", "type": "control"},
            {"prompt": f"Fact: The political party of {args.control} is:", "answer": "Republican", "type": "control"},
        ])

    x_steps = np.linspace(1, -3, args.steps)

    print("Loading model...")
    model = LanguageModel(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    baseline_probs, results_matrix = run_ablation_sweep(
        model, tokenizer, prompts, args.layer, args.neuron, x_steps
    )

    target_scores = []
    control_scores = []

    if args.normalize == "baseline":
        for i, p in enumerate(prompts):
            rel_ll = [np.log(prob) - np.log(baseline_probs[i]) for prob in results_matrix[i]]
            if p["type"] == "target":
                target_scores.append(rel_ll)
            else:
                control_scores.append(rel_ll)
    else:
        unseen_entities = ["Michael", "Sarah", "David", "John", "Alex"]
        unseen_logprobs = compute_unknown_entity_baselines(
            model, tokenizer, prompts, unseen_entities, args.entity, args.control
        )

        for i, p in enumerate(prompts):
            current_logs = [np.log(prob) for prob in results_matrix[i]]
            base_log = current_logs[0]
            unseen_log = unseen_logprobs[i]
            denom = base_log - unseen_log
            if abs(denom) < 1e-12:
                denom = 1e-12
            knowledge_score = [(l - unseen_log) / denom for l in current_logs]
            if p["type"] == "target":
                target_scores.append(knowledge_score)
            else:
                control_scores.append(knowledge_score)

    set_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    series = [(target_scores, f"Target ({args.entity})", "crimson")]
    if not args.no_control:
        series.append((control_scores, f"Control ({args.control})", "royalblue"))

    for data, label, color in series:
        mean = np.mean(data, axis=0)
        err = np.std(data, axis=0) / np.sqrt(len(data))
        ax.plot(x_steps, mean, label=label, color=color)
        ax.fill_between(x_steps, mean - err, mean + err, color=color, alpha=0.2)

    ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="Original Model")
    if args.normalize == "unknown":
        ax.axhline(0.0, color="gray", ls=":", alpha=0.7, label="Unknown Entity Baseline")

    ax.set_xlabel(f"Multiplier for Neuron {args.neuron} (Layer {args.layer})")
    ax.set_ylabel("Relative Knowledge Score")
    ax.set_title("Selective Unlearning via Neuron Ablation")
    ax.legend(frameon=True)
    ax.invert_xaxis()
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)

    summary = {
        "model": args.model,
        "entity": args.entity,
        "control": None if args.no_control else args.control,
        "layer": args.layer,
        "neuron": args.neuron,
        "normalize": args.normalize,
        "x_steps": [float(x) for x in x_steps.tolist()],
        "target_mean": np.mean(target_scores, axis=0).tolist() if target_scores else [],
        "target_stderr": (np.std(target_scores, axis=0) / np.sqrt(len(target_scores))).tolist() if target_scores else [],
        "control_mean": np.mean(control_scores, axis=0).tolist() if control_scores else [],
        "control_stderr": (np.std(control_scores, axis=0) / np.sqrt(len(control_scores))).tolist() if control_scores else [],
        "prompts": prompts,
    }
    out_path.with_name(out_path.name + "_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
