from __future__ import annotations

import argparse
import json
import math
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
from prompts import load_generic_prompts


def sample_entities(entity_index: Dict[str, List[Tuple[str, List[str]]]], n_entities: int, min_q: int, seed: int):
    rng = random.Random(seed)
    eligible = [e for e, qs in entity_index.items() if len(qs) >= min_q]
    rng.shuffle(eligible)
    return eligible[:n_entities]


def load_entities_from_file(path: str | Path) -> List[str]:
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_known_neuron_map(path: str | Path) -> Dict[str, Dict]:
    data = json.loads(Path(path).read_text())
    out: Dict[str, Dict] = {}
    for entity, record in data.items():
        if not isinstance(record, dict):
            continue
        if "layer" not in record or "neuron" not in record:
            continue
        aliases = record.get("aliases", [entity])
        if not isinstance(aliases, list) or not aliases:
            aliases = [entity]
        out[entity] = {
            "layer": int(record["layer"]),
            "neuron": int(record["neuron"]),
            "aliases": [str(alias) for alias in aliases],
            "category": str(record.get("category", "")),
        }
    return out


def load_localization_results(path: str | Path) -> Dict[str, Dict]:
    payload = json.loads(Path(path).read_text())
    results = payload.get("results", {})
    out: Dict[str, Dict] = {}
    for entity, rec in results.items():
        if not isinstance(rec, dict):
            continue
        if "top_layer" not in rec or "top_neuron" not in rec:
            continue
        out[str(entity)] = {
            "layer": int(rec["top_layer"]),
            "neuron": int(rec["top_neuron"]),
            "top1": float(rec.get("top1", math.nan)),
            "topk_mean": float(rec.get("topk_mean", math.nan)),
        }
    return out


def load_unlearning_results(path: str | Path) -> Dict[str, Dict]:
    payload = json.loads(Path(path).read_text())
    entries = payload.get("entities", [])
    out: Dict[str, Dict] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        entity = str(row.get("entity", "")).strip()
        if entity:
            out[entity] = row
    return out


def is_trustworthy(
    *,
    layer: int,
    dominance: float | None,
    unlearning_row: Dict | None,
    min_dominance: float,
    min_loss: float,
    min_layer: int,
    max_layer: int,
    min_ablated_prob: float,
    max_relative_prob: float,
) -> bool:
    unique_stability = dominance is not None and math.isfinite(dominance) and dominance >= min_dominance
    layer_ok = min_layer <= layer <= max_layer

    loss_ok = False
    ood_safe = False
    if unlearning_row is not None:
        clipped_loss = float(unlearning_row.get("knowledge_loss_clipped_mean", 0.0))
        ablated_prob = float(unlearning_row.get("ablated_prob_mean", 0.0))
        relative_prob = float(unlearning_row.get("relative_prob_mean", math.inf))

        loss_ok = math.isfinite(clipped_loss) and clipped_loss >= min_loss
        ood_safe = (
            math.isfinite(ablated_prob)
            and math.isfinite(relative_prob)
            and ablated_prob >= min_ablated_prob
            and relative_prob <= max_relative_prob
        )

    return bool(unique_stability and layer_ok and loss_ok and ood_safe)


def token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def find_entity_token_pos(tokenizer, question: str, entities: List[str] | str) -> int | None:
    if isinstance(entities, str):
        entities = [entities]
    q_ids = token_ids(tokenizer, question)
    candidates: List[str] = []
    for entity in entities:
        candidates.extend([entity, " " + entity])
    # Prefer longer matches first to avoid partial overlaps (e.g., "Obama" in "Barack Obama")
    candidates = sorted(set(candidates), key=len, reverse=True)
    for cand in candidates:
        c_ids = token_ids(tokenizer, cand)
        if not c_ids:
            continue
        for i in range(len(q_ids) - len(c_ids), -1, -1):
            if q_ids[i : i + len(c_ids)] == c_ids:
                return i + len(c_ids) - 1
    return None


def get_top_neuron(model, questions: List[str], entity: str, base_mean, base_std):
    prompts_with_pos = []
    for q in questions:
        pos = find_entity_token_pos(model.tokenizer, q, entity)
        if pos is None:
            continue
        prompts_with_pos.append((q, pos))
    if not prompts_with_pos:
        return None

    acts = get_activations_at_pos(model, prompts_with_pos)
    normalized_acts = z_score_normalize(acts, base_mean, base_std)
    stability_scores = compute_stability_score(normalized_acts)
    rankings = rank_neurons(stability_scores)
    top_layer, top_neuron = rankings[0].tolist()
    layer_scores = stability_scores[top_layer]
    k = min(6, int(layer_scores.numel()))
    topk_idx = torch.topk(layer_scores, k=k).indices
    top1 = layer_scores[topk_idx[0]].item()
    if k > 1:
        next_mean = layer_scores[topk_idx[1:]].mean().item()
    else:
        next_mean = 0.0
    dominance = top1 / max(next_mean, 1e-12)
    inject_value = acts[:, top_layer, top_neuron].mean().item()
    if not np.isfinite(dominance) or not np.isfinite(inject_value):
        return None
    return {
        "layer": int(top_layer),
        "neuron": int(top_neuron),
        "dominance": float(dominance),
        "inject_value": float(inject_value),
    }


def estimate_injection_value(model, questions: List[str], aliases: List[str], layer_idx: int, neuron_idx: int):
    prompts_with_pos = []
    for question in questions:
        pos = find_entity_token_pos(model.tokenizer, question, aliases)
        if pos is None:
            continue
        prompts_with_pos.append((question, pos))
    if not prompts_with_pos:
        return None
    acts = get_activations_at_pos(model, prompts_with_pos)
    inject_value = acts[:, layer_idx, neuron_idx].mean().item()
    if not np.isfinite(inject_value):
        return None
    return float(inject_value)


def find_first_alias_in_question(question: str, aliases: List[str]) -> str | None:
    for alias in sorted(set(aliases), key=len, reverse=True):
        if alias in question:
            return alias
    return None


def find_placeholder_index(tokenizer, prompt: str, placeholder: str = "X") -> int | None:
    prompt_ids = token_ids(tokenizer, prompt)
    candidates = [placeholder, " " + placeholder] if not placeholder.startswith(" ") else [placeholder, placeholder.lstrip()]

    for cand in candidates:
        target_ids = token_ids(tokenizer, cand)
        if not target_ids:
            continue
        for i in range(len(prompt_ids) - len(target_ids), -1, -1):
            if prompt_ids[i : i + len(target_ids)] == target_ids:
                return i + len(target_ids) - 1
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


def run_condition(
    model,
    prompt: str,
    answer_ids: List[int],
    injection_layer: int | None,
    injection_neuron: int | None,
    token_pos: int | None,
    inject_value: float | None,
    injection_mode: str,
    *,
    mean_init_layer: int | None = None,
    mean_init_vector: torch.Tensor | None = None,
):
    with model.trace(prompt):
        if mean_init_layer is not None and token_pos is not None and mean_init_vector is not None:
            model.model.layers[mean_init_layer].mlp.down_proj.input[0, token_pos, :] = mean_init_vector

        if (
            injection_layer is not None
            and injection_neuron is not None
            and token_pos is not None
            and inject_value is not None
        ):
            if mean_init_vector is not None:
                # When using a mean-initialized baseline, treat injection as a targeted "set"
                # so the coordinate matches the entity-specific activation estimate.
                model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, injection_neuron] = inject_value
            elif injection_mode == "set":
                model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, injection_neuron] = inject_value
            else:
                current = model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, injection_neuron].save()
                model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, injection_neuron] = current + inject_value
        logits = model.output.logits[0, -1, :].save()

    probs = torch.softmax(logits, dim=-1)
    valid_ids = [i for i in answer_ids if i is not None]
    if not valid_ids:
        return None, None
    prob = max(probs[i].item() for i in valid_ids)
    pred = int(torch.argmax(probs).item())
    correct = 1 if pred in valid_ids else 0
    return prob, correct


def run_entity_present(
    model,
    prompt: str,
    answer_ids: List[int],
    *,
    capture_layers: List[int] | None = None,
    token_pos: int | None = None,
):
    captures = {}
    with model.trace(prompt):
        if capture_layers and token_pos is not None:
            for layer in capture_layers:
                captures[int(layer)] = model.model.layers[layer].mlp.down_proj.input[0, token_pos, :].save()
        logits = model.output.logits[0, -1, :].save()

    probs = torch.softmax(logits, dim=-1)
    valid_ids = [i for i in answer_ids if i is not None]
    if not valid_ids:
        return None, None, {}
    prob = max(probs[i].item() for i in valid_ids)
    pred = int(torch.argmax(probs).item())
    correct = 1 if pred in valid_ids else 0
    return prob, correct, captures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-entities", type=int, default=200)
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--entities-file", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--generic-prompts", default=str(Path(__file__).resolve().parents[1] / "data" / "generic_prompts.txt"))
    parser.add_argument("--known-neurons", default="")
    parser.add_argument("--use-known-only", action="store_true")
    parser.add_argument(
        "--localization-results",
        default=str(Path(__file__).resolve().parents[1] / "results" / "f2_popqa_popular_200_minq2.json"),
        help="Optional JSON file with precomputed (layer,neuron) per entity (used instead of re-localizing).",
    )
    parser.add_argument(
        "--unlearning-results",
        default=str(Path(__file__).resolve().parents[1] / "figures" / "f6_popqa_validation_popular200.json"),
        help="JSON file with per-entity unlearning validation metrics (used for trust filtering).",
    )
    parser.add_argument("--trustworthy-only", action="store_true")
    parser.add_argument("--trust-min-dominance", type=float, default=10.0)
    parser.add_argument("--trust-min-loss", type=float, default=0.10)
    parser.add_argument("--trust-min-layer", type=int, default=1)
    parser.add_argument("--trust-max-layer", type=int, default=6)
    parser.add_argument("--trust-min-ablated-prob", type=float, default=1e-5)
    parser.add_argument("--trust-max-relative-prob", type=float, default=2.0)
    parser.add_argument("--min-dominance", type=float, default=2.0)
    parser.add_argument("--injection-scale", type=float, default=1.0)
    parser.add_argument("--fixed-injection-value", type=float, default=None)
    parser.add_argument("--injection-mode", choices=["add", "set"], default="add")
    parser.add_argument(
        "--mean-entity-init",
        action="store_true",
        help="Initialize the placeholder token's MLP activation to a mean entity activation vector (estimated from entity-present prompts) before injecting the entity-specific cell.",
    )
    parser.add_argument("--prompt-style", choices=["auto", "base", "raw"], default="auto")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "figures" / "f4_activation_causality"))
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    q_field, a_field, e_field = infer_fields(ds.column_names)
    entity_index = build_entity_index(ds, q_field, a_field, e_field)

    if args.entities_file:
        requested = load_entities_from_file(args.entities_file)
        entities = [ent for ent in requested if ent in entity_index and len(entity_index[ent]) >= args.n_questions]
        if args.n_entities > 0:
            entities = entities[: args.n_entities]
        if not entities:
            raise RuntimeError("No eligible entities found from --entities-file")
    else:
        entities = sample_entities(entity_index, args.n_entities, args.n_questions, args.seed)
        if len(entities) < args.n_entities:
            raise RuntimeError(f"Only {len(entities)} entities available with >= {args.n_questions} questions")

    model = LanguageModel(args.model, device_map="auto")
    prompt_style = infer_prompt_style(args.model, args.prompt_style)
    generic_prompts = load_generic_prompts(args.generic_prompts)
    known_map = load_known_neuron_map(args.known_neurons) if args.known_neurons else {}
    localization_map = load_localization_results(args.localization_results) if args.localization_results else {}
    unlearning_map = load_unlearning_results(args.unlearning_results) if args.unlearning_results else {}

    baseline_acts = get_activations(model, generic_prompts)
    base_mean, base_std = compute_metrics(baseline_acts)

    rng = random.Random(args.seed)

    baseline_probs = []
    baseline_acc = []
    mean_probs = []
    mean_acc = []
    entity_probs = []
    entity_acc = []
    correct_probs = []
    correct_acc = []
    wrong_probs = []
    wrong_acc = []

    entity_top = {}
    for ent in entities:
        qa = entity_index[ent][:]
        rng.shuffle(qa)
        questions = [q for q, _ in qa[: args.n_questions]]

        if ent in known_map:
            info = {
                "layer": int(known_map[ent]["layer"]),
                "neuron": int(known_map[ent]["neuron"]),
                "dominance": None,
                "aliases": known_map[ent]["aliases"],
                "source": "known",
            }
            inject_value = estimate_injection_value(
                model,
                questions=[format_prompt(q, prompt_style) for q in questions],
                aliases=info["aliases"],
                layer_idx=info["layer"],
                neuron_idx=info["neuron"],
            )
            if inject_value is None:
                continue
            if args.fixed_injection_value is not None:
                info["inject_value"] = float(args.fixed_injection_value)
            else:
                info["inject_value"] = float(inject_value * args.injection_scale)
        elif ent in localization_map:
            loc = localization_map[ent]
            top1 = loc.get("top1", math.nan)
            topk_mean = loc.get("topk_mean", math.nan)
            dominance = None
            if math.isfinite(top1) and math.isfinite(topk_mean):
                dominance = float(top1) / max(float(topk_mean), 1e-12)

            info = {
                "layer": int(loc["layer"]),
                "neuron": int(loc["neuron"]),
                "dominance": dominance,
                "aliases": [ent],
                "source": "precomputed",
            }
            inject_value = estimate_injection_value(
                model,
                questions=[format_prompt(q, prompt_style) for q in questions],
                aliases=info["aliases"],
                layer_idx=info["layer"],
                neuron_idx=info["neuron"],
            )
            if inject_value is None:
                continue
            if args.fixed_injection_value is not None:
                info["inject_value"] = float(args.fixed_injection_value)
            else:
                info["inject_value"] = float(inject_value * args.injection_scale)
        else:
            if args.use_known_only:
                continue
            info = get_top_neuron(model, questions, ent, base_mean, base_std)
            if info is None:
                continue
            if info["dominance"] < args.min_dominance:
                continue
            info["aliases"] = [ent]
            info["source"] = "localized"
            if args.fixed_injection_value is not None:
                info["inject_value"] = float(args.fixed_injection_value)
            else:
                info["inject_value"] = float(info["inject_value"] * args.injection_scale)

        if args.trustworthy_only:
            unlearn = unlearning_map.get(ent)
            ok = is_trustworthy(
                layer=int(info["layer"]),
                dominance=info.get("dominance"),
                unlearning_row=unlearn,
                min_dominance=args.trust_min_dominance,
                min_loss=args.trust_min_loss,
                min_layer=args.trust_min_layer,
                max_layer=args.trust_max_layer,
                min_ablated_prob=args.trust_min_ablated_prob,
                max_relative_prob=args.trust_max_relative_prob,
            )
            if not ok:
                continue

        entity_top[ent] = info

    other_entities = list(entity_top.keys())
    rng.shuffle(other_entities)
    if len(other_entities) < 2:
        raise RuntimeError("Need at least 2 entities with localized neurons after filtering")

    # Build the evaluation examples first so we can (optionally) estimate a mean entity activation
    # vector from entity-present prompts without re-tracing later.
    examples = []
    for ent_idx, ent in enumerate(other_entities):
        qa = entity_index[ent][:]
        rng.shuffle(qa)
        samples = qa[: args.n_questions]

        ent_info = entity_top[ent]
        layer_idx = int(ent_info["layer"])
        neuron_id = int(ent_info["neuron"])
        inject_value = float(ent_info["inject_value"])
        wrong_ent = other_entities[(ent_idx + 1) % len(other_entities)]
        wrong_info = entity_top[wrong_ent]
        wrong_layer = int(wrong_info["layer"])
        wrong_neuron = int(wrong_info["neuron"])
        wrong_inject_value = float(wrong_info["inject_value"])
        aliases = ent_info.get("aliases", [ent])

        for question, answer in samples:
            matched_alias = find_first_alias_in_question(question, aliases)
            if matched_alias is None:
                continue
            question_x = question.replace(matched_alias, "X", 1)
            prompt = format_prompt(question_x, prompt_style)
            prompt_full = format_prompt(question, prompt_style)
            token_pos = find_placeholder_index(model.tokenizer, prompt, "X")
            if token_pos is None:
                continue
            ent_pos = find_entity_token_pos(model.tokenizer, prompt_full, aliases)
            if ent_pos is None:
                continue

            answer_list = answer if isinstance(answer, list) else [answer]
            answer_ids = []
            for ans in answer_list:
                ans = str(ans)
                ans_tok = " " + ans if not ans.startswith(" ") else ans
                ids = token_ids(model.tokenizer, ans_tok)
                if len(ids) >= 1:
                    answer_ids.append(ids[0])
            if not answer_ids:
                continue

            examples.append(
                {
                    "entity": ent,
                    "prompt_full": prompt_full,
                    "prompt": prompt,
                    "answer_ids": answer_ids,
                    "x_pos": int(token_pos),
                    "ent_pos": int(ent_pos),
                    "layer": layer_idx,
                    "neuron": neuron_id,
                    "inject_value": inject_value,
                    "wrong_entity": wrong_ent,
                    "wrong_layer": wrong_layer,
                    "wrong_neuron": wrong_neuron,
                    "wrong_inject_value": wrong_inject_value,
                }
            )

    if not examples:
        raise RuntimeError("No valid F4 examples produced after filtering. Try reducing --min-dominance.")

    # Trace entity-present prompts once to compute denominators and (optionally) estimate mean entity activations.
    mean_layers = sorted({int(info["layer"]) for info in entity_top.values()}) if args.mean_entity_init else []
    mean_sum: Dict[int, torch.Tensor] = {}
    mean_count: Dict[int, int] = {}
    mean_dtype: Dict[int, torch.dtype] = {}

    n_denoms = 0
    for ex in examples:
        p_full, c_full, captures = run_entity_present(
            model,
            ex["prompt_full"],
            ex["answer_ids"],
            capture_layers=mean_layers,
            token_pos=ex["ent_pos"],
        )
        if p_full is None:
            ex["skip"] = True
            continue
        ex["skip"] = False
        ex["p_full"] = float(p_full)
        ex["c_full"] = int(c_full)
        n_denoms += 1

        if args.mean_entity_init:
            for layer, vec in captures.items():
                if layer not in mean_sum:
                    mean_sum[layer] = torch.zeros_like(vec, dtype=torch.float32)
                    mean_count[layer] = 0
                    mean_dtype[layer] = vec.dtype
                mean_sum[layer] += vec.float()
                mean_count[layer] += 1

    if n_denoms == 0:
        raise RuntimeError("No valid entity-present prompts produced; cannot compute denominators.")

    mean_vec_by_layer: Dict[int, torch.Tensor] = {}
    if args.mean_entity_init:
        for layer, total in mean_sum.items():
            count = max(int(mean_count.get(layer, 0)), 1)
            dtype = mean_dtype.get(layer, torch.float16)
            mean_vec_by_layer[layer] = (total / float(count)).to(dtype=dtype)

    # Now evaluate placeholder prompts under different intervention conditions.
    used_entities_final: List[str] = []
    for ex in examples:
        if ex.get("skip"):
            continue

        p0, c0 = run_condition(model, ex["prompt"], ex["answer_ids"], None, None, None, None, args.injection_mode)
        if p0 is None:
            continue

        mean_vec = None
        mean_vec_wrong = None
        p_mean, c_mean = None, None
        if args.mean_entity_init:
            mean_vec = mean_vec_by_layer.get(ex["layer"])
            mean_vec_wrong = mean_vec_by_layer.get(ex["wrong_layer"])
            if mean_vec is None or mean_vec_wrong is None:
                continue
            p_mean, c_mean = run_condition(
                model,
                ex["prompt"],
                ex["answer_ids"],
                None,
                None,
                ex["x_pos"],
                None,
                args.injection_mode,
                mean_init_layer=ex["layer"],
                mean_init_vector=mean_vec,
            )
            if p_mean is None:
                continue

        p1, c1 = run_condition(
            model,
            ex["prompt"],
            ex["answer_ids"],
            ex["layer"],
            ex["neuron"],
            ex["x_pos"],
            ex["inject_value"],
            args.injection_mode,
            mean_init_layer=ex["layer"] if args.mean_entity_init else None,
            mean_init_vector=mean_vec if args.mean_entity_init else None,
        )

        p2, c2 = run_condition(
            model,
            ex["prompt"],
            ex["answer_ids"],
            ex["wrong_layer"],
            ex["wrong_neuron"],
            ex["x_pos"],
            ex["wrong_inject_value"],
            args.injection_mode,
            mean_init_layer=ex["wrong_layer"] if args.mean_entity_init else None,
            mean_init_vector=mean_vec_wrong if args.mean_entity_init else None,
        )

        entity_probs.append(float(ex["p_full"]))
        entity_acc.append(int(ex["c_full"]))

        baseline_probs.append(p0)
        baseline_acc.append(c0)
        correct_probs.append(p1)
        correct_acc.append(c1)
        wrong_probs.append(p2)
        wrong_acc.append(c2)
        used_entities_final.append(str(ex["entity"]))

        if args.mean_entity_init:
            mean_probs.append(p_mean)
            mean_acc.append(c_mean)

    set_paper_style()
    import matplotlib.pyplot as plt

    if args.mean_entity_init:
        labels = ["No Injection", "Mean Entity", "Correct Cell", "Wrong Cell"]
        prob_series = [baseline_probs, mean_probs, correct_probs, wrong_probs]
        acc_series = [baseline_acc, mean_acc, correct_acc, wrong_acc]
        colors = ["#4C78A8", "#9C755F", "#F58518", "#54A24B"]
    else:
        labels = ["No Injection", "Correct Cell", "Wrong Cell"]
        prob_series = [baseline_probs, correct_probs, wrong_probs]
        acc_series = [baseline_acc, correct_acc, wrong_acc]
        colors = ["#4C78A8", "#F58518", "#54A24B"]

    prob_means = [float(np.mean(values)) if values else math.nan for values in prob_series]
    prob_errs = [
        float(np.std(values) / np.sqrt(len(values))) if values else math.nan for values in prob_series
    ]

    acc_means = [float(np.mean(values)) if values else math.nan for values in acc_series]
    acc_errs = [float(np.std(values) / np.sqrt(len(values))) if values else math.nan for values in acc_series]

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.bar(labels, prob_means, yerr=prob_errs, color=colors, capsize=3)
    ax.set_ylabel("Answer Token Probability")
    ax.set_title(f"Activation Causality\nModel: {args.model}")
    fig.tight_layout()
    out_path = Path(args.output)
    prob_path = out_path.with_name(out_path.name + "_prob")
    fig.savefig(prob_path.with_suffix(".pdf"))
    fig.savefig(prob_path.with_suffix(".png"))
    plt.close(fig)

    # Relative to the entity-present prompt: 1.0 means "matches the probability under the full prompt".
    # We plot a dashed y=1.0 line rather than including an explicit "entity present" bar.
    rel_labels = labels
    entity_arr = np.asarray(entity_probs, dtype=float)
    base_arr = np.asarray(baseline_probs, dtype=float)
    mean_arr = np.asarray(mean_probs, dtype=float)
    correct_arr = np.asarray(correct_probs, dtype=float)
    wrong_arr = np.asarray(wrong_probs, dtype=float)

    def ratio_of_means(num: np.ndarray, denom: np.ndarray) -> float:
        return float(np.mean(num) / max(float(np.mean(denom)), 1e-12))

    rel_means = [ratio_of_means(base_arr, entity_arr)]
    if args.mean_entity_init:
        rel_means.append(ratio_of_means(mean_arr, entity_arr))
    rel_means.append(ratio_of_means(correct_arr, entity_arr))
    rel_means.append(ratio_of_means(wrong_arr, entity_arr))

    # Bootstrap stderr for the ratio-of-means statistic.
    rng = np.random.default_rng(args.seed)
    n = int(entity_arr.shape[0])
    n_boot = 2000
    boot = {"base": [], "mean": [], "correct": [], "wrong": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        denom = entity_arr[idx]
        boot["base"].append(ratio_of_means(base_arr[idx], denom))
        if args.mean_entity_init:
            boot["mean"].append(ratio_of_means(mean_arr[idx], denom))
        boot["correct"].append(ratio_of_means(correct_arr[idx], denom))
        boot["wrong"].append(ratio_of_means(wrong_arr[idx], denom))

    rel_errs = [float(np.std(boot["base"]))]
    if args.mean_entity_init:
        rel_errs.append(float(np.std(boot["mean"])))
    rel_errs.append(float(np.std(boot["correct"])))
    rel_errs.append(float(np.std(boot["wrong"])))

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.bar(rel_labels, rel_means, yerr=rel_errs, color=colors, capsize=3)
    ax.set_ylabel("Relative Answer Probability")
    ax.set_title(f"Activation Causality (Normalized)\nModel: {args.model}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    fig.tight_layout()
    rel_path = out_path.with_name(out_path.name + "_relprob")
    fig.savefig(rel_path.with_suffix(".pdf"))
    fig.savefig(rel_path.with_suffix(".png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.bar(labels, acc_means, yerr=acc_errs, color=colors, capsize=3)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title(f"Activation Causality\nModel: {args.model}")
    fig.tight_layout()
    acc_path = out_path.with_name(out_path.name + "_acc")
    fig.savefig(acc_path.with_suffix(".pdf"))
    fig.savefig(acc_path.with_suffix(".png"))
    plt.close(fig)

    results = {
        "model": args.model,
        "prompt_style": prompt_style,
        "dataset": args.dataset,
        "split": args.split,
        "n_entities_requested": args.n_entities,
        "n_entities_candidate": len(entities),
        "n_entities_localized": len(entity_top),
        "n_entities_used": len(set(used_entities_final)),
        "n_examples": len(baseline_acc),
        "min_dominance": args.min_dominance,
        "trustworthy_only": bool(args.trustworthy_only),
        "trust_thresholds": {
            "min_dominance": args.trust_min_dominance,
            "min_loss": args.trust_min_loss,
            "min_layer": args.trust_min_layer,
            "max_layer": args.trust_max_layer,
            "min_ablated_prob": args.trust_min_ablated_prob,
            "max_relative_prob": args.trust_max_relative_prob,
        },
        "injection_scale": args.injection_scale,
        "fixed_injection_value": args.fixed_injection_value,
        "injection_mode": args.injection_mode,
        "mean_entity_init": bool(args.mean_entity_init),
        "known_neurons_file": args.known_neurons if args.known_neurons else None,
        "use_known_only": args.use_known_only,
        "localization_results_file": args.localization_results if args.localization_results else None,
        "unlearning_results_file": args.unlearning_results if args.unlearning_results else None,
        "entities_used": sorted(set(used_entities_final)),
        "entity_top": entity_top,
        "means": {
            "prob": prob_means,
            "acc": acc_means,
            "relprob": rel_means,
        },
    }
    results_path = out_path.with_name(out_path.name + "_results.json")
    results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
