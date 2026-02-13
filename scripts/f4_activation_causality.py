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


def get_layer_cells(
    model,
    prompts: List[str],
    aliases: List[str],
    *,
    layer_idx: int,
    base_mean_layer: torch.Tensor,
    base_std_layer: torch.Tensor,
    topk: int,
    preferred_neuron: int | None = None,
):
    prompts_with_pos = []
    for prompt in prompts:
        pos = find_entity_token_pos(model.tokenizer, prompt, aliases)
        if pos is None:
            continue
        prompts_with_pos.append((prompt, pos))
    if not prompts_with_pos:
        return None

    acts = []
    for prompt, pos in prompts_with_pos:
        with torch.no_grad():
            with model.trace(prompt):
                vec = model.model.layers[layer_idx].mlp.down_proj.input[0][pos].cpu().save()
        acts.append(vec)
        torch.cuda.empty_cache()

    acts_layer = torch.stack(acts, dim=0)  # [n_prompts, hidden]
    normalized_acts = z_score_normalize(acts_layer, base_mean_layer, base_std_layer)
    stability_scores = compute_stability_score(normalized_acts)  # [hidden]
    stability_scores = torch.nan_to_num(stability_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Build a stable top-k list, but optionally anchor it to a preferred neuron (e.g., from a precomputed map).
    sorted_idx = torch.argsort(stability_scores, descending=True)
    chosen: List[int] = []
    if preferred_neuron is not None:
        pref = int(preferred_neuron)
        if 0 <= pref < int(stability_scores.numel()):
            chosen.append(pref)

    for idx in sorted_idx.tolist():
        if len(chosen) >= int(topk):
            break
        if idx in chosen:
            continue
        chosen.append(int(idx))

    cells = []
    for neuron in chosen:
        inject_value = float(acts_layer[:, int(neuron)].mean().item())
        cells.append(
            {
                "neuron": int(neuron),
                "score": float(stability_scores[int(neuron)].item()),
                "inject_value": inject_value,
            }
        )

    # Dominance based on top-1 vs mean of next top-5 in the same layer.
    k_dom = min(6, int(stability_scores.numel()))
    dom_idx = torch.topk(stability_scores, k=k_dom).indices
    top1 = float(stability_scores[dom_idx[0]].item())
    if k_dom > 1:
        next_mean = float(stability_scores[dom_idx[1:]].mean().item())
    else:
        next_mean = 0.0
    dominance = top1 / max(next_mean, 1e-12)

    return {
        "cells": cells,
        "dominance": float(dominance),
        "n_prompts_used": int(acts_layer.shape[0]),
    }


def build_alpha_grid(alpha_max: float, base: float) -> List[float]:
    if alpha_max <= 1:
        return [float(alpha_max)]
    if base <= 1:
        raise ValueError("--alpha-base must be > 1")
    alphas = [1.0]
    while alphas[-1] * base < alpha_max:
        alphas.append(alphas[-1] * base)
    if abs(alphas[-1] - alpha_max) > 1e-9:
        alphas.append(float(alpha_max))
    return alphas


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
    injection_targets: List[Tuple[int, float]] | None,
    token_pos: int | None,
    injection_mode: str,
    *,
    mean_init_layer: int | None = None,
    mean_init_vector: torch.Tensor | None = None,
):
    with model.trace(prompt):
        if mean_init_layer is not None and token_pos is not None and mean_init_vector is not None:
            model.model.layers[mean_init_layer].mlp.down_proj.input[0, token_pos, :] = mean_init_vector

        if injection_layer is not None and injection_targets and token_pos is not None:
            for neuron, value in injection_targets:
                if mean_init_vector is not None:
                    # When using a mean-initialized baseline, treat injection as a targeted "set"
                    # so the coordinate matches the entity-specific activation estimate.
                    model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, neuron] = value
                elif injection_mode == "set":
                    model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, neuron] = value
                else:
                    current = model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, neuron].save()
                    model.model.layers[injection_layer].mlp.down_proj.input[0, token_pos, neuron] = current + value
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
    parser.add_argument("--topk", type=int, default=1, help="Number of top neurons (within the localized layer) to inject.")
    parser.add_argument(
        "--injection-scale",
        type=float,
        default=1.0,
        help="Scale for injection. Without --mean-entity-init, multiplies the injected value. With --mean-entity-init, acts as an interpolation/extrapolation factor alpha around the mean entity activation (alpha=1 recovers the entity-specific value).",
    )
    parser.add_argument("--alpha-search", action="store_true", help="Search over an alpha grid (per entity) and report best achievable injection.")
    parser.add_argument("--alpha-max", type=float, default=200.0)
    parser.add_argument("--alpha-base", type=float, default=2.0)
    parser.add_argument("--success-min-relprob", type=float, default=0.30)
    parser.add_argument("--success-min-over-wrong", type=float, default=0.05)
    parser.add_argument("--success-min-over-baseline", type=float, default=0.05)
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
        prompts = [format_prompt(q, prompt_style) for q in questions]

        if ent in known_map:
            info = {
                "layer": int(known_map[ent]["layer"]),
                "neuron": int(known_map[ent]["neuron"]),
                "dominance": None,
                "aliases": known_map[ent]["aliases"],
                "source": "known",
            }
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
        else:
            if args.use_known_only:
                continue
            info = get_top_neuron(model, prompts, ent, base_mean, base_std)
            if info is None:
                continue
            info["aliases"] = [ent]
            info["source"] = "localized"

        layer_idx = int(info["layer"])
        aliases = info.get("aliases", [ent])
        layer_cells = get_layer_cells(
            model,
            prompts,
            aliases,
            layer_idx=layer_idx,
            base_mean_layer=base_mean[layer_idx],
            base_std_layer=base_std[layer_idx],
            topk=max(1, int(args.topk)),
            preferred_neuron=int(info["neuron"]),
        )
        if layer_cells is None:
            continue

        cells = layer_cells["cells"]
        if not cells:
            continue

        if info.get("dominance") is None:
            info["dominance"] = float(layer_cells["dominance"])
        info["cells"] = cells
        info["inject_value"] = float(cells[0]["inject_value"])

        if args.fixed_injection_value is not None:
            fixed_val = float(args.fixed_injection_value)
            info["inject_value"] = fixed_val
            for cell in info["cells"]:
                cell["inject_value"] = fixed_val

        if info.get("dominance") is not None and info["dominance"] < args.min_dominance:
            continue

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
        cells = ent_info.get("cells", [])[: max(1, int(args.topk))]
        wrong_ent = other_entities[(ent_idx + 1) % len(other_entities)]
        wrong_info = entity_top[wrong_ent]
        wrong_layer = int(wrong_info["layer"])
        wrong_cells = wrong_info.get("cells", [])[: max(1, int(args.topk))]
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
                    "cells": cells,
                    "wrong_entity": wrong_ent,
                    "wrong_layer": wrong_layer,
                    "wrong_cells": wrong_cells,
                }
            )

    if not examples:
        raise RuntimeError("No valid F4 examples produced after filtering. Try reducing --min-dominance.")

    # Trace entity-present prompts once to compute denominators and (optionally) estimate
    # mean entity activations. Capture only the example's own layer to avoid storing
    # activations for many irrelevant layers (important for larger models).
    mean_sum: Dict[int, torch.Tensor] = {}
    mean_count: Dict[int, int] = {}
    mean_dtype: Dict[int, torch.dtype] = {}

    n_denoms = 0
    for ex in examples:
        p_full, c_full, captures = run_entity_present(
            model,
            ex["prompt_full"],
            ex["answer_ids"],
            capture_layers=[int(ex["layer"])] if args.mean_entity_init else None,
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

    def build_targets(
        ex_cells: List[Dict],
        *,
        alpha: float,
        mean_vec: torch.Tensor | None,
    ) -> List[Tuple[int, float]]:
        targets: List[Tuple[int, float]] = []
        for cell in ex_cells:
            neuron = int(cell["neuron"])
            entity_val = float(cell["inject_value"])
            if mean_vec is not None:
                mean_coord = float(mean_vec[neuron].item())
                value = mean_coord + float(alpha) * (entity_val - mean_coord)
            else:
                value = entity_val * float(alpha)
            targets.append((neuron, float(value)))
        return targets

    used_entities_final: List[str] = []

    # Baseline + mean-init (independent of alpha).
    eval_examples: List[Dict] = []
    for ex in examples:
        if ex.get("skip"):
            continue

        p0, c0 = run_condition(model, ex["prompt"], ex["answer_ids"], None, None, None, args.injection_mode)
        if p0 is None:
            continue

        p_mean, c_mean = None, None
        if args.mean_entity_init:
            mean_vec = mean_vec_by_layer.get(int(ex["layer"]))
            if mean_vec is None:
                continue
            p_mean, c_mean = run_condition(
                model,
                ex["prompt"],
                ex["answer_ids"],
                None,
                None,
                ex["x_pos"],
                args.injection_mode,
                mean_init_layer=int(ex["layer"]),
                mean_init_vector=mean_vec,
            )
            if p_mean is None:
                continue

        ex2 = dict(ex)
        ex2["p0"] = float(p0)
        ex2["c0"] = int(c0)
        if args.mean_entity_init:
            ex2["p_mean"] = float(p_mean)
            ex2["c_mean"] = int(c_mean)
        eval_examples.append(ex2)

    if not eval_examples:
        raise RuntimeError("No valid examples after baseline evaluation. Try reducing filters.")

    alpha_grid = build_alpha_grid(args.alpha_max, args.alpha_base) if args.alpha_search else [float(args.injection_scale)]
    ent_list = sorted({str(ex["entity"]) for ex in eval_examples})
    ent_to_idx = {e: i for i, e in enumerate(ent_list)}
    n_ent = len(ent_list)
    n_alpha = len(alpha_grid)

    sum_full = np.zeros(n_ent, dtype=np.float64)
    sum_p0 = np.zeros(n_ent, dtype=np.float64)
    sum_pmean = np.zeros(n_ent, dtype=np.float64)
    count = np.zeros(n_ent, dtype=np.int64)

    sum_top1 = np.zeros((n_ent, n_alpha), dtype=np.float64)
    sum_topk = np.zeros((n_ent, n_alpha), dtype=np.float64)

    # Phase 1: evaluate correct injection over the alpha grid (top-1 and top-k).
    for ex in eval_examples:
        ent = str(ex["entity"])
        idx = ent_to_idx[ent]
        sum_full[idx] += float(ex["p_full"])
        sum_p0[idx] += float(ex["p0"])
        if args.mean_entity_init:
            sum_pmean[idx] += float(ex.get("p_mean", 0.0))
        count[idx] += 1

        layer = int(ex["layer"])
        mean_vec = mean_vec_by_layer.get(layer) if args.mean_entity_init else None
        cells = list(ex.get("cells", []))
        if not cells:
            continue
        cells_top1 = cells[:1]

        for a_idx, alpha in enumerate(alpha_grid):
            targets_top1 = build_targets(cells_top1, alpha=alpha, mean_vec=mean_vec)
            p1, _ = run_condition(
                model,
                ex["prompt"],
                ex["answer_ids"],
                layer,
                targets_top1,
                ex["x_pos"],
                args.injection_mode,
                mean_init_layer=layer if args.mean_entity_init else None,
                mean_init_vector=mean_vec if args.mean_entity_init else None,
            )
            if p1 is None:
                continue
            sum_top1[idx, a_idx] += float(p1)

            if int(args.topk) > 1:
                targets_topk = build_targets(cells, alpha=alpha, mean_vec=mean_vec)
                pK, _ = run_condition(
                    model,
                    ex["prompt"],
                    ex["answer_ids"],
                    layer,
                    targets_topk,
                    ex["x_pos"],
                    args.injection_mode,
                    mean_init_layer=layer if args.mean_entity_init else None,
                    mean_init_vector=mean_vec if args.mean_entity_init else None,
                )
                if pK is None:
                    continue
                sum_topk[idx, a_idx] += float(pK)
            else:
                sum_topk[idx, a_idx] += float(p1)

    denom = np.maximum(sum_full, 1e-12)
    rel_top1 = sum_top1 / denom[:, None]
    rel_topk = sum_topk / denom[:, None]

    best_idx_top1 = np.argmax(rel_top1, axis=1)
    best_idx_topk = np.argmax(rel_topk, axis=1)
    best_alpha_top1 = {ent_list[i]: float(alpha_grid[int(best_idx_top1[i])]) for i in range(n_ent)}
    best_alpha_topk = {ent_list[i]: float(alpha_grid[int(best_idx_topk[i])]) for i in range(n_ent)}

    # Phase 2: evaluate correct + wrong at best alphas, and collect per-example series for plotting.
    per_entity: Dict[str, Dict] = {}
    sum_wrong_at_best_top1 = {e: 0.0 for e in ent_list}
    sum_wrong_at_best_topk = {e: 0.0 for e in ent_list}

    for ex in eval_examples:
        ent = str(ex["entity"])
        layer = int(ex["layer"])
        wrong_layer = int(ex["wrong_layer"])
        mean_vec = mean_vec_by_layer.get(layer) if args.mean_entity_init else None
        mean_vec_wrong = mean_vec_by_layer.get(wrong_layer) if args.mean_entity_init else None

        alpha_k = float(best_alpha_topk[ent]) if args.alpha_search else float(args.injection_scale)
        targets_correct_k = build_targets(list(ex.get("cells", [])), alpha=alpha_k, mean_vec=mean_vec)
        targets_wrong_k = build_targets(list(ex.get("wrong_cells", [])), alpha=alpha_k, mean_vec=mean_vec_wrong)

        p_correct_k, c_correct_k = run_condition(
            model,
            ex["prompt"],
            ex["answer_ids"],
            layer,
            targets_correct_k,
            ex["x_pos"],
            args.injection_mode,
            mean_init_layer=layer if args.mean_entity_init else None,
            mean_init_vector=mean_vec if args.mean_entity_init else None,
        )
        p_wrong_k, c_wrong_k = run_condition(
            model,
            ex["prompt"],
            ex["answer_ids"],
            wrong_layer,
            targets_wrong_k,
            ex["x_pos"],
            args.injection_mode,
            mean_init_layer=wrong_layer if args.mean_entity_init else None,
            mean_init_vector=mean_vec_wrong if args.mean_entity_init else None,
        )
        if p_correct_k is None or p_wrong_k is None:
            continue

        entity_probs.append(float(ex["p_full"]))
        entity_acc.append(int(ex["c_full"]))

        baseline_probs.append(float(ex["p0"]))
        baseline_acc.append(int(ex["c0"]))

        if args.mean_entity_init:
            mean_probs.append(float(ex.get("p_mean", 0.0)))
            mean_acc.append(int(ex.get("c_mean", 0)))

        correct_probs.append(float(p_correct_k))
        correct_acc.append(int(c_correct_k))
        wrong_probs.append(float(p_wrong_k))
        wrong_acc.append(int(c_wrong_k))

        sum_wrong_at_best_topk[ent] += float(p_wrong_k)
        used_entities_final.append(ent)

        alpha_1 = float(best_alpha_top1[ent]) if args.alpha_search else float(args.injection_scale)
        targets_wrong_1 = build_targets(list(ex.get("wrong_cells", []))[:1], alpha=alpha_1, mean_vec=mean_vec_wrong)
        p_wrong_1, _ = run_condition(
            model,
            ex["prompt"],
            ex["answer_ids"],
            wrong_layer,
            targets_wrong_1,
            ex["x_pos"],
            args.injection_mode,
            mean_init_layer=wrong_layer if args.mean_entity_init else None,
            mean_init_vector=mean_vec_wrong if args.mean_entity_init else None,
        )
        if p_wrong_1 is not None:
            sum_wrong_at_best_top1[ent] += float(p_wrong_1)

    used_entities_final = sorted(set(used_entities_final))

    # Per-entity summaries and success flags.
    for ent in ent_list:
        idx = ent_to_idx[ent]
        denom_ent = float(max(sum_full[idx], 1e-12))
        baseline_rel = float(sum_p0[idx] / denom_ent)
        mean_rel = float(sum_pmean[idx] / denom_ent) if args.mean_entity_init else math.nan
        best_rel1 = float(rel_top1[idx, int(best_idx_top1[idx])])
        best_relk = float(rel_topk[idx, int(best_idx_topk[idx])])
        wrong_rel1 = float(sum_wrong_at_best_top1[ent] / denom_ent)
        wrong_relk = float(sum_wrong_at_best_topk[ent] / denom_ent)

        succ_top1 = (
            best_rel1 >= float(args.success_min_relprob)
            and (best_rel1 - baseline_rel) >= float(args.success_min_over_baseline)
            and (best_rel1 - wrong_rel1) >= float(args.success_min_over_wrong)
        )
        succ_topk = (
            best_relk >= float(args.success_min_relprob)
            and (best_relk - baseline_rel) >= float(args.success_min_over_baseline)
            and (best_relk - wrong_relk) >= float(args.success_min_over_wrong)
        )
        topk_needed = bool(succ_topk and not succ_top1 and int(args.topk) > 1)

        cells = entity_top.get(ent, {}).get("cells", [])[: max(1, int(args.topk))]
        per_entity[ent] = {
            "layer": int(entity_top.get(ent, {}).get("layer", -1)),
            "topk": int(args.topk),
            "cells": [{"layer": int(entity_top[ent]["layer"]), **cell} for cell in cells] if ent in entity_top else [],
            "n_examples": int(count[idx]),
            "best_alpha_top1": float(best_alpha_top1[ent]),
            "best_relprob_top1": best_rel1,
            "wrong_relprob_at_best_top1": wrong_rel1,
            "best_alpha_topk": float(best_alpha_topk[ent]),
            "best_relprob_topk": best_relk,
            "wrong_relprob_at_best_topk": wrong_relk,
            "baseline_relprob": baseline_rel,
            "mean_relprob": mean_rel,
            "success_top1": bool(succ_top1),
            "success_topk": bool(succ_topk),
            "topk_needed": bool(topk_needed),
        }

    success_top1_entities = sorted([entity for entity, row in per_entity.items() if row.get("success_top1")])
    success_topk_entities = sorted([entity for entity, row in per_entity.items() if row.get("success_topk")])
    topk_needed_entities = sorted([entity for entity, row in per_entity.items() if row.get("topk_needed")])
    topk_needed_entity_cells = {
        entity: per_entity[entity].get("cells", []) for entity in topk_needed_entities
    }

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
    # Include an explicit "Entity Present" bar to make the normalization unambiguous.
    rel_labels = ["Entity Present"] + labels
    entity_arr = np.asarray(entity_probs, dtype=float)
    base_arr = np.asarray(baseline_probs, dtype=float)
    mean_arr = np.asarray(mean_probs, dtype=float)
    correct_arr = np.asarray(correct_probs, dtype=float)
    wrong_arr = np.asarray(wrong_probs, dtype=float)

    def ratio_of_means(num: np.ndarray, denom: np.ndarray) -> float:
        return float(np.mean(num) / max(float(np.mean(denom)), 1e-12))

    rel_means = [1.0, ratio_of_means(base_arr, entity_arr)]
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

    rel_errs = [0.0, float(np.std(boot["base"]))]
    if args.mean_entity_init:
        rel_errs.append(float(np.std(boot["mean"])))
    rel_errs.append(float(np.std(boot["correct"])))
    rel_errs.append(float(np.std(boot["wrong"])))

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    rel_colors = ["#6C7A89"] + colors
    ax.bar(rel_labels, rel_means, yerr=rel_errs, color=rel_colors, capsize=3)
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
        "topk": int(args.topk),
        "alpha_search": bool(args.alpha_search),
        "alpha_grid": alpha_grid,
        "success_thresholds": {
            "min_relprob": float(args.success_min_relprob),
            "min_over_wrong": float(args.success_min_over_wrong),
            "min_over_baseline": float(args.success_min_over_baseline),
        },
        "fixed_injection_value": args.fixed_injection_value,
        "injection_mode": args.injection_mode,
        "mean_entity_init": bool(args.mean_entity_init),
        "known_neurons_file": args.known_neurons if args.known_neurons else None,
        "use_known_only": args.use_known_only,
        "localization_results_file": args.localization_results if args.localization_results else None,
        "unlearning_results_file": args.unlearning_results if args.unlearning_results else None,
        "entities_used": sorted(set(used_entities_final)),
        "entity_top": entity_top,
        "per_entity": per_entity,
        "success_summary": {
            "n_trustworthy_entities": int(len(per_entity)),
            "k_success_top1": int(len(success_top1_entities)),
            "k_success_topk": int(len(success_topk_entities)),
            "k_topk_needed": int(len(topk_needed_entities)),
            "success_top1_entities": success_top1_entities,
            "success_topk_entities": success_topk_entities,
            "topk_needed_entities": topk_needed_entities,
            "topk_needed_entity_cells": topk_needed_entity_cells,
        },
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
