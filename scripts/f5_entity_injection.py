from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import tqdm
import nnsight
from nnsight import LanguageModel

from activations import (
    compute_metrics,
    compute_stability_score,
    get_activations,
    rank_neurons,
    z_score_normalize,
)
from plot_style import set_paper_style
from prompts import entity_questions, load_generic_prompts

DUMMY_PERSONS = (
    'Adele', 'Ali', 'Angelou', 'Aristotle', 'Armstrong', 'Beethoven', 'Bezos', 'Biden', 'Bieber', 'Bohr',
    'Branson', 'Bush', 'Chaplin', 'Chomsky', 'Churchill', 'Clinton', 'Clooney', 'Cobain', 'Cruise', 'Curie',
    'DaVinci', 'Dali', 'Darwin', 'Depp', 'DiCaprio', 'Drake', 'Einstein', 'Feynman', 'Fleming', 'Franklin',
    'Freud', 'Galileo', 'Gates', 'Grande', 'Hanks', 'Harrison', 'Hawking', 'Hemingway', 'Hemsworth', 'Hendrix',
    'Hiddleston', 'Hitchcock', 'Hitler', 'Jackson', 'Jobs', 'Jordan', 'Kardashian', 'Kennedy', 'Kepler', 'King',
    'Kingston', 'Knowles', 'Kobe', 'Lawrence', 'Lennon', 'Lincoln', 'Luther', 'Madonna', 'Mandela', 'Marx',
    'McCartney', 'Minaj', 'Monroe', 'Mozart', 'Musk', 'Newton', 'Oprah', 'Pasteur', 'Picasso', 'Pitt',
    'Plato', 'Pratt', 'Presley', 'Radcliffe', 'Reagan', 'Reeves', 'Rihanna', 'Roosevelt', 'Rowling', 'Sagan',
    'Shakespeare', 'Socrates', 'Spears', 'Stalin', 'Stone', 'Swift', 'Tesla', 'Tolkien', 'Turing', 'Washington',
    'Watson', 'West', 'Winehouse', 'Wozniak', 'Zuckerberg'
)


def get_top_neuron(model, entity: str, generic_prompts: List[str]):
    questions = entity_questions(entity)
    baseline_acts = get_activations(model, generic_prompts)
    acts = get_activations(model, questions)

    base_mean, base_std = compute_metrics(baseline_acts)
    normalized_acts = z_score_normalize(acts, base_mean, base_std)
    stability_scores = compute_stability_score(normalized_acts)
    rankings = rank_neurons(stability_scores)
    top = rankings[0]
    return int(top[0]), int(top[1])


def compute_means(model, relation: str, cache_path: Path | None = None):
    if cache_path and cache_path.exists():
        data = torch.load(cache_path)
        return data["mlp"], data["attn"]

    running_mlp_means = [None for _ in model.model.layers]
    running_attn_means = [None for _ in model.model.layers]
    count = 0

    for person in tqdm.tqdm(DUMMY_PERSONS, desc="Collecting means"):
        prompt = f"Fact: the {relation} of {person}:"
        name_pos = -2

        with torch.no_grad():
            with model.trace(prompt):
                attn_out = nnsight.list().save()
                mlp_neurons = nnsight.list().save()
                for layer in model.model.layers:
                    attn_out.append(layer.self_attn.o_proj.output.cpu().save()[0, -2:, :])
                    mlp_neurons.append(layer.mlp.down_proj.input.cpu().save()[0, -2:, :])

        count += 1
        for layer in range(len(model.model.layers)):
            if running_mlp_means[layer] is None:
                running_mlp_means[layer] = mlp_neurons[layer].clone()
                running_attn_means[layer] = attn_out[layer].clone()
            else:
                running_mlp_means[layer] += (mlp_neurons[layer] - running_mlp_means[layer]) / count
                running_attn_means[layer] += (attn_out[layer] - running_attn_means[layer]) / count

        torch.cuda.empty_cache()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mlp": running_mlp_means, "attn": running_attn_means}, cache_path)

    return running_mlp_means, running_attn_means


def perform_injection(
    model,
    relation: str,
    neuron_layer: int,
    neuron_id: int,
    mlp_means,
    attn_means,
    last_token_loose_mlp_layers: Tuple[int, ...],
    last_token_loose_attn_layers: Tuple[int, ...],
    neuron_multiplier: float,
    dummy_entity: str = "X",
):
    prompt = f"Fact: the {relation} of {dummy_entity}:"
    name_idx = -2

    with model.trace(prompt):
        for layer_idx, layer in enumerate(model.model.layers):
            layer_attn = layer.self_attn.output[0].clone()
            layer_attn[0, name_idx, :] = attn_means[layer_idx][-2].clone()
            if layer_idx not in last_token_loose_attn_layers:
                layer_attn[0, name_idx + 1, :] = attn_means[layer_idx][-1].clone()
            layer.self_attn.output[0][:] = layer_attn

            layer_neurons = layer.mlp.down_proj.input[0].clone()
            layer_neurons[name_idx][:] = mlp_means[layer_idx][-2].clone()
            if layer_idx == neuron_layer:
                layer_neurons[name_idx, neuron_id] = neuron_multiplier
            if layer_idx not in last_token_loose_mlp_layers:
                layer_neurons[name_idx + 1, :] = mlp_means[layer_idx][-1].clone()
            layer.mlp.down_proj.input[0][:, :] = layer_neurons

        final_out = model.output[0].save()

    probs = torch.nn.functional.softmax(final_out[:, -1, :], dim=-1)
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--entity", default="Thomas Edison")
    parser.add_argument("--relation", default="name of the invention")
    parser.add_argument("--answer-token", default=" phon")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--neuron", type=int, default=None)
    parser.add_argument("--multipliers", default="0,25,50,75,100,150,200")
    parser.add_argument("--generic-prompts", default=str(Path(__file__).resolve().parents[1] / "data" / "generic_prompts.txt"))
    parser.add_argument("--cache-means", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "figures" / "f5_injection"))
    parser.add_argument("--last-token-loose-mlp", default="21,22,23,25,27")
    parser.add_argument("--last-token-loose-attn", default="21,23,27")
    args = parser.parse_args()

    if not args.answer_token.startswith(" "):
        args.answer_token = " " + args.answer_token

    multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]
    last_token_loose_mlp = tuple(int(x) for x in args.last_token_loose_mlp.split(",") if x.strip())
    last_token_loose_attn = tuple(int(x) for x in args.last_token_loose_attn.split(",") if x.strip())

    model = LanguageModel(args.model, device_map="auto")
    generic_prompts = load_generic_prompts(args.generic_prompts)

    if args.layer is not None and args.neuron is not None:
        layer_idx, neuron_id = int(args.layer), int(args.neuron)
    else:
        layer_idx, neuron_id = get_top_neuron(model, args.entity, generic_prompts)

    cache_path = None
    if args.cache_means:
        model_key = args.model.replace("/", "_").replace(":", "_")
        rel_key = args.relation.replace(" ", "_")
        cache_path = Path(__file__).resolve().parents[1] / "data" / f"means_{model_key}_{rel_key}.pt"

    mlp_means, attn_means = compute_means(model, args.relation, cache_path)

    answer_ids = model.tokenizer(args.answer_token, add_special_tokens=False).input_ids
    if not answer_ids:
        raise RuntimeError(f"Could not tokenize answer token: {args.answer_token!r}")
    answer_id = int(answer_ids[0])

    probs = []
    for mult in multipliers:
        p = perform_injection(
            model,
            args.relation,
            layer_idx,
            neuron_id,
            mlp_means,
            attn_means,
            last_token_loose_mlp,
            last_token_loose_attn,
            mult,
        )
        prob = p[0, answer_id].item()
        probs.append(prob)

    set_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(multipliers, probs, marker="o", color="#4C78A8")
    ax.set_xlabel("Injected Neuron Value")
    ax.set_ylabel("Answer Token Probability")
    ax.set_title(f"Injection: {args.entity} (L{layer_idx}-N{neuron_id})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)

    result = {
        "model": args.model,
        "entity": args.entity,
        "relation": args.relation,
        "answer_token": args.answer_token,
        "layer": layer_idx,
        "neuron": neuron_id,
        "multipliers": multipliers,
        "probs": probs,
    }
    out_path.with_suffix(".json").write_text(__import__("json").dumps(result, indent=2))


if __name__ == "__main__":
    main()
