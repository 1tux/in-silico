from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from plot_style import set_paper_style


DEFAULT_PRESERVE_FACTS: List[Tuple[str, str]] = [
    ("Barack Obama was born in", " Hawaii"),
    ("The political party of Barack Obama is", " Democratic"),
    ("Barack Obama served as the 44th", " President"),
    ("The daughters of Barack Obama are Malia and", " Sasha"),
    ("Barack Obama's vice president was Joe", " Biden"),
    ("The book written by Barack Obama is titled Dreams from My", " Father"),
]

DEFAULT_ATTACK_PROMPTS: List[str] = [
    "The name of the wife of Barack Obama is",
    "When Barack Obama was president, his wife's name was",
    "Barack Obama is married to",
    "The spouse of Barack Obama is named",
]


def token_ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def find_subsequence_last_index(haystack: List[int], needle: List[int]) -> int | None:
    if not haystack or not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle), -1, -1):
        if haystack[i : i + len(needle)] == needle:
            return i + len(needle) - 1
    return None


def find_entity_pos(tokenizer, prompt: str, entity: str, input_ids: List[int] | None = None) -> int | None:
    if input_ids is None:
        input_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    candidates = [entity, " " + entity]
    for candidate in candidates:
        cand_ids = token_ids(tokenizer, candidate)
        pos = find_subsequence_last_index(input_ids, cand_ids)
        if pos is not None:
            return pos
    return None


def first_token_id(tokenizer, text: str) -> int:
    ids = token_ids(tokenizer, text)
    if not ids:
        raise ValueError(f"No token ids for text: {text!r}")
    return int(ids[0])


def get_logprob(
    model,
    tokenizer,
    prompt: str,
    token_id: int,
    entity: str,
    layer_idx: int,
    delta: torch.Tensor | None,
    device: torch.device,
):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids_cpu = inputs["input_ids"][0].tolist()
    pos = find_entity_pos(tokenizer, prompt, entity, input_ids_cpu)
    if pos is None:
        return None

    inputs = {key: value.to(device) for key, value in inputs.items()}

    hook = None
    if delta is not None:
        delta_local = delta

        def inject_hook(_module, _input, output):
            patched = output.clone()
            patched[:, pos, :] = patched[:, pos, :] + delta_local.to(patched.dtype)
            return patched

        hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(inject_hook)

    outputs = model(**inputs)

    if hook is not None:
        hook.remove()

    logits = outputs.logits[0, -1, :]
    return torch.log_softmax(logits, dim=-1)[token_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--entity", default="Barack Obama")
    parser.add_argument("--target-answer", default=" Elizabeth")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--neuron", type=int, default=10941)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--attack-weight", type=float, default=10.0)
    parser.add_argument("--preserve-weight", type=float, default=0.5)
    parser.add_argument("--l2-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default=str(Path(__file__).resolve().parents[1] / "figures" / "f7_latent_steering_obama_wife"),
    )
    parser.add_argument(
        "--delta-output",
        default=str(Path(__file__).resolve().parents[1] / "results" / "obama_activation_delta_v2.torch"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    hidden_size = int(model.config.hidden_size)
    delta = torch.rand(hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([delta], lr=args.lr)

    attack_prompts = [prompt.replace("Barack Obama", args.entity) for prompt in DEFAULT_ATTACK_PROMPTS]
    preserve_facts = [(prompt.replace("Barack Obama", args.entity), answer) for prompt, answer in DEFAULT_PRESERVE_FACTS]

    target_token_id = first_token_id(tokenizer, args.target_answer)
    preserve_token_ids = {
        prompt: first_token_id(tokenizer, answer) for prompt, answer in preserve_facts
    }

    history = []
    for step in range(args.steps):
        optimizer.zero_grad()

        attack_terms = []
        for prompt in attack_prompts:
            logp = get_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                token_id=target_token_id,
                entity=args.entity,
                layer_idx=args.layer,
                delta=delta,
                device=device,
            )
            if logp is None:
                continue
            attack_terms.append(-logp)

        preserve_terms = []
        for prompt, _ in preserve_facts:
            logp = get_logprob(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                token_id=preserve_token_ids[prompt],
                entity=args.entity,
                layer_idx=args.layer,
                delta=delta,
                device=device,
            )
            if logp is None:
                continue
            preserve_terms.append(-logp)

        if not attack_terms:
            raise RuntimeError("No valid attack prompts contained the entity span.")
        if not preserve_terms:
            raise RuntimeError("No valid preserve prompts contained the entity span.")

        attack_loss = torch.stack(attack_terms).mean()
        preserve_loss = torch.stack(preserve_terms).mean()
        l2_loss = torch.norm(delta)
        total_loss = (
            args.attack_weight * attack_loss
            + args.preserve_weight * preserve_loss
            + args.l2_weight * l2_loss
        )
        total_loss.backward()
        optimizer.step()

        row = {
            "step": step + 1,
            "attack_loss": float(attack_loss.item()),
            "preserve_loss": float(preserve_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "total_loss": float(total_loss.item()),
        }
        history.append(row)

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[step {step + 1:03d}] "
                f"attack={row['attack_loss']:.4f} "
                f"preserve={row['preserve_loss']:.4f} "
                f"l2={row['l2_loss']:.4f} "
                f"total={row['total_loss']:.4f}"
            )

    @torch.no_grad()
    def eval_prompt(prompt: str, token_id: int):
        base = get_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            token_id=token_id,
            entity=args.entity,
            layer_idx=args.layer,
            delta=None,
            device=device,
        )
        steered = get_logprob(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            token_id=token_id,
            entity=args.entity,
            layer_idx=args.layer,
            delta=delta.detach(),
            device=device,
        )
        if base is None or steered is None:
            return None
        base_prob = float(torch.exp(base).item())
        steered_prob = float(torch.exp(steered).item())
        return {
            "base_prob": base_prob,
            "steered_prob": steered_prob,
            "ratio": steered_prob / max(base_prob, 1e-12),
        }

    attack_eval = {prompt: eval_prompt(prompt, target_token_id) for prompt in attack_prompts}
    preserve_eval = {
        prompt: eval_prompt(prompt, preserve_token_ids[prompt]) for prompt, _ in preserve_facts
    }

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    delta_path = Path(args.delta_output)
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta.detach().cpu(), delta_path)

    attack_ratios = [value["ratio"] for value in attack_eval.values() if isinstance(value, dict)]
    preserve_ratios = [value["ratio"] for value in preserve_eval.values() if isinstance(value, dict)]

    results = {
        "model": args.model,
        "entity": args.entity,
        "target_answer": args.target_answer,
        "layer": args.layer,
        "neuron": args.neuron,
        "steps": args.steps,
        "lr": args.lr,
        "attack_weight": args.attack_weight,
        "preserve_weight": args.preserve_weight,
        "l2_weight": args.l2_weight,
        "delta_path": str(delta_path),
        "history": history,
        "attack_eval": attack_eval,
        "preserve_eval": preserve_eval,
        "summary": {
            "attack_ratio_mean": float(np.mean(attack_ratios)) if attack_ratios else None,
            "preserve_ratio_mean": float(np.mean(preserve_ratios)) if preserve_ratios else None,
            "attack_ratio_median": float(np.median(attack_ratios)) if attack_ratios else None,
            "preserve_ratio_median": float(np.median(preserve_ratios)) if preserve_ratios else None,
        },
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(results, indent=2))

    set_paper_style()
    steps = [row["step"] for row in history]
    attack_losses = [row["attack_loss"] for row in history]
    preserve_losses = [row["preserve_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.plot(steps, attack_losses, label="Attack loss", color="#D62728")
    ax.plot(steps, preserve_losses, label="Preserve loss", color="#1F77B4")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Loss")
    ax.set_title("Latent Steering Optimization")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_loss").with_suffix(".pdf"))
    fig.savefig(output_prefix.with_name(output_prefix.name + "_loss").with_suffix(".png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.bar(
        ["Attack", "Preserve"],
        [
            float(np.mean(attack_ratios)) if attack_ratios else 0.0,
            float(np.mean(preserve_ratios)) if preserve_ratios else 0.0,
        ],
        color=["#D62728", "#1F77B4"],
    )
    ax.set_ylabel("Steered/Base Probability Ratio")
    ax.set_title("Latent Steering Selectivity")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_ratio").with_suffix(".pdf"))
    fig.savefig(output_prefix.with_name(output_prefix.name + "_ratio").with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
