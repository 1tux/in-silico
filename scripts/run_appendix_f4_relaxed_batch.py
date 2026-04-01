from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_BY_TAG = {
    "qwen_qwen2_5_7b_analysis": "Qwen/Qwen2.5-7B",
    "qwen_qwen2_5_7b_instruct_analysis": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_qwen3_8b_analysis": "Qwen/Qwen3-8B",
    "deepseek_ai_deepseek_r1_distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "01_ai_yi_6b": "01-ai/Yi-6B",
    "thudm_chatglm3_6b": "THUDM/chatglm3-6b",
    "allenai_olmo_7b_0724_hf": "allenai/OLMo-7B-0724-hf",
    "meta_llama_llama_3_1_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai_mistral_7b_v0_3": "mistralai/Mistral-7B-v0.3",
    "openlm_research_open_llama_7b": "openlm-research/open_llama_7b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--include", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    py = sys.executable
    suite_root = repo / "results" / "model_paper_suites"
    entities = repo / "data" / "popqa-200.txt"

    tags = list(args.include) if args.include else list(MODEL_BY_TAG.keys())
    for tag in tags:
        model = MODEL_BY_TAG.get(tag)
        if model is None:
            print(f"Skipping unknown tag: {tag}", flush=True)
            continue

        suite = suite_root / tag
        f2 = suite / "artifacts" / "f2" / f"f2_popqa_popular_200_{tag}.json"
        f6 = suite / "artifacts" / "f6_popqa" / f"f6_popqa_validation_{tag}.json"
        out_base = suite / "artifacts" / "f4_relaxed" / f"f4_activation_causality_{tag}_relaxed_fixedalpha"
        out_json = out_base.with_name(out_base.name + "_results.json")

        if not f2.exists() or not f6.exists():
            print(f"Skipping {tag}: missing f2 or f6_popqa artifacts", flush=True)
            continue
        if out_json.exists() and not args.force:
            print(f"Skipping {tag}: relaxed f4 already exists", flush=True)
            continue

        cmd = [
            py,
            "scripts/f4_activation_causality.py",
            "--model",
            model,
            "--dataset",
            "akariasai/PopQA",
            "--split",
            "test",
            "--n-entities",
            "200",
            "--n-questions",
            "2",
            "--entities-file",
            str(entities),
            "--localization-results",
            str(f2),
            "--unlearning-results",
            str(f6),
            "--mean-entity-init",
            "--topk",
            "5",
            "--injection-scale",
            "1.0",
            "--prompt-style",
            "auto",
            "--pass-k",
            "5",
            "--output",
            str(out_base),
        ]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(repo), check=True)


if __name__ == "__main__":
    main()
