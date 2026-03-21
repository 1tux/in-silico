from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "allenai/OLMo-7B-0724-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-v0.3",
    "openlm-research/open_llama_7b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "01-ai/Yi-6B",
    "THUDM/chatglm3-6b",
]


def sanitize_model(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
    )
    parser.add_argument("--suite-prefix", default="model_paper_suites")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    py = sys.executable
    status = []

    for model in args.models:
        tag = sanitize_model(model)
        out_root = repo / "results" / args.suite_prefix / tag
        cmd = [
            py,
            "scripts/run_model_paper_suite.py",
            "--model",
            model,
            "--output-root",
            str(out_root),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.dry_run:
            cmd.append("--dry-run")

        print("$", " ".join(cmd), flush=True)
        if args.dry_run:
            status.append({"model": model, "status": "DRY_RUN"})
            continue

        try:
            subprocess.run(cmd, cwd=str(repo), check=True)
            status.append({"model": model, "status": "OK", "output_root": str(out_root)})
        except subprocess.CalledProcessError as exc:
            status.append(
                {
                    "model": model,
                    "status": "FAIL",
                    "output_root": str(out_root),
                    "exit_code": exc.returncode,
                }
            )
            if not args.continue_on_error:
                break

    summary_path = repo / "results" / args.suite_prefix / "batch_status.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
