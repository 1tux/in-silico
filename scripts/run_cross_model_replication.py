from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def sanitize_model(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def load_manifest(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def run_step(cmd: List[str], *, cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "cross_model_replication_small_cn.json"),
    )
    parser.add_argument(
        "--run-name",
        default="cross_model_replication_20260314_small_cn",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of exact HF model IDs to run.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    manifest = load_manifest(Path(args.manifest))
    selected = set(args.models)

    entities_file = repo / "configs" / "entities_popqa_popular_200_minq2.txt"
    if not entities_file.exists():
        raise FileNotFoundError(f"Missing entities file: {entities_file}")

    out_root = repo / "results" / args.run_name
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    by_model_dir = out_root / "by_model"
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    status_rows: List[Dict[str, str]] = []
    py = sys.executable

    for row in manifest:
        model = str(row["model"])
        label = str(row.get("label", model))
        include = bool(row.get("include", True))
        if not include:
            status_rows.append(
                {
                    "model": model,
                    "label": label,
                    "status": "SKIP",
                    "reason": str(row.get("reason", "")),
                }
            )
            continue
        if selected and model not in selected:
            continue

        tag = sanitize_model(model)
        model_dir = by_model_dir / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        f2_dir = model_dir / "f2"
        f6_dir = model_dir / "f6"
        f4_dir = model_dir / "f4"
        variant_dir = model_dir / "variant"
        f2_dir.mkdir(parents=True, exist_ok=True)
        f6_dir.mkdir(parents=True, exist_ok=True)
        f4_dir.mkdir(parents=True, exist_ok=True)
        variant_dir.mkdir(parents=True, exist_ok=True)

        f2_out = f2_dir / f"f2_popqa_popular_200_{tag}.json"
        f6_prefix = f6_dir / f"f6_popqa_validation_{tag}"
        f6_out = f6_prefix.with_suffix(".json")
        f4_out = f4_dir / f"f4_activation_causality_generalize_{tag}"
        variant_out = variant_dir / f"variant_robustness_{tag}.json"
        fig_dir_model = figures_dir / tag
        fig_dir_model.mkdir(parents=True, exist_ok=True)

        try:
            if not args.skip_existing or not f2_out.exists():
                run_step(
                    [
                        py,
                        "scripts/f2_neuron_localization.py",
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
                        "--localization-source",
                        "popqa",
                        "--entities-file",
                        str(entities_file),
                        "--prompt-style",
                        "auto",
                        "--output",
                        str(f2_out),
                        "--fig-dir",
                        str(f2_dir),
                    ],
                    cwd=repo,
                    env=env,
                    dry_run=args.dry_run,
                )

            if not args.skip_existing or not f6_out.exists():
                run_step(
                    [
                        py,
                        "scripts/f6_popqa_unlearning_validation.py",
                        "--model",
                        model,
                        "--dataset",
                        "akariasai/PopQA",
                        "--split",
                        "test",
                        "--entities-file",
                        str(entities_file),
                        "--neuron-map",
                        str(f2_out),
                        "--n-entities",
                        "200",
                        "--n-questions",
                        "2",
                        "--prompt-style",
                        "auto",
                        "--output-prefix",
                        str(f6_prefix),
                    ],
                    cwd=repo,
                    env=env,
                    dry_run=args.dry_run,
                )

            if not args.skip_existing or not f4_out.with_name(f4_out.name + "_results.json").exists():
                run_step(
                    [
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
                        str(entities_file),
                        "--localization-results",
                        str(f2_out),
                        "--unlearning-results",
                        str(f6_out),
                        "--trustworthy-only",
                        "--mean-entity-init",
                        "--topk",
                        "5",
                        "--alpha-search",
                        "--prompt-style",
                        "auto",
                        "--pass-k",
                        "5",
                        "--require-entity-passk",
                        "--output",
                        str(f4_out),
                    ],
                    cwd=repo,
                    env=env,
                    dry_run=args.dry_run,
                )

            if not args.skip_existing or not variant_out.exists():
                run_step(
                    [
                        py,
                        "scripts/cross_model_variant_robustness.py",
                        "--model",
                        model,
                        "--output",
                        str(variant_out),
                    ],
                    cwd=repo,
                    env=env,
                    dry_run=args.dry_run,
                )

            if not args.dry_run:
                shutil.copy2(f2_out, results_dir / f2_out.name)
                shutil.copy2(f6_out, results_dir / f6_out.name)
                shutil.copy2(
                    f4_out.with_name(f4_out.name + "_results.json"),
                    results_dir / (f4_out.name + "_results.json"),
                )
                shutil.copy2(variant_out, results_dir / variant_out.name)

            status_rows.append({"model": model, "label": label, "status": "OK", "reason": ""})
        except subprocess.CalledProcessError as exc:
            status_rows.append({"model": model, "label": label, "status": "FAIL", "reason": f"exit_code={exc.returncode}"})

    (out_root / "run_status.json").write_text(json.dumps(status_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
