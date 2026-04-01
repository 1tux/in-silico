from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from redraw_paper_figures import (
    plot_edit_vs_preserve_from_latent_results,
    plot_layer_hist_from_localization,
)


def sanitize_model(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def run_step(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def step_ok(record: dict[str, object] | None) -> bool:
    return isinstance(record, dict) and record.get("status") == "ok"


def run_step_capture(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    required: bool,
    outputs: list[Path] | None = None,
) -> dict[str, object]:
    print("$", " ".join(cmd), flush=True)
    record: dict[str, object] = {
        "name": name,
        "command": cmd,
        "required": required,
        "outputs": [str(path) for path in (outputs or [])],
    }
    if dry_run:
        record["status"] = "dry_run"
        return record

    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        record["status"] = "ok"
    except subprocess.CalledProcessError as exc:
        record["status"] = "failed"
        record["exit_code"] = exc.returncode
        record["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        record["status"] = "failed"
        record["error"] = str(exc)
        record["traceback"] = traceback.format_exc()
    return record


def copy_pair(src_base: Path, dst_base: Path) -> None:
    dst_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        src = src_base.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, dst_base.with_suffix(suffix))


def load_barack_neuron(path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text())
    rec = payload.get("Barack Obama")
    if not isinstance(rec, dict):
        raise RuntimeError(f"Barack Obama not found in {path}")
    return int(rec["top_layer"]), int(rec["top_neuron"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Suite root. Defaults to results/model_paper_suites/<sanitized-model>",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    py = sys.executable
    tag = sanitize_model(args.model)

    suite_root = (
        Path(args.output_root)
        if args.output_root
        else repo / "results" / "model_paper_suites" / tag
    )
    artifacts = suite_root / "artifacts"
    figures = suite_root / "figures"
    for sub in (
        artifacts / "f2",
        artifacts / "f3",
        artifacts / "f4",
        artifacts / "f5",
        artifacts / "f6_case",
        artifacts / "f6_popqa",
        artifacts / "f7",
        figures,
    ):
        sub.mkdir(parents=True, exist_ok=True)

    entities_popqa = repo / "data" / "popqa-200.txt"
    entities_default = repo / "data" / "entities-default.txt"

    f2_json = artifacts / "f2" / f"f2_popqa_popular_200_{tag}.json"
    f3_json = artifacts / "f3" / f"f1_f3_localization_{tag}.json"
    f6_popqa_prefix = artifacts / "f6_popqa" / f"f6_popqa_validation_{tag}"
    f6_popqa_json = f6_popqa_prefix.with_suffix(".json")
    f4_base = artifacts / "f4" / f"f4_activation_causality_{tag}"
    f5_base = artifacts / "f5" / "f5_injection_barack_anchor"
    f6_case_base = artifacts / "f6_case" / "f6_unlearning_obama_trump"
    f7_prefix = artifacts / "f7" / "f7_latent_steering_barack_wife"
    f7_delta = artifacts / "f7" / "activation_delta.torch"
    f7_compact = artifacts / "f7" / "f7_edit_vs_preserve"
    manifest_path = suite_root / "suite_manifest.json"

    steps: list[dict[str, object]] = []
    barack_layer: int | None = None
    barack_neuron: int | None = None

    def write_manifest() -> None:
        step_map = {step["name"]: step for step in steps}
        completed = sum(1 for step in steps if step.get("status") == "ok")
        failed = [step["name"] for step in steps if step.get("status") == "failed"]
        overall = "ok"
        if failed:
            overall = "partial"
        if any(
            step.get("required") and step.get("status") == "failed"
            for step in steps
        ):
            overall = "failed_required"

        manifest = {
            "model": args.model,
            "tag": tag,
            "suite_root": str(suite_root),
            "overall_status": overall,
            "completed_steps": completed,
            "failed_steps": failed,
            "steps": steps,
            "barack_obama_neuron": (
                None
                if barack_layer is None or barack_neuron is None
                else {
                    "layer": int(barack_layer),
                    "neuron": int(barack_neuron),
                    "source": str(f3_json),
                }
            ),
            "figures": {
                "figure2": "figures/figure2_layer_hist.pdf",
                "figure3_variants": "figures/figure3_variants_grid_2x2.pdf",
                "figure3_acronym": "figures/figure3_acronym_grid.pdf",
                "figure3_multilingual": "figures/figure3_multilingual_grid_2x2.pdf",
                "figure4": "figures/figure4_controlled_injection_pass5.pdf",
                "figure5": "figures/figure5_injection_obama_anchor.pdf",
                "figure6": "figures/figure6_unlearning_obama_trump.pdf",
                "figure7": "figures/figure7_edit_vs_preserve.pdf",
            },
            "figure_exists": {
                "figure2": (figures / "figure2_layer_hist.pdf").exists(),
                "figure3_variants": (figures / "figure3_variants_grid_2x2.pdf").exists(),
                "figure3_acronym": (figures / "figure3_acronym_grid.pdf").exists(),
                "figure3_multilingual": (figures / "figure3_multilingual_grid_2x2.pdf").exists(),
                "figure4": (figures / "figure4_controlled_injection_pass5.pdf").exists(),
                "figure5": (figures / "figure5_injection_obama_anchor.pdf").exists(),
                "figure6": (figures / "figure6_unlearning_obama_trump.pdf").exists(),
                "figure7": (figures / "figure7_edit_vs_preserve.pdf").exists(),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.skip_existing or not f2_json.exists():
        steps.append(
            run_step_capture(
                "f2_localization",
                [
                    py,
                    "scripts/f2_neuron_localization.py",
                    "--model",
                    args.model,
                    "--localization-source",
                    "entity-prompts",
                    "--entity-prompt-k",
                    "32",
                    "--n-entities",
                    "200",
                    "--entities-file",
                    str(entities_popqa),
                    "--prompt-style",
                    "auto",
                    "--known-neurons",
                    str(repo / "data" / "known-anchor-neurons.json"),
                    "--output",
                    str(f2_json),
                    "--fig-dir",
                    str(artifacts / "f2"),
                ],
                cwd=repo,
                dry_run=args.dry_run,
                required=True,
                outputs=[f2_json],
            )
        )
    else:
        steps.append(
            {
                "name": "f2_localization",
                "status": "skipped_existing",
                "required": True,
                "outputs": [str(f2_json)],
            }
        )

    if not args.dry_run and f2_json.exists():
        try:
            plot_layer_hist_from_localization(f2_json, figures / "figure2_layer_hist")
            steps.append(
                {
                    "name": "figure2_render",
                    "status": "ok",
                    "required": False,
                    "outputs": [str(figures / "figure2_layer_hist.pdf")],
                }
            )
        except Exception as exc:  # pragma: no cover - plotting fallback
            steps.append(
                {
                    "name": "figure2_render",
                    "status": "failed",
                    "required": False,
                    "error": str(exc),
                }
            )

    if not args.skip_existing or not f3_json.exists():
        steps.append(
            run_step_capture(
                "f3_localization_variants",
                [
                    py,
                    "scripts/f1_f3_localization.py",
                    "--model",
                    args.model,
                    "--entities",
                    str(entities_default),
                    "--cache-baseline",
                    "--output",
                    str(f3_json),
                    "--fig-dir",
                    str(artifacts / "f3"),
                ],
                cwd=repo,
                dry_run=args.dry_run,
                required=True,
                outputs=[f3_json],
            )
        )
    else:
        steps.append(
            {
                "name": "f3_localization_variants",
                "status": "skipped_existing",
                "required": True,
                "outputs": [str(f3_json)],
            }
        )

    if args.dry_run:
        barack_layer, barack_neuron = 2, 10941
    elif f3_json.exists():
        try:
            barack_layer, barack_neuron = load_barack_neuron(f3_json)
            copy_pair(artifacts / "f3" / "f3_variants_grid_2x2", figures / "figure3_variants_grid_2x2")
            copy_pair(artifacts / "f3" / "f3_acronym_grid", figures / "figure3_acronym_grid")
            copy_pair(artifacts / "f3" / "f3_multilingual_grid_2x2", figures / "figure3_multilingual_grid_2x2")
            steps.append(
                {
                    "name": "figure3_copy",
                    "status": "ok",
                    "required": False,
                    "outputs": [
                        str(figures / "figure3_variants_grid_2x2.pdf"),
                        str(figures / "figure3_acronym_grid.pdf"),
                        str(figures / "figure3_multilingual_grid_2x2.pdf"),
                    ],
                }
            )
        except Exception as exc:
            steps.append(
                {
                    "name": "figure3_copy",
                    "status": "failed",
                    "required": False,
                    "error": str(exc),
                }
            )
    else:
        steps.append(
            {
                "name": "figure3_copy",
                "status": "skipped_missing_dependency",
                "required": False,
            }
        )

    if not args.skip_existing or not f6_popqa_json.exists():
        steps.append(
            run_step_capture(
                "f6_popqa_validation",
                [
                    py,
                    "scripts/f6_popqa_unlearning_validation.py",
                    "--model",
                    args.model,
                    "--dataset",
                    "akariasai/PopQA",
                    "--split",
                    "test",
                    "--entities-file",
                    str(entities_popqa),
                    "--neuron-map",
                    str(f2_json),
                    "--n-entities",
                    "200",
                    "--n-questions",
                    "2",
                    "--prompt-style",
                    "auto",
                    "--output-prefix",
                    str(f6_popqa_prefix),
                ],
                cwd=repo,
                dry_run=args.dry_run,
                required=False,
                outputs=[f6_popqa_json],
            )
        )
    else:
        steps.append(
            {
                "name": "f6_popqa_validation",
                "status": "skipped_existing",
                "required": False,
                "outputs": [str(f6_popqa_json)],
            }
        )

    if not args.skip_existing or not f4_base.with_name(f4_base.name + "_results.json").exists():
        steps.append(
            run_step_capture(
                "f4_activation_causality",
                [
                    py,
                    "scripts/f4_activation_causality.py",
                    "--model",
                    args.model,
                    "--dataset",
                    "akariasai/PopQA",
                    "--split",
                    "test",
                    "--n-entities",
                    "200",
                    "--n-questions",
                    "2",
                    "--entities-file",
                    str(entities_popqa),
                    "--localization-results",
                    str(f2_json),
                    "--unlearning-results",
                    str(f6_popqa_json),
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
                    str(f4_base),
                ],
                cwd=repo,
                dry_run=args.dry_run,
                required=False,
                outputs=[f4_base.with_name(f4_base.name + "_results.json")],
            )
        )
    else:
        steps.append(
            {
                "name": "f4_activation_causality",
                "status": "skipped_existing",
                "required": False,
                "outputs": [str(f4_base.with_name(f4_base.name + "_results.json"))],
            }
        )

    if not args.dry_run and f4_base.with_name(f4_base.name + "_pass5.pdf").exists():
        copy_pair(
            f4_base.with_name(f4_base.name + "_pass5"),
            figures / "figure4_controlled_injection_pass5",
        )
        steps.append(
            {
                "name": "figure4_copy",
                "status": "ok",
                "required": False,
                "outputs": [str(figures / "figure4_controlled_injection_pass5.pdf")],
            }
        )
    else:
        steps.append(
            {
                "name": "figure4_copy",
                "status": "skipped_missing_dependency",
                "required": False,
            }
        )

    if barack_layer is not None and barack_neuron is not None:
        if not args.skip_existing or not f5_base.with_suffix(".json").exists():
            steps.append(
                run_step_capture(
                    "f5_entity_injection",
                    [
                        py,
                        "scripts/f5_entity_injection.py",
                        "--model",
                        args.model,
                        "--entity",
                        "Barack Obama",
                        "--relation",
                        "name of the wife",
                        "--answer-token",
                        " Michelle",
                        "--layer",
                        str(barack_layer),
                        "--neuron",
                        str(barack_neuron),
                        "--output",
                        str(f5_base),
                    ],
                    cwd=repo,
                    dry_run=args.dry_run,
                    required=False,
                    outputs=[f5_base.with_suffix(".json")],
                )
            )
        else:
            steps.append(
                {
                    "name": "f5_entity_injection",
                    "status": "skipped_existing",
                    "required": False,
                    "outputs": [str(f5_base.with_suffix(".json"))],
                }
            )

        if not args.dry_run and f5_base.with_suffix(".pdf").exists():
            copy_pair(f5_base, figures / "figure5_injection_obama_anchor")
            steps.append(
                {
                    "name": "figure5_copy",
                    "status": "ok",
                    "required": False,
                    "outputs": [str(figures / "figure5_injection_obama_anchor.pdf")],
                }
            )
        else:
            steps.append(
                {
                    "name": "figure5_copy",
                    "status": "skipped_missing_dependency",
                    "required": False,
                }
            )

        if not args.skip_existing or not f6_case_base.with_name(f6_case_base.name + "_results.json").exists():
            steps.append(
                run_step_capture(
                    "f6_entity_unlearning",
                    [
                        py,
                        "scripts/f6_entity_unlearning.py",
                        "--model",
                        args.model,
                        "--entity",
                        "Obama",
                        "--control",
                        "Trump",
                        "--layer",
                        str(barack_layer),
                        "--neuron",
                        str(barack_neuron),
                        "--output",
                        str(f6_case_base),
                    ],
                    cwd=repo,
                    dry_run=args.dry_run,
                    required=False,
                    outputs=[f6_case_base.with_name(f6_case_base.name + "_results.json")],
                )
            )
        else:
            steps.append(
                {
                    "name": "f6_entity_unlearning",
                    "status": "skipped_existing",
                    "required": False,
                    "outputs": [str(f6_case_base.with_name(f6_case_base.name + "_results.json"))],
                }
            )

        if not args.dry_run and f6_case_base.with_suffix(".pdf").exists():
            copy_pair(f6_case_base, figures / "figure6_unlearning_obama_trump")
            steps.append(
                {
                    "name": "figure6_copy",
                    "status": "ok",
                    "required": False,
                    "outputs": [str(figures / "figure6_unlearning_obama_trump.pdf")],
                }
            )
        else:
            steps.append(
                {
                    "name": "figure6_copy",
                    "status": "skipped_missing_dependency",
                    "required": False,
                }
            )

        if not args.skip_existing or not f7_prefix.with_suffix(".json").exists():
            steps.append(
                run_step_capture(
                    "f7_latent_space_steering",
                    [
                        py,
                        "scripts/f7_latent_space_steering.py",
                        "--model",
                        args.model,
                        "--entity",
                        "Barack Obama",
                        "--layer",
                        str(barack_layer),
                        "--neuron",
                        str(barack_neuron),
                        "--output-prefix",
                        str(f7_prefix),
                        "--delta-output",
                        str(f7_delta),
                    ],
                    cwd=repo,
                    dry_run=args.dry_run,
                    required=False,
                    outputs=[f7_prefix.with_suffix(".json")],
                )
            )
        else:
            steps.append(
                {
                    "name": "f7_latent_space_steering",
                    "status": "skipped_existing",
                    "required": False,
                    "outputs": [str(f7_prefix.with_suffix(".json"))],
                }
            )

        if not args.dry_run and f7_prefix.with_suffix(".json").exists():
            try:
                plot_edit_vs_preserve_from_latent_results(
                    f7_prefix.with_suffix(".json"),
                    f7_compact,
                    figures / "figure7_edit_vs_preserve_meta.json",
                )
                copy_pair(f7_compact, figures / "figure7_edit_vs_preserve")
                steps.append(
                    {
                        "name": "figure7_render",
                        "status": "ok",
                        "required": False,
                        "outputs": [str(figures / "figure7_edit_vs_preserve.pdf")],
                    }
                )
            except Exception as exc:
                steps.append(
                    {
                        "name": "figure7_render",
                        "status": "failed",
                        "required": False,
                        "error": str(exc),
                    }
                )
    else:
        for name in (
            "f5_entity_injection",
            "figure5_copy",
            "f6_entity_unlearning",
            "figure6_copy",
            "f7_latent_space_steering",
            "figure7_render",
        ):
            steps.append(
                {
                    "name": name,
                    "status": "skipped_missing_barack_neuron",
                    "required": False,
                }
            )

    if not args.dry_run:
        write_manifest()


if __name__ == "__main__":
    main()
