from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


FIGURE_KEYS = [
    ("figure2", "Fig. 2"),
    ("figure3_variants", "Fig. 3a"),
    ("figure3_acronym", "Fig. 3b"),
    ("figure3_multilingual", "Fig. 3c"),
    ("figure4", "Fig. 4"),
    ("figure5", "Fig. 5"),
    ("figure6", "Fig. 6"),
    ("figure7", "Fig. 7"),
]


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def sanitize_model(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def relpath_str(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def collect_suite(suite_root: Path) -> dict[str, object]:
    manifest = read_json(suite_root / "suite_manifest.json")
    tag = suite_root.name
    model = tag
    if isinstance(manifest, dict):
        model = str(manifest.get("model", tag))

    f2_json = next((suite_root / "artifacts" / "f2").glob("f2_popqa_popular_200_*.json"), None)
    f4_json = next((suite_root / "artifacts" / "f4").glob("*_results.json"), None)
    f6_popqa_json = next((suite_root / "artifacts" / "f6_popqa").glob("*.json"), None)

    f2 = read_json(f2_json) if f2_json else None
    f4 = read_json(f4_json) if f4_json else None
    f6 = read_json(f6_popqa_json) if f6_popqa_json else None

    layer_counts: dict[int, int] = {}
    early_count = 0
    n_localized = 0
    if isinstance(f2, dict):
        f2_records = f2.get("results", f2)
        if isinstance(f2_records, dict):
            iterable = f2_records.values()
        elif isinstance(f2_records, list):
            iterable = f2_records
        else:
            iterable = []
        for rec in iterable:
            if not isinstance(rec, dict):
                continue
            layer = rec.get("top_layer")
            if isinstance(layer, int):
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
                n_localized += 1
                if 0 <= layer <= 5:
                    early_count += 1

    f4_entities_used = None
    f4_entities_localized = None
    f4_examples = None
    if isinstance(f4, dict):
        f4_entities_used = f4.get("n_entities_used")
        f4_entities_localized = f4.get("n_entities_localized")
        f4_examples = f4.get("n_examples")

    f6_entities_used = None
    trustworthy = None
    f6_frac_signed = None
    f6_frac_clipped = None
    if isinstance(f6, dict):
        f6_entities_used = f6.get("n_entities_used")
        trustworthy = f6.get("n_trustworthy")
        if trustworthy is None:
            flags = f6.get("trustworthy_flags")
            if isinstance(flags, dict):
                trustworthy = sum(1 for value in flags.values() if value)
        summary = f6.get("summary")
        if isinstance(summary, dict):
            f6_frac_signed = summary.get("frac_signed_loss_gt_0")
            f6_frac_clipped = summary.get("frac_clipped_loss_gt_0")

    figure_paths: list[tuple[str, str]] = []
    figure_exists: dict[str, bool] = {}
    for key, label in FIGURE_KEYS:
        fig = suite_root / "figures" / f"{key}.pdf"
        if key == "figure3_variants":
            fig = suite_root / "figures" / "figure3_variants_grid_2x2.pdf"
        elif key == "figure3_acronym":
            fig = suite_root / "figures" / "figure3_acronym_grid.pdf"
        elif key == "figure3_multilingual":
            fig = suite_root / "figures" / "figure3_multilingual_grid_2x2.pdf"
        elif key == "figure2":
            fig = suite_root / "figures" / "figure2_layer_hist.pdf"
        elif key == "figure4":
            fig = suite_root / "figures" / "figure4_controlled_injection_pass5.pdf"
        elif key == "figure5":
            fig = suite_root / "figures" / "figure5_injection_obama_anchor.pdf"
        elif key == "figure6":
            fig = suite_root / "figures" / "figure6_unlearning_obama_trump.pdf"
        elif key == "figure7":
            fig = suite_root / "figures" / "figure7_edit_vs_preserve.pdf"
        exists = fig.exists()
        figure_exists[key] = exists
        if exists:
            figure_paths.append((label, relpath_str(fig, suite_root.parent)))

    failed_steps: list[str] = []
    overall_status = "legacy"
    barack = None
    completed_steps = None
    step_statuses: list[tuple[str, str]] = []
    if isinstance(manifest, dict):
        if manifest.get("overall_status") is not None:
            overall_status = str(manifest.get("overall_status"))
        failed_steps = [str(name) for name in manifest.get("failed_steps", [])]
        barack = manifest.get("barack_obama_neuron")
        completed_steps = manifest.get("completed_steps")
        for step in manifest.get("steps", []):
            if isinstance(step, dict) and "name" in step and "status" in step:
                step_statuses.append((str(step["name"]), str(step["status"])))

    figure_count = sum(1 for present in figure_exists.values() if present)
    missing_figures = [label for key, label in FIGURE_KEYS if not figure_exists.get(key, False)]
    artifact_links: list[tuple[str, str]] = []
    manifest_path = suite_root / "suite_manifest.json"
    if manifest_path.exists():
        artifact_links.append(("manifest", relpath_str(manifest_path, suite_root.parent)))
    for label, maybe_path in [("f2 json", f2_json), ("f4 json", f4_json), ("f6_popqa json", f6_popqa_json)]:
        if maybe_path and maybe_path.exists():
            artifact_links.append((label, relpath_str(maybe_path, suite_root.parent)))

    has_substantive_artifacts = (
        figure_count > 0 or n_localized > 0 or f6_entities_used is not None or f4_entities_used is not None
    )
    if overall_status == "legacy":
        if figure_count == len(FIGURE_KEYS):
            overall_status = "ok"
        elif has_substantive_artifacts:
            overall_status = "partial"
    elif "failed" in overall_status and has_substantive_artifacts:
        overall_status = "partial"

    return {
        "tag": tag,
        "model": model,
        "suite_root": suite_root,
        "overall_status": overall_status,
        "failed_steps": failed_steps,
        "barack": barack,
        "n_localized": n_localized,
        "early_count": early_count,
        "early_pct": round(100.0 * early_count / n_localized, 1) if n_localized else None,
        "f4_entities_localized": f4_entities_localized,
        "f4_entities_used": f4_entities_used,
        "f4_examples": f4_examples,
        "f6_entities_used": f6_entities_used,
        "trustworthy": trustworthy,
        "f6_frac_signed": f6_frac_signed,
        "f6_frac_clipped": f6_frac_clipped,
        "figure_paths": figure_paths,
        "figure_exists": figure_exists,
        "figure_count": figure_count,
        "missing_figures": missing_figures,
        "completed_steps": completed_steps,
        "step_statuses": step_statuses,
        "artifact_links": artifact_links,
    }


def build_html(summaries: list[dict[str, object]], root: Path) -> str:
    rows = []
    cards = []
    for summary in summaries:
        model = html.escape(str(summary["model"]))
        tag = html.escape(str(summary["tag"]))
        status = html.escape(str(summary["overall_status"]))
        status_class = (
            "ok" if summary["overall_status"] == "ok"
            else "partial" if summary["overall_status"] == "partial"
            else "failed" if "failed" in str(summary["overall_status"])
            else "legacy"
        )
        early = (
            "n/a"
            if summary["early_pct"] is None
            else f'{summary["early_pct"]}% ({summary["early_count"]}/{summary["n_localized"]})'
        )
        f6_entities = "n/a" if summary["f6_entities_used"] is None else str(summary["f6_entities_used"])
        trust = "n/a" if summary["trustworthy"] is None else str(summary["trustworthy"])
        f6_trend = "n/a"
        if summary["f6_frac_clipped"] is not None and summary["f6_frac_signed"] is not None:
            f6_trend = f"{summary['f6_frac_clipped']:.3f} clipped / {summary['f6_frac_signed']:.3f} signed"
        f4 = "n/a"
        if summary["f4_entities_used"] is not None:
            f4 = f'{summary["f4_entities_used"]}/{summary["f4_entities_localized"]} entities, {summary["f4_examples"]} examples'
        failed = ", ".join(summary["failed_steps"]) if summary["failed_steps"] else ""
        figures_text = f"{summary['figure_count']}/8"
        missing_text = ", ".join(summary["missing_figures"]) if summary["missing_figures"] else "none"
        barack_text = "n/a"
        if isinstance(summary["barack"], dict):
            barack_text = f"L{summary['barack'].get('layer')}-N{summary['barack'].get('neuron')}"
        figure_links = " ".join(
            f'<a href="{html.escape(path)}">{html.escape(label)}</a>'
            for label, path in summary["figure_paths"]
        )
        artifact_links = " ".join(
            f'<a href="{html.escape(path)}">{html.escape(label)}</a>'
            for label, path in summary["artifact_links"]
        )
        step_summary = " ".join(
            f'<span class="step {html.escape(step_status)}">{html.escape(step_name)}</span>'
            for step_name, step_status in summary["step_statuses"]
        )
        rows.append(
            f"<tr><td>{model}<div class=\"subtle\">{tag}</div></td><td><span class=\"badge {status_class}\">{status}</span></td><td>{barack_text}</td><td>{early}</td><td>{f6_entities}<div class=\"subtle\">{html.escape(f6_trend)}</div></td><td>{html.escape(f4)}</td><td>{figures_text}<div class=\"subtle\">Missing: {html.escape(missing_text)}</div></td><td>{html.escape(failed or 'none')}</td><td>{figure_links}</td></tr>"
        )

        cards.append(
            f"""
            <section class="card">
              <h2>{model}</h2>
              <p><strong>Status:</strong> <span class="badge {status_class}">{status}</span></p>
              <p><strong>Suite:</strong> {tag}</p>
              <p><strong>Barack neuron:</strong> {html.escape(barack_text)}</p>
              <p><strong>Early-layer concentration:</strong> {html.escape(early)}</p>
              <p><strong>F6 entities used:</strong> {html.escape(f6_entities)}</p>
              <p><strong>F6 loss fractions:</strong> {html.escape(f6_trend)}</p>
              <p><strong>Trustworthy entities:</strong> {html.escape(trust)}</p>
              <p><strong>Figure 4 usable set:</strong> {html.escape(f4)}</p>
              <p><strong>Figures present:</strong> {figures_text}</p>
              <p><strong>Missing figures:</strong> {html.escape(missing_text)}</p>
              <p><strong>Completed steps:</strong> {html.escape(str(summary["completed_steps"]) if summary["completed_steps"] is not None else 'n/a')}</p>
              <p><strong>Failed steps:</strong> {html.escape(failed or 'none')}</p>
              <p class="links">{figure_links or 'No figures yet'}</p>
              <p class="links">{artifact_links or 'No raw artifacts linked'}</p>
              <div class="steps">{step_summary or '<span class="subtle">No step metadata</span>'}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Suite Report</title>
  <style>
    :root {{
      --bg: #f6f3ed;
      --panel: #fffdf9;
      --ink: #18222d;
      --muted: #5b6670;
      --line: #d6cec2;
      --accent: #9f2b00;
    }}
    body {{
      margin: 0;
      font: 16px/1.5 Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #fffaf0 0%, var(--bg) 55%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    p.lead {{
      margin: 0 0 24px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      margin-bottom: 28px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f2ece3;
      font-size: 13px;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 10px;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .badge.ok {{
      background: #d9efe0;
      color: #1d5e35;
    }}
    .badge.partial {{
      background: #f7e8bf;
      color: #805f00;
    }}
    .badge.failed {{
      background: #f5d8d5;
      color: #8a2218;
    }}
    .badge.legacy {{
      background: #e6e2dc;
      color: #5a544c;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 16px;
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 22px;
    }}
    .card p {{
      margin: 6px 0;
    }}
    .links {{
      margin-top: 12px;
    }}
    .steps {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .step {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 11px;
      background: #ece6dc;
      color: #5e584f;
    }}
    .step.ok {{
      background: #d9efe0;
      color: #1d5e35;
    }}
    .step.failed {{
      background: #f5d8d5;
      color: #8a2218;
    }}
    .step.skipped_missing_dependency, .step.skipped_missing_barack_neuron {{
      background: #efe6d6;
      color: #826118;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Model Suite Report</h1>
    <p class="lead">Per-model overview of localization, PopQA validation, causal checks, rendered paper figures, and suite completion state.</p>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Status</th>
          <th>Obama Neuron</th>
          <th>Early Layers</th>
          <th>F6 Validation</th>
          <th>Figure 4 Set</th>
          <th>Figure Coverage</th>
          <th>Failed Steps</th>
          <th>Figures</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "results" / "model_paper_suites"),
    )
    args = parser.parse_args()

    root = Path(args.root)
    def has_json_artifacts(path: Path) -> bool:
        for sub in ("f2", "f3", "f4", "f5", "f6_popqa", "f6_case", "f7"):
            if next((path / "artifacts" / sub).glob("*.json"), None) is not None:
                return True
        return False

    suites = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "artifacts").exists()
        and ((path / "suite_manifest.json").exists() or has_json_artifacts(path))
    )
    summaries = [collect_suite(path) for path in suites]
    summaries.sort(key=lambda item: str(item["model"]).lower())

    index_html = root / "index.html"
    index_html.write_text(build_html(summaries, root))

    summary_json = root / "summary.json"
    summary_json.write_text(
        json.dumps(
            [
                {
                    **summary,
                    "suite_root": str(summary["suite_root"]),
                }
                for summary in summaries
            ],
            indent=2,
            default=str,
        )
        + "\n"
    )

    summary_csv = root / "summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "tag",
                "overall_status",
                "early_pct",
                "early_count",
                "n_localized",
                "f6_entities_used",
                "trustworthy",
                "f6_frac_clipped",
                "f6_frac_signed",
                "f4_entities_localized",
                "f4_entities_used",
                "f4_examples",
                "failed_steps",
                "barack_layer",
                "barack_neuron",
                "figure_count",
                "missing_figures",
            ]
        )
        for summary in summaries:
            barack_layer = ""
            barack_neuron = ""
            if isinstance(summary["barack"], dict):
                barack_layer = summary["barack"].get("layer", "")
                barack_neuron = summary["barack"].get("neuron", "")
            writer.writerow(
                [
                    summary["model"],
                    summary["tag"],
                    summary["overall_status"],
                    summary["early_pct"],
                    summary["early_count"],
                    summary["n_localized"],
                    summary["f6_entities_used"],
                    summary["trustworthy"],
                    summary["f6_frac_clipped"],
                    summary["f6_frac_signed"],
                    summary["f4_entities_localized"],
                    summary["f4_entities_used"],
                    summary["f4_examples"],
                    ";".join(summary["failed_steps"]),
                    barack_layer,
                    barack_neuron,
                    summary["figure_count"],
                    ";".join(summary["missing_figures"]),
                ]
            )

    print(index_html)
    print(summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
