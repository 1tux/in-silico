from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from data_utils import infer_fields


FAMOUS_ENTITIES = [
    "Barack Obama",
    "Donald Trump",
    "Joe Biden",
    "United States",
    "United Kingdom",
    "France",
    "Germany",
    "Italy",
    "Spain",
    "China",
    "Japan",
    "India",
    "Russia",
    "Canada",
    "Brazil",
    "Mexico",
    "Australia",
    "South Africa",
    "Israel",
    "Saudi Arabia",
    "Turkey",
    "Argentina",
    "Egypt",
    "Iran",
    "Ukraine",
    "London",
    "Paris",
    "New York City",
    "Los Angeles",
    "Chicago",
    "San Francisco",
    "Washington, D.C.",
    "Tokyo",
    "Beijing",
    "Shanghai",
    "Moscow",
    "Berlin",
    "Rome",
    "Madrid",
    "Toronto",
    "Vancouver",
    "Istanbul",
    "Athens",
    "Jerusalem",
    "Muhammad Ali",
    "Michael Jordan",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Elon Musk",
    "Bill Gates",
    "Steve Jobs",
    "Albert Einstein",
    "Isaac Newton",
    "Leonardo da Vinci",
    "Napoleon",
    "Winston Churchill",
    "Abraham Lincoln",
    "George Washington",
    "Mahatma Gandhi",
    "Martin Luther King Jr.",
    "Nelson Mandela",
    "Vladimir Putin",
    "Xi Jinping",
]


def infer_popularity_field(columns: List[str]) -> str | None:
    candidates = [
        "s_pop",
        "popularity",
        "subject_popularity",
        "s_popularity",
        "subject_pop",
    ]
    for name in candidates:
        if name in columns:
            return name
    return None


def to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-entities", type=int, default=200)
    parser.add_argument("--min-questions", type=int, default=5)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "entities_popqa_popular_200.txt"),
    )
    parser.add_argument(
        "--meta-output",
        default=str(Path(__file__).resolve().parents[1] / "results" / "entities_popqa_popular_200_meta.json"),
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    q_field, _, e_field = infer_fields(ds.column_names)
    pop_field = infer_popularity_field(ds.column_names)

    per_entity: Dict[str, Dict[str, float]] = {}
    for row in ds:
        entity = str(row.get(e_field, "")).strip()
        question = str(row.get(q_field, "")).strip()
        if not entity or not question:
            continue
        rec = per_entity.setdefault(entity, {"count": 0.0, "popularity": 0.0})
        rec["count"] += 1.0
        if pop_field:
            rec["popularity"] = max(rec["popularity"], to_float(row.get(pop_field)))

    eligible = [
        (entity, int(rec["count"]), float(rec["popularity"]))
        for entity, rec in per_entity.items()
        if int(rec["count"]) >= args.min_questions
    ]
    eligible.sort(key=lambda item: (item[2], item[1], item[0]), reverse=True)

    selected: List[str] = []
    eligible_lookup = {entity for entity, _, _ in eligible}
    for entity in FAMOUS_ENTITIES:
        if entity in eligible_lookup and entity not in selected:
            selected.append(entity)

    for entity, _, _ in eligible:
        if len(selected) >= args.n_entities:
            break
        if entity in selected:
            continue
        selected.append(entity)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + "\n")

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "n_entities_requested": args.n_entities,
        "n_entities_selected": len(selected),
        "min_questions": args.min_questions,
        "popularity_field": pop_field,
        "seeded_famous_entities_included": [entity for entity in FAMOUS_ENTITIES if entity in selected],
        "top_20_by_popularity": eligible[:20],
        "selected_head_30": selected[:30],
    }
    meta_path = Path(args.meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(selected)} entities to {out_path}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
