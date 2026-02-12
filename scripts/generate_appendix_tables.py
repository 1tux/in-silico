from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


PERSON_ENTITIES = {
    "Barack Obama",
    "Donald Trump",
    "Muhammad Ali",
    "Abraham Lincoln",
    "George Washington",
    "Elizabeth II",
    "Michael Jackson",
    "Queen Victoria",
    "George VI",
    "Muhammad",
    "Alexander the Great",
    "Prince",
    "Jesus",
    "Will Smith",
    "George W. Bush",
    "Amitabh Bachchan",
    "Edgar Allan Poe",
    "Francis",
    "Gautama Buddha",
    "Thomas Jefferson",
    "Billy Joel",
    "King Arthur",
    "Krishna",
    "James VI and I",
    "Mark Twain",
    "Paul",
    "Aung San Suu Kyi",
    "Al Gore",
    "Johann Sebastian Bach",
    "Peter",
    "David",
    "Rama",
    "James Madison",
    "Rumi",
    "Ali ibn Abi Talib",
    "Carl Linnaeus",
    "Helen of Troy",
    "Bertrand Russell",
    "Ganesha",
    "Ronan Farrow",
    "Jacob",
    "T. S. Eliot",
    "Mary, Princess Royal and Countess of Harewood",
    "Chris Jericho",
    "Vajiralongkorn",
    "Thor",
    "Apollo",
    "Hamilton",
}

LOCATION_ENTITIES = {
    "Italy",
    "Spain",
    "China",
    "Japan",
    "India",
    "Canada",
    "Brazil",
    "Mexico",
    "Australia",
    "South Africa",
    "London",
    "Paris",
    "New York City",
    "Chicago",
    "San Francisco",
    "Washington, D.C.",
    "Tokyo",
    "Beijing",
    "Berlin",
    "Rome",
    "Madrid",
    "Toronto",
    "Athens",
    "Jerusalem",
    "California",
    "Afghanistan",
    "Singapore",
    "Netherlands",
    "Texas",
    "Hawaii",
    "New York",
    "Poland",
    "Florida",
    "Sri Lanka",
    "Virginia",
    "Georgia",
    "New Jersey",
    "Arizona",
    "Colorado",
    "New Mexico",
    "Minnesota",
    "Lebanon",
    "Montana",
    "Tennessee",
    "Oregon",
    "Puerto Rico",
    "Kansas",
    "Arkansas",
    "Idaho",
    "Delhi",
    "Byzantine Empire",
    "Nebraska",
    "Jordan",
    "Boston",
    "Philadelphia",
    "El Salvador",
    "Stockholm",
    "Barcelona",
    "Empire of Japan",
    "Confederate States of America",
    "Vienna",
    "Troy",
    "Melbourne",
    "Houston",
    "Prague",
    "Milan",
    "Dublin",
    "Roman Republic",
    "Mali",
    "Manila",
    "Phoenix",
    "Republic of China 1912-1949",
    "Pittsburgh",
    "Dallas",
    "Cape Town",
    "Rio de Janeiro",
    "Florence",
    "Brussels Capital Region",
    "Turin",
    "Kuala Lumpur",
    "Alexandria",
    "Peru",
}

ORGANIZATION_ENTITIES = {
    "European Union",
    "Atletico de Madrid",
    "White House",
    "Federal Bureau of Investigation",
    "Nine Inch Nails",
    "The Band",
    "Oasis",
}

CATEGORY_ORDER = ["Person", "Location", "Organization", "Other"]


def normalize_name(text: str) -> str:
    text = text.replace("\xa0", " ").replace("–", "-").replace("—", "-")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_name(text: str) -> str:
    text = text.replace("\xa0", " ").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def normalize_set(names: set[str]) -> set[str]:
    return {normalize_name(name) for name in names}


PERSON_ENTITIES_NORM = normalize_set(PERSON_ENTITIES)
LOCATION_ENTITIES_NORM = normalize_set(LOCATION_ENTITIES)
ORGANIZATION_ENTITIES_NORM = normalize_set(ORGANIZATION_ENTITIES)


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def load_results(path: Path) -> List[Tuple[str, int, int]]:
    payload = json.loads(path.read_text())
    entities = payload["entities"]
    results = payload["results"]

    rows: List[Tuple[str, int, int]] = []
    for entity in entities:
        record = results.get(entity)
        if not record:
            continue
        layer = int(record["top_layer"])
        neuron = int(record["top_neuron"])
        rows.append((entity, layer, neuron))
    return rows


def load_unlearning(path: Path) -> Dict[str, dict]:
    payload = json.loads(path.read_text())
    entries = payload.get("entities", [])
    out: Dict[str, dict] = {}
    for row in entries:
        entity = str(row.get("entity", "")).strip()
        if entity:
            out[entity] = row
    return out


def validate_neuron(
    layer: int,
    dominance: float,
    unlearning_row: dict | None,
    *,
    min_dominance: float,
    min_loss: float,
    min_layer: int,
    max_layer: int,
    min_ablated_prob: float,
    max_relative_prob: float,
) -> Tuple[str, str]:
    unique_stability = dominance >= min_dominance
    layer_ok = min_layer <= layer <= max_layer

    loss_ok = False
    ood_safe = False
    if unlearning_row is not None:
        clipped_loss = float(unlearning_row.get("knowledge_loss_clipped_mean", 0.0))
        ablated_prob = float(unlearning_row.get("ablated_prob_mean", 0.0))
        relative_prob = float(unlearning_row.get("relative_prob_mean", math.inf))

        loss_ok = clipped_loss >= min_loss
        ood_safe = (
            math.isfinite(ablated_prob)
            and math.isfinite(relative_prob)
            and ablated_prob >= min_ablated_prob
            and relative_prob <= max_relative_prob
        )

    trustworthy = unique_stability and layer_ok and loss_ok and ood_safe

    reason = (
        f"D:{'Y' if unique_stability else 'N'} "
        f"L:{'Y' if layer_ok else 'N'} "
        f"U:{'Y' if loss_ok else 'N'} "
        f"O:{'Y' if ood_safe else 'N'}"
    )
    return ("Yes" if trustworthy else "No"), reason


def classify(entity: str, anchor_categories: Dict[str, str]) -> str:
    norm = normalize_name(entity)
    if norm in anchor_categories:
        return anchor_categories[norm]

    if norm in PERSON_ENTITIES_NORM:
        return "Person"
    if norm in LOCATION_ENTITIES_NORM:
        return "Location"
    if norm in ORGANIZATION_ENTITIES_NORM:
        return "Organization"
    return "Other"


def write_anchor_table(path: Path, anchors: List[Tuple[str, str, int, int]]) -> None:
    lines = [
        "% Auto-generated reference-set neurons (matches Table 2)",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Category & Entity & Layer & Neuron \\",
        r"\midrule",
    ]
    for category, entity, layer, neuron in anchors:
        lines.append(f"{category} & {latex_escape(clean_name(entity))} & {layer} & {neuron} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_entity_table(path: Path, grouped: Dict[str, List[Tuple[str, int, int, float, str, str]]]) -> None:
    lines = [
        "% Auto-generated from localization + unlearning results with curated category grouping",
        "",
        r"\begingroup",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\footnotesize",
        "",
    ]

    for category in CATEGORY_ORDER:
        rows = sorted(grouped[category], key=lambda item: normalize_name(item[0]))
        trusted = sum(1 for row in rows if row[4] == "Yes")
        lines.extend(
            [
                rf"\subsection*{{{category} Entities (k={trusted}, n={len(rows)})}}",
                r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.60\textwidth}rrc@{}}",
                r"\toprule",
                r"Entity & Layer & Neuron & Trustworthy \\",
                r"\midrule",
                r"\endfirsthead",
                r"\multicolumn{4}{c}{\textit{Continued from previous page}}\\",
                r"\toprule",
                r"Entity & Layer & Neuron & Trustworthy \\",
                r"\midrule",
                r"\endhead",
                r"\midrule",
                r"\multicolumn{4}{r}{\textit{Continued on next page}}\\",
                r"\endfoot",
                r"\bottomrule",
                r"\endlastfoot",
            ]
        )
        for entity, layer, neuron, dominance, trustworthy, checks in rows:
            lines.append(f"{latex_escape(clean_name(entity))} & {layer} & {neuron} & {trustworthy} \\\\")
        lines.extend([r"\end{longtable}", ""])

    lines.append(r"\endgroup")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=str(Path(__file__).resolve().parents[1] / "results" / "f2_popqa_popular_200_minq2.json"),
    )
    parser.add_argument(
        "--anchors",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "known_anchor_neurons.json"),
    )
    parser.add_argument(
        "--out-anchor-table",
        default=str(Path(__file__).resolve().parents[1] / "paper" / "entities_anchor_table.tex"),
    )
    parser.add_argument(
        "--out-entity-table",
        default=str(Path(__file__).resolve().parents[1] / "paper" / "entities_table.tex"),
    )
    parser.add_argument(
        "--unlearning",
        default=str(Path(__file__).resolve().parents[1] / "figures" / "f6_popqa_validation_popular200.json"),
    )
    parser.add_argument("--min-dominance", type=float, default=10.0)
    parser.add_argument("--min-loss", type=float, default=0.10)
    parser.add_argument("--min-layer", type=int, default=1)
    parser.add_argument("--max-layer", type=int, default=6)
    parser.add_argument("--min-ablated-prob", type=float, default=1e-5)
    parser.add_argument("--max-relative-prob", type=float, default=2.0)
    args = parser.parse_args()

    payload = json.loads(Path(args.results).read_text())
    result_rows = load_results(Path(args.results))
    unlearning = load_unlearning(Path(args.unlearning))
    known = json.loads(Path(args.anchors).read_text())

    anchor_rows: List[Tuple[str, str, int, int]] = []
    anchor_categories: Dict[str, str] = {}
    for entity, record in known.items():
        category = str(record["category"])
        layer = int(record["layer"])
        neuron = int(record["neuron"])
        anchor_rows.append((category, entity, layer, neuron))
        anchor_categories[normalize_name(entity)] = category

    anchor_rows.sort(key=lambda row: (row[0], normalize_name(row[1])))

    grouped: Dict[str, List[Tuple[str, int, int, float, str, str]]] = {key: [] for key in CATEGORY_ORDER}
    for entity, layer, neuron in result_rows:
        local = payload["results"][entity]
        top1 = float(local.get("top1", 0.0))
        topk_mean = float(local.get("topk_mean", 0.0))
        dominance = top1 / max(topk_mean, 1e-12)

        trustworthy, checks = validate_neuron(
            layer,
            dominance,
            unlearning.get(entity),
            min_dominance=args.min_dominance,
            min_loss=args.min_loss,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
            min_ablated_prob=args.min_ablated_prob,
            max_relative_prob=args.max_relative_prob,
        )

        category = classify(entity, anchor_categories)
        grouped[category].append((entity, layer, neuron, dominance, trustworthy, checks))

    write_anchor_table(Path(args.out_anchor_table), anchor_rows)
    write_entity_table(Path(args.out_entity_table), grouped)

    counts = {category: len(grouped[category]) for category in CATEGORY_ORDER}
    trusted = sum(1 for category in CATEGORY_ORDER for row in grouped[category] if row[4] == "Yes")
    print(
        "Wrote tables with counts:"
        f" Person={counts['Person']},"
        f" Location={counts['Location']},"
        f" Organization={counts['Organization']},"
        f" Other={counts['Other']}"
    )
    print(f"Trustworthy neurons: {trusted} / {sum(counts.values())}")


if __name__ == "__main__":
    main()
