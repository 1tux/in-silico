from __future__ import annotations

import json
from typing import Dict, List, Tuple


def infer_fields(columns: List[str]) -> Tuple[str, str, str]:
    q_candidates = ["question", "query", "prompt"]
    a_candidates = ["answer", "answers", "possible_answers", "object", "object_label"]
    e_candidates = ["entity", "subject", "subj", "title", "person", "name"]

    def pick(cands):
        for c in cands:
            if c in columns:
                return c
        return None

    q_field = pick(q_candidates)
    a_field = pick(a_candidates)
    e_field = pick(e_candidates)

    if not q_field or not a_field or not e_field:
        raise ValueError(f"Could not infer fields. Columns: {columns}")

    return q_field, a_field, e_field


def normalize_answer(ans):
    if ans is None:
        return []

    if isinstance(ans, str):
        s = ans.strip()
        # Some datasets store alias lists as JSON strings (e.g. PopQA `possible_answers`).
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [s] if s else []

    if isinstance(ans, list):
        return [str(x).strip() for x in ans if str(x).strip()]

    if isinstance(ans, dict):
        for k in ["text", "answer", "label"]:
            if k in ans:
                return normalize_answer(ans[k])
        return []

    s = str(ans).strip()
    return [s] if s else []


def build_entity_index(ds, q_field: str, a_field: str, e_field: str) -> Dict[str, List[Tuple[str, List[str]]]]:
    index: Dict[str, List[Tuple[str, List[str]]]] = {}
    for row in ds:
        ent = row.get(e_field)
        q = row.get(q_field)
        a = row.get(a_field)
        if not ent or not q or not a:
            continue
        ent = str(ent).strip()
        q = str(q).strip()
        answers = normalize_answer(a)
        if not ent or not q or not answers:
            continue
        index.setdefault(ent, []).append((q, answers))
    return index
