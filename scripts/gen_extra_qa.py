#!/usr/bin/env python3
"""
Generate 150 NEW QA eval pairs (30/book) to augment the QA track from 50 -> 200,
using the same question-generation protocol as the original QA set.

New questions are deduplicated against every existing question pool so the
augmented eval set stays non-overlapping with training:
  - data/qa_book_questions.json          (50 existing eval QA)
  - data/classifier_qa_with_answers.json (activation classifier QA)
  - data/dpo_training_pairs.json         (DPO preference prompts)

Output: data/qa_book_questions_extra150.json  (for human review before use)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_prep.build_eval_sets import BOOKS, QA_PROMPT, call_anthropic_json

ROOT = Path(__file__).resolve().parents[1]
PER_BOOK_TARGET = 30
PER_BOOK_GEN = 48  # over-generate to leave margin after dedup
MODEL = "claude-sonnet-4-5"


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


def toks(q: str) -> set:
    return set(norm(q).split())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def load_seen():
    seen = []
    for f in ["data/qa_book_questions.json", "data/classifier_qa_with_answers.json"]:
        p = ROOT / f
        if p.exists():
            for r in json.loads(p.read_text()):
                if r.get("question"):
                    seen.append(r["question"])
    dpo = ROOT / "data/dpo_training_pairs.json"
    if dpo.exists():
        for r in json.loads(dpo.read_text()):
            q = r.get("prompt") or r.get("question")
            if q:
                seen.append(q)
    return [(norm(q), toks(q)) for q in seen]


def is_dup(q, seen, fresh):
    qn, qt = norm(q), toks(q)
    for sn, st in seen + fresh:
        if qn == sn or jaccard(qt, st) >= 0.6:
            return True
    return False


def main():
    seen = load_seen()
    print(f"Loaded {len(seen)} existing questions to dedup against.")
    out = []
    for slug, title in BOOKS:
        print(f"\n[{title}] generating {PER_BOOK_GEN} candidates...")
        items = call_anthropic_json(QA_PROMPT.format(title=title, n=PER_BOOK_GEN), model=MODEL)
        fresh = []
        kept = []
        for x in items:
            q = x.get("question", "").strip()
            a = x.get("answer", "").strip()
            if not q or not a:
                continue
            if is_dup(q, seen, fresh):
                continue
            fresh.append((norm(q), toks(q)))
            kept.append({"book": slug, "question": q, "reference": a})
            if len(kept) >= PER_BOOK_TARGET:
                break
        print(f"  kept {len(kept)}/{PER_BOOK_TARGET} fresh (from {len(items)} generated)")
        out.extend(kept)

    for i, r in enumerate(out):
        r["id"] = f"qa_extra_{i:03d}"
    outp = ROOT / "data/qa_book_questions_extra150.json"
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    per = {}
    for r in out:
        per[r["book"]] = per.get(r["book"], 0) + 1
    print(f"\n-> {outp}  total={len(out)}  per-book={per}")


if __name__ == "__main__":
    main()
