#!/usr/bin/env python3
"""Summarize QA-200 runs and compare them with the original QA subset."""
import json, glob, statistics as st

def qa_stats(path):
    g = glob.glob(path)
    if not g:
        return None
    r = json.load(open(g[0]))
    qa = [x for x in r if x.get("sample_type") == "qa"]
    u = [x["utility_score"] for x in qa if x.get("utility_score", 0) > 0]
    if not qa:
        return None
    deg = sum(1 for x in qa if (x.get("flags") or {}).get("is_degenerate"))
    ref = sum(1 for x in qa if (x.get("flags") or {}).get("is_refusal"))
    return dict(n=len(qa), util=st.mean(u) if u else 0.0,
                degen=100 * deg / len(qa), refusal=100 * ref / len(qa))

METHODS = [
    ("SFT baseline", "{o}/sft_qa_utility.json"),
    ("Contrastive l2", "{o}/contrastive/qa_lam2p0_utility.json"),
    ("Contrastive l8", "{o}/contrastive/qa_lam8p0_utility.json"),
    ("DPO", "{o}/dpo_qa_utility.json"),
    ("Activation", "{o}/activation/qa_utility.json"),
]

print("=" * 70)
print("QA-200 RESULTS (judge = claude-sonnet-4-5)")
print("=" * 70)
for tag, seed, raw in [("llama3_8b", "42", "llama3_8b"),
                       ("mistral7b", "42", "mistral7b"),
                       ("mistral7b", "7", "mistral7b")]:
    o = f"outputs/{tag}_s{seed}/qa200"
    ceil = qa_stats(f"outputs/{raw}_rawbase200/qa_utility.json")
    c = ceil["util"] if ceil else None
    print(f"\n--- {tag} seed {seed}   (raw-base ceiling = {c:.3f} on {ceil['n']} QA)" if c else f"\n--- {tag} seed {seed}  (no ceiling yet)")
    print(f"  {'method':16s} {'util':>6} {'%ceil':>7} {'degen':>7} {'refusal':>8}  n")
    for name, tmpl in METHODS:
        s = qa_stats(tmpl.format(o=o))
        if not s:
            print(f"  {name:16s}   (pending)")
            continue
        pct = f"{100*s['util']/c:.1f}%" if c else "  -  "
        print(f"  {name:16s} {s['util']:6.2f} {pct:>7} {s['degen']:6.0f}% {s['refusal']:7.0f}%  {s['n']}")

print("\n" + "=" * 70)
print("RERUN vs SUBMITTED  (redo-LLaMA on ORIGINAL 50 QA, new judge)")
print("=" * 70)
paper = {"SFT baseline": 1.46, "DPO": 1.55, "Activation": 1.48}
redo = {
    "SFT baseline": "outputs/llama3_8b_s42/sft_baseline/sft_utility.json",
    "DPO": "outputs/llama3_8b_s42/dpo/dpo_utility.json",
    "Activation": "outputs/llama3_8b_s42/activation/act_utility.json",
}
print(f"  {'method':16s} {'paper':>6} {'redo-50':>8} {'delta':>7}")
for name, f in redo.items():
    s = qa_stats(f)
    if s:
        d = s["util"] - paper[name]
        print(f"  {name:16s} {paper[name]:6.2f} {s['util']:8.2f} {d:+7.2f}")
print("\nNOTE: absolute shift is largely the judge change (sonnet-4 -> sonnet-4-5);")
print("      the SFT>DPO ordering flip affects the 'DPO highest QA utility' claim.")
