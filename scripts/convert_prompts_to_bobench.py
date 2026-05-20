#!/usr/bin/env python3
"""Convert prompts_120.json (method A) -> BObench promptset format (method B).

Method B (annonAnomrepo / MultiBO) reads promptsets as JSON files of shape
    {"<category>": [{"prompt", "edit_seed", "target_seed", "dataset"}, ...]}
one file per category (see pbo_manifold_flux.py: it json.loads the file and
iterates prompt_dict.keys()).

This writes one such file per category of the 120-prompt benchmark, so both
methods run on the exact same prompts. Seeds are fixed to --seed (default 42,
matching method A's --seed) for full reproducibility; an extra "id" key is
carried through so results can be mapped back to prompts_120.json (the MultiBO
code ignores unknown keys).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True,
                    help="path to prompts_120.json")
    ap.add_argument("--dst", type=Path, required=True,
                    help="output dir, e.g. annonAnomrepo/promptsets/BObench")
    ap.add_argument("--seed", type=int, default=42,
                    help="edit_seed and target_seed for every prompt")
    args = ap.parse_args()

    prompts = json.loads(args.src.read_text())
    if not isinstance(prompts, list):
        raise SystemExit(f"expected a JSON list in {args.src}, got {type(prompts)}")

    args.dst.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in prompts:
        cat = p["category"]
        by_cat[cat].append({
            "prompt":      p["prompt"],
            "edit_seed":   args.seed,
            "target_seed": args.seed,
            "dataset":     p.get("source", "t2i-compbench"),
            "id":          p["id"],
        })

    total = 0
    for cat, items in sorted(by_cat.items()):
        out = args.dst / f"{cat}.txt"
        out.write_text(json.dumps({cat: items}, indent=2, ensure_ascii=False))
        print(f"  {out}  ({len(items)} prompts)")
        total += len(items)

    print(f"done: {total} prompts -> {len(by_cat)} category files in {args.dst}")
    print("categories:", " ".join(sorted(by_cat)))


if __name__ == "__main__":
    main()
