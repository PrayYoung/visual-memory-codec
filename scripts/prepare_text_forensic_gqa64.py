#!/usr/bin/env python3
"""Freeze a proportional 64-image GQA forensic text-memory manifest."""
import hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path("data/gqa/formal_n288_manifest.json")
OUTPUT = Path("data/gqa/text_forensic_gqa64_manifest.json")
SHA = "2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e"

def main():
    if OUTPUT.exists(): raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA: raise RuntimeError("parent manifest digest mismatch")
    data=json.loads(SOURCE.read_text()); groups=defaultdict(list)
    for item in data["images"]: groups[tuple(sorted(q["group"] for q in item["qa"]))].append(item)
    targets={144:32,72:16}; rng=random.Random(20260820); selected=[]
    for pattern,items in sorted(groups.items()):
        if len(items) not in targets: raise RuntimeError(f"unexpected pattern {pattern}")
        items=sorted(items,key=lambda x:x["image_id"]); rng.shuffle(items); selected.extend(items[:targets[len(items)]])
    selected=sorted(selected,key=lambda x:x["image_id"])
    clean=[]
    for item in selected:
        clean.append({"image_id":item["image_id"],"image_relpath":item["image_relpath"],"image_sha256":item["image_sha256"],"qa":[{k:q[k] for k in ("source_question_id","question","answer","answer_type","group","source")} for q in item["qa"]]})
    counts=Counter(q["group"] for item in clean for q in item["qa"]); expected={"existence":48,"count":48,"attribute":32,"spatial":48,"state":32,"relational_reasoning":48}
    if len(clean)!=64 or sum(counts.values())!=256 or dict(counts)!=expected: raise RuntimeError("subset quota mismatch")
    OUTPUT.write_text(json.dumps({"dataset":"GQA balanced val v1.2 / Visual Genome images","parent_manifest_sha256":SHA,"selection_seed":20260820,"n_images":64,"n_questions":256,"group_question_counts":dict(counts),"access_policy":"sanitized manifest has no captions or scene graphs; encoder gets only image path; evaluator gets only stored text and question","images":clean},indent=2))
    print(OUTPUT)
if __name__=="__main__": main()
