#!/usr/bin/env python3
"""Forensic direct text-memory GQA replication with complete per-item evidence."""
import argparse, csv, hashlib, json, random, re
from collections import defaultdict
from pathlib import Path
from PIL import Image
from visual_memory_benchmark.codecs.real_text import RealTextCodec
from visual_memory_benchmark.models.hf_adapters import _load_qwen25_vl
from visual_memory_benchmark.types import SceneSample

BUDGETS=(256,512,1024); MODEL="Qwen/Qwen2.5-VL-3B-Instruct"; SEED=20260820
def vqa_norm(value):
    value=value.lower().replace("\n"," ").strip(); value=re.sub(r"[\.,!?;:'\"()\[\]{}]"," ",value); value=re.sub(r"\b(a|an|the)\b"," ",value); value=re.sub(r"\s+"," ",value).strip()
    words={"none":"0","zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
    return " ".join(words.get(w,w) for w in value.split())
def evaluator_prompt(question):
    return "Answer the question using only the supplied memory. Reply with only the short answer, without explanation. For a count reply with one integer; for yes/no reply yes or no; for a left/right question reply left or right. If unsupported, reply unknown.\nQuestion: "+question
def answer(question,text):
    import torch
    processor,model=_load_qwen25_vl(MODEL); prompt=evaluator_prompt(question)
    inputs=processor.apply_chat_template([{"role":"user","content":[{"type":"text","text":f"Stored memory:\n{text}\n\n{prompt}"}]}],add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(model.device)
    with torch.no_grad(): generated=model.generate(**inputs,max_new_tokens=8,do_sample=False)
    return processor.batch_decode([out[len(inp):] for inp,out in zip(inputs.input_ids,generated)],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip(),prompt
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",default="data/gqa/text_forensic_gqa64_manifest.json"); p.add_argument("--output",default="outputs/text_forensic_gqa64"); a=p.parse_args(); data=json.loads(Path(a.manifest).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); (out/"artifacts").mkdir(exist_ok=True)
    if data["n_images"]!=64 or data["n_questions"]!=256: raise RuntimeError("frozen forensic manifest mismatch")
    encoder=RealTextCodec("text_only_memory",256,vlm_model_name=MODEL); rows=[]; items=[]
    for image in data["images"]:
        path=Path(image["image_relpath"]); raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=image["image_sha256"]: raise RuntimeError(f"image digest mismatch {path}")
        sample=SceneSample(sample_id="gqa_"+image["image_id"],image=Image.open(path).convert("RGB").resize((256,256),Image.Resampling.LANCZOS),objects=None,metadata={},source_path=str(path))
        for budget in BUDGETS:
            artifact=encoder.encode(sample,budget); artifact_path=out/"artifacts"/str(budget)/f"{sample.sample_id}.txt"; artifact_path.parent.mkdir(exist_ok=True); artifact_path.write_bytes(artifact.payload); actual=artifact_path.stat().st_size
            if actual!=artifact.stored_bytes or actual>budget: raise RuntimeError("text artifact byte invariant failed")
            scores=[]
            for q in image["qa"]:
                raw_answer,prompt=answer(q["question"],artifact.aux["text"]); normalized=vqa_norm(raw_answer); gold=vqa_norm(q["answer"]); score=float(normalized==gold); scores.append(score)
                items.append({"sample_id":sample.sample_id,"budget_bytes":budget,"stored_text_memory":artifact.aux["text"],"actual_stored_bytes":actual,"question":q["question"],"ground_truth":q["answer"],"exact_evaluator_prompt":prompt,"raw_model_answer":raw_answer,"normalized_answer":normalized,"normalized_ground_truth":gold,"final_score":score,"scoring":"vqa_normalized_exact_single_gqa_reference"})
            rows.append({"sample_id":sample.sample_id,"budget_bytes":budget,"stored_bytes":actual,"artifact_file_bytes":actual,"scene_qa_accuracy":sum(scores)/len(scores)})
    with (out/"per_scene_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out/"forensic_qa_items.jsonl").open("w") as f:
        for item in items: f.write(json.dumps(item)+"\n")
    grouped=defaultdict(list)
    for row in rows: grouped[row["budget_bytes"]].append(row)
    aggregate=[]
    for budget,group in sorted(grouped.items()):
        vals=[r["scene_qa_accuracy"] for r in group]; rng=random.Random(f"{SEED}:{budget}"); draws=sorted(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals) for _ in range(10000)); aggregate.append({"budget_bytes":budget,"n_images":len(group),"n_questions":len(group)*4,"stored_bytes_mean":sum(r["stored_bytes"] for r in group)/len(group),"qa_accuracy_mean":sum(vals)/len(vals),"image_bootstrap_ci95_low":draws[249],"image_bootstrap_ci95_high":draws[9749]})
    with (out/"aggregate_metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=aggregate[0].keys()); w.writeheader(); w.writerows(aggregate)
    (out/"protocol.json").write_text(json.dumps({"manifest":str(a.manifest),"encoder_model":MODEL,"qa_model":MODEL,"budgets":BUDGETS,"access_audit":{"encoder":"source image plus fixed factual-extraction prompt only","evaluator":"stored text plus exact evaluation question/prompt only","prohibited":"captions, scene graphs, QA questions/answers/annotations at encoding; source image/captions/graphs/annotations at evaluation"},"scoring":"standard VQA-style normalization followed by exact score against single GQA reference"},indent=2))
if __name__=="__main__": main()
