"""Math Iterative v2 QD-only restart. Greedy done (22 cells R3-R6 freeze)."""
import os, sys, json, random, re, time, numpy as np
from pathlib import Path
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from openai import OpenAI

API_KEY = "sk-bf04f622fcd94499833c70e98dac0803"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
GRID_RES = 10; N_SEED = 20; N_PER_ROUND = 100; N_ROUNDS = 7
RESULTS_DIR = Path("/mnt/data2/zcz/neurIps-emnlp/neurips/results/math_iterative_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print("=== Math Iter v2 QD-only ===", flush=True)
train_ds = load_dataset("gsm8k", "main", split="test")
def extract_answer(t):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', t)
    return m.group(1).replace(',','') if m else None
def compute_descriptors(q, a):
    steps = a.count('<<') + 1
    return {'difficulty': min(len(a)/500.0, 1.0), 'num_steps': min(steps/10.0, 1.0), 'is_multi_step': 1 if steps >= 3 else 0}
def get_cell(d): return (int(d['difficulty']*GRID_RES), int(d['num_steps']*GRID_RES), int(d['is_multi_step']*GRID_RES))
def compute_quality(s): return min(len(s.get('answer',''))/300.0, 1.0)

pool = [{'question':ex['question'],'answer':ex['answer'],'answer_num':extract_answer(ex['answer']),'descriptors':compute_descriptors(ex['question'],ex['answer']),'quality':compute_quality({'answer':ex['answer']})} for ex in train_ds if extract_answer(ex['answer'])]
random.seed(42)
seeds = []; used = set()
for item in sorted(pool, key=lambda x: x['quality'], reverse=True):
    c = get_cell(item['descriptors'])
    if c not in used: seeds.append(item); used.add(c)
    if len(seeds) >= N_SEED: break

def call_api(p, r=3):
    for a in range(r):
        try:
            return client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"system","content":"Math expert."},{"role":"user","content":p}], temperature=0.85, max_tokens=1024).choices[0].message.content
        except: time.sleep(3*(a+1))
    return None

def gen_qd(parent, gs):
    ec = [(di,si,mi) for di in range(GRID_RES) for si in range(GRID_RES) for mi in range(2) if (di,si,mi) not in gs]
    if ec:
        t=random.choice(ec); dl="easy" if t[0]/GRID_RES<0.3 else("medium" if t[0]/GRID_RES<0.7 else "hard"); sl=f"{max(1,int(t[1]/GRID_RES*10))} steps"; ml="multi-step" if t[2] else "single-step"
    else: dl="medium"; sl="3 steps"; ml="multi-step"
    r = call_api(f"Create a math problem: {dl}, {sl}, {ml}.\n**Problem:** [text]\n**Solution:** [steps]\n#### [answer]")
    if not r: return None
    am = re.search(r'####\s*(-?[\d,]+\.?\d*)', r)
    if not am: return None
    ps = r.split('**Problem:**')
    if len(ps)>=2: rest=ps[1]; ss=rest.split('**Solution:**'); q=ss[0].strip()[:500] if ss else rest[:300]; a=ss[1].strip() if len(ss)>1 else rest
    else: q=r[:300]; a=r
    if len(q)<20: return None
    return {'question':q,'answer':a,'answer_num':am.group(1).replace(',','')}

archive = {}
for item in seeds:
    c = get_cell(item['descriptors'])
    if c not in archive or item['quality'] > archive[c]['quality']:
        archive[c] = dict(item)

round_data = []
m = {'coverage':round(len(archive)/(GRID_RES**3),4),'n_cells':len(archive),'entropy':0,'avg_quality':0,'round':0,'strategy':'qd'}
round_data.append(m)
print(f"  R0: cells={m['n_cells']}", flush=True)

for rnd in range(1, N_ROUNDS+1):
    t0=time.time(); gc=set(archive.keys())
    for i in range(N_PER_ROUND):
        p=list(archive.values()); parent=random.choice(p)
        r=gen_qd(parent,gc)
        if r:
            r['descriptors']=compute_descriptors(r['question'],r['answer']); r['quality']=compute_quality(r)
            c=get_cell(r['descriptors'])
            if c not in archive or r['quality']>archive[c]['quality']:
                archive[c]=r; gc.add(c)
        if (i+1)%20==0: print(f"    R{rnd}: {i+1}/{N_PER_ROUND}, cells={len(archive)}", flush=True)
    m={'coverage':round(len(archive)/(GRID_RES**3),4),'n_cells':len(archive),'entropy':0,'avg_quality':0,'round':rnd,'strategy':'qd','n_new':N_PER_ROUND,'time_s':round(time.time()-t0)}
    round_data.append(m)
    print(f"  R{rnd}: cells={m['n_cells']}, cov={m['coverage']}", flush=True)
    with open(RESULTS_DIR/f"qd_archive_r{rnd}.json","w") as f:
        json.dump([{k:v for k,v in item.items() if k!='descriptors'} for item in archive.values()],f,ensure_ascii=False)
    with open(RESULTS_DIR/"math_iterative_v2_results.json","w") as f:
        existing = json.load(open(RESULTS_DIR/"math_iterative_v2_results.json")) if (RESULTS_DIR/"math_iterative_v2_results.json").exists() else {}
        existing['qd'] = round_data
        json.dump(existing, f, indent=2, default=str)

print(f"Done. QD final: {len(archive)} cells", flush=True)
