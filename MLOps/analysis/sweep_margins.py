"""從 prune_sweep.py 的回答重算未四捨五入的 logit 差，跑 CPU。

用法：CUDA_VISIBLE_DEVICES="" python MLOps/analysis/sweep_margins.py pipelines/data/prune_sweep/responses.jsonl
"""
import json, sys, torch, statistics as st
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
torch.set_num_threads(2)
N="natong19/refusal_classifier"
tok=AutoTokenizer.from_pretrained(N); m=AutoModelForSequenceClassification.from_pretrained(N).eval()
idx=next((i for i,l in {int(k):str(v).lower() for k,v in m.config.id2label.items()}.items() if "refus" in l and "non" not in l),1)
R=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8')]
ids=[r['id'] for r in R if r['list']=='top' and r['k']==1][:4]
keep=[r for r in R if r['id'] in ids and r['split']=='tune' and r['list'] in('baseline','top','random','reverse')]
out=[]
for s in range(0,len(keep),16):
    b=[r['response'] for r in keep[s:s+16]]
    e=tok(b,return_tensors='pt',padding=True,truncation=True,max_length=512)
    with torch.no_grad(): l=m(**e).logits
    out+=(l[:,idx]-l[:,1-idx]).tolist()
d=defaultdict(dict); base={}
for r,x in zip(keep,out):
    if r['list']=='baseline': base[r['id']]=x
    else: d[r['k']].setdefault(r['list'],{})[r['id']]=x
print('ids',ids); print('baseline',[round(base[i],2) for i in ids],'mean',round(st.mean(base[i] for i in ids),2))
for k in sorted(d):
    print(f"k={k:<6}"+"  ".join(f"{l}: mean={st.mean(d[k][l][i] for i in ids if i in d[k][l]):6.2f} min={min(d[k][l][i] for i in ids if i in d[k][l]):6.2f}" for l in('top','random','reverse') if l in d[k]))
