"""原始亮起率模擬：統計同分節點數量，並模擬 K=M 的選取結果，不做裁減。

用法：python MLOps/analysis/raw_rate_simulation.py MLOps pipelines/data/refusal_dataset.jsonl pipelines/data/raw_rate_stats.pt
注意：這份統計用全部 204 題，只供預覽，正式掃描的名單只看 rank 集。
"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from neuron_probe import GenerationProbe, GroupStats, load_seeds, load_target_model, DEFAULT_MODEL_NAME, DEFAULT_SYSTEM

records = load_seeds([sys.argv[2]], ("refused", "complied"))
model, tok = load_target_model(None, "final", DEFAULT_MODEL_NAME)
model.config.use_cache = True
N = 64
g = {"refused": GroupStats(), "complied": GroupStats()}
cnt = {}
with GenerationProbe(model, tok, system=DEFAULT_SYSTEM, act_threshold=0.0) as p:
    for i, r in enumerate(records, 1):
        acts, _ = p.run(r["text"], 8)
        for t in range(acts.shape[0]):
            g[r["group"]].update(acts[t], 0.0)
            c = cnt.setdefault(r["group"], torch.zeros(acts.shape[1] * acts.shape[2]))
            c[acts[t].flatten().topk(N).indices] += 1
        if i % 50 == 0: print("done", i, flush=True)

R, C = g["refused"], g["complied"]
torch.save({"r_rate": R.active_rate, "c_rate": C.active_rate, "r_mean": R.mean, "c_mean": C.mean,
            "r_std": R.std, "c_std": C.std, "r_n": R.n, "c_n": C.n, "cnt": cnt}, sys.argv[3])
fl = lambda x: x.flatten()
d = fl((R.mean - C.mean) / ((R.std**2 + C.std**2) / 2).sqrt().clamp_min(1e-6))
rr, cr = fl(R.active_rate), fl(C.active_rate)
total = rr.numel()
print("total", total, "tokens", R.n, C.n)
for th in (1.0, 0.999, 0.99, 0.95, 0.9):
    print(f"rate>={th}: refused {(rr>=th).sum().item()} complied {(cr>=th).sum().item()} both {((rr>=th)&(cr>=th)).sum().item()}")
print("refused-only rate==1:", ((rr>=1)&(cr<1)).sum().item())
d64 = set(d.topk(64).indices.tolist())

def simulate(name, rm, cm):
    print("==", name)
    rorder = rm.argsort(descending=True, stable=True)
    corder = cm.argsort(descending=True, stable=True)
    for k in (5, 10, 25, 50, 100, 500, 1000, 5000):
        ctop = set(corder[:k].tolist())
        sel, removed = [], 0
        for idx in rorder.tolist():
            if idx in ctop: removed += 1; continue
            sel.append(idx)
            if len(sel) == k: break
        s = torch.tensor(sel)
        print(f"k={k:<5} removed={removed:<5} d_mean={d[s].mean():.3f} d_min={d[s].min():.3f} rate_r={rr[s].mean():.3f} rate_c={cr[s].mean():.3f} in_d64={len(d64&set(sel))} first={[(i//8960,i%8960) for i in sel[:3]]}")

simulate("active rate (ties by index)", rr, cr)
simulate("active rate (ties by mean)", rr + 1e-4 * torch.tanh(fl(R.mean)), cr + 1e-4 * torch.tanh(fl(C.mean)))
simulate("mean activation", fl(R.mean), fl(C.mean))
simulate(f"freq in per-token top-{N}", cnt["refused"] / R.n, cnt["complied"] / C.n)
for nm, c, n in (("refused", cnt["refused"], R.n), ("complied", cnt["complied"], C.n)):
    print(nm, f"top{N}-freq: nonzero {(c>0).sum().item()}, top rates", [round(v, 3) for v in (c / n).topk(8).values.tolist()])
print("freq overlap top64 between groups:", len(set(cnt["refused"].topk(64).indices.tolist()) & set(cnt["complied"].topk(64).indices.tolist())))
