"""由 raw_rate_simulation.py 存下的統計，計算 (K, M) 網格的節點集合差異，不需要 GPU。

用法：python MLOps/analysis/raw_rate_grid.py pipelines/data/raw_rate_stats.pt
"""
import sys, torch
S = torch.load(sys.argv[1], weights_only=True)
R_rate, C_rate = S["r_rate"].flatten(), S["c_rate"].flatten()
rm, cm = S["r_mean"].flatten(), S["c_mean"].flatten()
d = (rm - cm) / ((S["r_std"].flatten()**2 + S["c_std"].flatten()**2) / 2).sqrt().clamp_min(1e-6)
cf, rf = S["cnt"]["complied"] / S["c_n"], S["cnt"]["refused"] / S["r_n"]
Ks = (5, 25, 125, 625, 3125); Ms = (0, 5, 50, 500, 5000)
def grid(name, a, b):
    ro = a.argsort(descending=True, stable=True).tolist()
    co = b.argsort(descending=True, stable=True).tolist()
    sets = {}
    print("==", name)
    for k in Ks:
        row = []
        for m in Ms:
            ex = set(co[:m]); sel = []
            for i in ro:
                if i not in ex:
                    sel.append(i)
                    if len(sel) == k: break
            sets[(k, m)] = frozenset(sel)
            base = sets[(k, 0)]
            row.append(f"{len(sets[(k,m)] & base)/k:.2f}/{d[torch.tensor(sel)].mean():.2f}")
        print(f"k={k:<5}", "  ".join(row))
    print("distinct sets", len(set(sets.values())), "of", len(sets))
grid("active rate", R_rate, C_rate)
grid("mean activation", rm, cm)
grid("top-64 freq", rf, cf)
print("cols: m=", Ms, "; cell = overlap with m=0 / mean d")
