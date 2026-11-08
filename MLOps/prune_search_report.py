"""prune_search.py 的圖表。維持單一軸，顏色依固定順序分配。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def write_report(state: dict, path: Path) -> None:
    cells = state["cells"]
    sync = [c for c in cells if c["kind"] == "sync"]
    bounds = state.get("bounds") or {}
    k_equals_m = sorted((c for c in sync if c["k"] == c["m"]), key=lambda c: c["k"])

    fig = plt.figure(figsize=(13, 5 + 0.42 * max(len(cells), 6)), facecolor=SURFACE,
                     constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, max(1.0, len(cells) / 10)])

    ax = fig.add_subplot(grid[0, 0], facecolor=SURFACE)
    xs = [c["k"] for c in k_equals_m]
    ax.plot(xs, [c["flips"] for c in k_equals_m], color=BLUE, linewidth=2, marker="o", label="翻轉")
    ax.plot(xs, [c["auto_collapse"] for c in k_equals_m], color=ORANGE, linewidth=2, marker="o",
            label="自動崩壞")
    ax.set_xscale("log")
    ax.set_xlabel("K=M", color=MUTED)
    ax.set_ylabel("題數", color=MUTED)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    source = "粗測決定" if bounds.get("source") == "coarse" else "設定檔指定"
    ax.set_title(f"K=M 的翻轉與崩壞（範圍由{source}）", loc="left", color=INK, fontsize=13,
                 fontweight="bold")

    ax = fig.add_subplot(grid[0, 1], facecolor=SURFACE)
    ax.plot([c["m"] for c in sync], [c["k"] for c in sync], color=BLUE, linewidth=2, zorder=1)
    ax.scatter([c["m"] for c in sync], [c["k"] for c in sync],
               s=[24 + 8 * c["other_flips"] for c in sync], color=BLUE, edgecolors=SURFACE,
               linewidths=2, zorder=2)
    if len({c["m"] for c in sync}) > 1:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("M（排除數）", color=MUTED)
    ax.set_ylabel("K（選取數）", color=MUTED)
    ax.set_title("執行路徑（點越大，其餘翻轉越多）", loc="left", color=INK, fontsize=13, fontweight="bold")

    ax = fig.add_subplot(grid[1, :], facecolor=SURFACE)
    positions = list(range(len(cells)))[::-1]
    left = [0] * len(cells)
    for key, label, color in (("other_flips", "其餘翻轉（仍可能有事實錯亂）", BLUE),
                              ("auto_collapse", "自動崩壞", ORANGE), ("refused", "拒絕", AQUA)):
        values = [c[key] for c in cells]
        ax.barh(positions, values, left=left, height=0.62, color=color, label=label,
                edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, values)]
    names = [f"#{c['step']} {c['kind']} K={c['k']:,} M={c['m']:,}"
             + (f" r{c['repeat']}" if c["kind"].startswith("random") else "") for c in cells]
    ax.set_yticks(positions, names, color=INK, fontsize=9)
    ax.set_xlabel("題數", color=MUTED)
    ax.set_title("每格的回答組成（依執行順序）", loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.legend(ncols=3, frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.05))

    for axis in fig.axes:
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_visible(False)

    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
