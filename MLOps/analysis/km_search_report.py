"""把 prune_km_search.py 的結果畫成圖表並輸出比較數字。不需要 GPU。

用法：python MLOps/analysis/km_search_report.py pipelines/data/km_search

圖一：每格的回答組成，依實際執行順序
圖二：K、M 的搜尋路徑
圖三：有效題數與事實錯亂比例的關係
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [("ok", "有效", "#2a78d6"), ("factual", "事實錯亂", "#eb6834"),
          ("off_topic", "答非所問", "#1baf7a"), ("collapse", "崩壞", "#eda100"),
          ("refusal", "拒絕", "#e87ba4")]
PROBLEM_LIMIT = 8


def factual_share(cell: dict) -> float | None:
    answered = cell["ok"] + cell["factual"]
    return cell["factual"] / answered if answered else None


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for position in range(i, j + 1):
            ranks[order[position]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    return cov / (sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)) ** 0.5


def main() -> None:
    out_dir = Path(sys.argv[1])
    cells = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["cells"]
    names = [f"K={c['k']:,}  M={c['m']:,}" for c in cells]

    fig = plt.figure(figsize=(13, 13), facecolor=SURFACE, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

    ax = fig.add_subplot(grid[0, :], facecolor=SURFACE)
    positions = list(range(len(cells)))[::-1]
    left = [0] * len(cells)
    for key, label, color in SERIES:
        values = [c[key] for c in cells]
        ax.barh(positions, values, left=left, height=0.62, color=color, label=label,
                edgecolor=SURFACE, linewidth=2)
        if key == "ok":
            for y, value in zip(positions, values):
                ax.text(value - 0.25, y, str(value), va="center", ha="right", color="white",
                        fontsize=10, fontweight="bold")
        left = [a + b for a, b in zip(left, values)]
    for y, cell in zip(positions, cells):
        ax.text(27.3, y, f"問題 {cell['problems']}", va="center", color=INK if
                cell["problems"] >= PROBLEM_LIMIT else MUTED, fontsize=10,
                fontweight="bold" if cell["problems"] >= PROBLEM_LIMIT else "normal")
    ax.set_yticks(positions, names, color=INK, fontsize=10)
    ax.set_xlim(0, 30)
    ax.set_xlabel("題數（共 27 題）", color=MUTED)
    ax.set_title("每格的回答組成（依執行順序，問題達 8 題就降 K）", loc="left", color=INK,
                 fontsize=14, fontweight="bold")
    ax.legend(ncols=5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.07), labelcolor=INK)

    ax = fig.add_subplot(grid[1, 0], facecolor=SURFACE)
    ks, ms = [c["k"] for c in cells], [c["m"] for c in cells]
    ax.plot(ms, ks, color="#2a78d6", linewidth=2, zorder=1)
    ax.scatter(ms, ks, s=[20 + 10 * c["ok"] for c in cells], color="#2a78d6",
               edgecolors=SURFACE, linewidths=2, zorder=2)
    best = max(cells, key=lambda c: c["ok"] - c["problems"])
    for cell in (cells[0], best, cells[-1]):
        ax.annotate(f"K={cell['k']:,}\nM={cell['m']:,}", (cell["m"], cell["k"]),
                    textcoords="offset points", xytext=(8, -26), fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M（排除數）", color=MUTED)
    ax.set_ylabel("K（選取數）", color=MUTED)
    ax.set_title("K、M 搜尋路徑（點越大，有效題數越多）", loc="left", color=INK, fontsize=13,
                 fontweight="bold")

    ax = fig.add_subplot(grid[1, 1], facecolor=SURFACE)
    usable = [c for c in cells if factual_share(c) is not None]
    x = [c["ok"] for c in usable]
    y = [100 * factual_share(c) for c in usable]
    ax.scatter(x, y, s=80, color="#eb6834", edgecolors=SURFACE, linewidths=2)
    for cell, px, py in zip(usable, x, y):
        if (cell["k"], cell["m"]) in {(20000, 20000), (39062, 61035), (25000, 39062)}:
            ax.annotate(f"K={cell['k']:,} M={cell['m']:,}", (px, py), textcoords="offset points",
                        xytext=(-6, 10), ha="right", fontsize=9, color=INK)
    correlation = spearman(x, y)
    ax.set_xlabel("有效題數", color=MUTED)
    ax.set_ylabel("事實錯亂比例（有效＋事實錯亂中）%", color=MUTED)
    ax.set_ylim(0, 75)
    ax.set_title(f"有效越多，事實錯亂比例越低嗎？（Spearman {correlation:.2f}，n={len(usable)}）",
                 loc="left", color=INK, fontsize=13, fontweight="bold")

    for axis in fig.axes:
        axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_visible(False)

    fig.savefig(out_dir / "report.png", dpi=130, facecolor=SURFACE)
    print(f"Spearman(有效題數, 事實錯亂比例) = {correlation:.3f}，n={len(usable)}")
    print(f"已寫出 {out_dir / 'report.png'}")


if __name__ == "__main__":
    main()
