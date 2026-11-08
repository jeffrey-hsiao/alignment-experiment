"""
consume_worker.py — 跨視角共用 buffer 學習

Usage: python consume_worker.py <agent_name> <buffer_path>

從 cross_buffer.pkl 讀取此 agent 尚未消費的條目，
執行 regret update + strategy_sum 累積，儲存至 .2 存檔。
輸出 JSON: {"name": "...", "consumed_ids": [...]}
"""
from __future__ import annotations
import os, sys, json, pickle

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from experiment import AGENT_MAP, SAVE_FILES
from agents.utils import regret_match
from agents.v4_mixin import V4Mixin

CROSS_SAVES = {
    name: os.path.join(BASE, path.replace("_state.pkl", "_2_state.pkl"))
    for name, path in SAVE_FILES.items()
}
ARENA_SAVES = {
    name: os.path.join(BASE, path.replace("_state.pkl", "_1_state.pkl"))
    for name, path in SAVE_FILES.items()
}
ORIG_SAVES = {
    name: os.path.join(BASE, path)
    for name, path in SAVE_FILES.items()
}


def _uses_ext_key(cls) -> bool:
    """V4+ 系列（含 V5/V5_1）使用 13 元素的 ext_key。"""
    return any(c.__name__ == "V4Mixin" for c in cls.__mro__)


def _build_ext_key(base_key: tuple, raw: dict) -> tuple:
    """從 raw_seven 精確重建 V4+ 的 ext_key（13 元素 tuple）。
    T[opp] 來自上帝視角的精確值，非推算。
    """
    T, I, delta, R = raw["T"], raw["I"], raw["delta"], raw["R"]
    return base_key + (
        V4Mixin._bucket_T(T[0]),     V4Mixin._bucket_T(T[1]),
        V4Mixin._bucket_I(I[0]),     V4Mixin._bucket_I(I[1]),
        V4Mixin._bucket_delta(delta[0]), V4Mixin._bucket_delta(delta[1]),
        V4Mixin._bucket_R(R),
    )


def _apply_entry(agent, entry: dict, use_ext: bool) -> bool:
    """對 agent 套用一個 buffer 條目的學習更新。
    MCCFR 系列：regret update + strategy_sum 累積
    MCT   系列：UCB 統計量累積（視為一次觀測）
    """
    if use_ext:
        raw      = entry.get("raw_seven")
        base_key = entry.get("base_key")
        if raw is not None and base_key is not None:
            info_key = _build_ext_key(tuple(base_key), raw)
        else:
            # 舊格式 fallback（含預算的 ext_key）
            info_key = entry.get("ext_key")
            if info_key is None:
                return False
    else:
        info_key = entry.get("base_key")
        if info_key is None:
            return False

    actions = entry.get("actions")
    action_values = entry.get("action_values")
    if not actions or not action_values:
        return False

    # MCCFR 系列
    if hasattr(agent, "_regrets") and hasattr(agent, "_strategy_sum"):
        agent._update_regrets(info_key, actions, action_values)
        strategy = regret_match(
            {a: agent._regrets[(info_key, a)] for a in actions}, actions
        )
        for a in actions:
            agent._strategy_sum[(info_key, a)] += strategy[a]
        return True

    # MCT 系列：將 action_values 作為一次 UCB 觀測加入
    if hasattr(agent, "_ucb_values") and hasattr(agent, "_ucb_visits"):
        for a in actions:
            v = action_values.get(a, 0.0)
            agent._ucb_values[(info_key, a)] += v
            agent._ucb_visits[(info_key, a)] += 1
        return True

    return False


if __name__ == "__main__":
    name     = sys.argv[1]
    buf_path = sys.argv[2]

    cls = AGENT_MAP.get(name)
    if cls is None:
        print(json.dumps({"name": name, "consumed_ids": []}))
        sys.exit(0)

    # 讀 buffer
    if not os.path.exists(buf_path):
        print(json.dumps({"name": name, "consumed_ids": []}))
        sys.exit(0)
    with open(buf_path, "rb") as f:
        buf: list[dict] = pickle.load(f)

    # 載入 agent（.2 > .1 > original）
    agent = cls(0)
    for path in (CROSS_SAVES.get(name), ARENA_SAVES.get(name), ORIG_SAVES.get(name)):
        if path and os.path.exists(path) and hasattr(agent, "load"):
            agent.load(path)
            break

    use_ext = _uses_ext_key(cls)
    consumed_ids: list[str] = []

    for entry in buf:
        eid = entry["id"]
        if name in entry.get("consumed_by", set()):
            continue
        if _apply_entry(agent, entry, use_ext):
            consumed_ids.append(eid)

    # 存至 .2
    if consumed_ids:
        cross_path = CROSS_SAVES.get(name)
        if cross_path and hasattr(agent, "save"):
            agent.save(cross_path)

    print(json.dumps({"name": name, "consumed_ids": consumed_ids}))
