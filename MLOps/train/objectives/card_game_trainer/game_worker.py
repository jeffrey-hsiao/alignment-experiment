"""Single-game worker for arena matchmaking.

Usage:
  python game_worker.py <name_a> <name_b> <mode> [frozen]
  mode: eval  — load original saves, no save after game
        train — load arena .1 saves (fallback to original), save to .1 after game
        cross — load .2 saves (fallback .1, fallback original), save to .2 after game
                + capture trajectory for cross-agent buffer learning
  frozen（選填，逗號分隔，如 "a"、"b"、"a,b"）：指定哪一側本場不訓練——即使
  mode="train"，該側也不呼叫 .save()，且一律讀「原始」存檔（不吃 .1 版本），
  用途是在 train 模式的 roster 裡放一個固定不變的參照對手，讓其他成員練，
  同時不汙染它自己的存檔/ELO（ELO 的部分由呼叫端 arena.py 依這個欄位判斷跳過）。

  name_a/name_b 除了 AGENT_MAP 的真實類別名/別名，也接受泛用 '<base>_1' 變體
  （由 arena.py 對 V1~V4 合成，例如 'MCTAgentV3_1'）：一律優先讀 arena .1 存檔，
  不存在則 fallback 讀原始存檔；ELO/矩陣計分用的識別字串與基底版本不同，互不污染。
  見 _resolve()。

Prints one JSON line to stdout:
  {"winner": "<name_a|name_b|null>", "name_a": "...", "name_b": "...", "frozen": ["a", ...]}
  cross mode also includes "traj_blob": base64-encoded pickle of trajectory data
"""
from __future__ import annotations
import sys, io, os, json, inspect, base64, pickle as _pkl

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from experiment import run_game, AGENT_MAP, SAVE_FILES

ARENA_SAVES = {
    name: path.replace("_state.pkl", "_1_state.pkl")
    for name, path in SAVE_FILES.items()
}
CROSS_SAVES = {
    name: path.replace("_state.pkl", "_2_state.pkl")
    for name, path in SAVE_FILES.items()
}


def _supports_player_mode(cls) -> bool:
    return "player_mode" in inspect.signature(cls.__init__).parameters


_INIT_HP = 55  # DEFAULT_CHARACTER.max_hp


def _compute_seven_godview(state, player_id: int) -> dict:
    """從 state.history（上帝視角）精確計算 7 個歷史數字。
    T[opp] 使用 p{opp}_played 的精確值，不再推算。
    """
    T = [0.0, 0.0]
    I = [0, 0]
    delta = [0.0, 0.0]
    prev_hp = [_INIT_HP, _INIT_HP]

    for r in getattr(state, "history", []):
        T[0] += sum(r.p0_played) if r.p0_played else 0.0
        T[1] += sum(r.p1_played) if r.p1_played else 0.0
        I[0] += max(0, r.p0_hand_size_after - 2 - (1 if r.winner == 0 else 0))
        I[1] += max(0, r.p1_hand_size_after - 2 - (1 if r.winner == 1 else 0))
        delta[0] += max(0.0, prev_hp[0] - r.p0_hp_after)
        delta[1] += max(0.0, prev_hp[1] - r.p1_hp_after)
        prev_hp = [r.p0_hp_after, r.p1_hp_after]

    my  = player_id
    opp = 1 - player_id
    return {
        "T":     [T[my], T[opp]],
        "I":     [I[my], I[opp]],
        "delta": [delta[my], delta[opp]],
        "R":     getattr(state, "round_num", 0),
    }


class _TrajCaptureMixin:
    """攔截 _run_phase1，以上帝視角精確記錄 7 數字與 action_values。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._traj: list[dict] = []

    def _run_phase1(self, state):
        result = super()._run_phase1(state)
        action_values, counts, obs, info_key = result
        raw_seven = _compute_seven_godview(state, self.player_id)
        self._traj.append({
            "base_key":      tuple(info_key[:6]),
            "raw_seven":     raw_seven,
            "actions":       tuple(action_values.keys()),
            "action_values": dict(action_values),
        })
        return result


def _resolve(name):
    """解析 arena 傳入的 agent 識別字串。

    真實獨立類別（如 'MCTAgentV5_1'）直接命中 AGENT_MAP。泛用 _1 變體
    （如 'MCTAgentV3_1'，由 arena.py 對 V1~V4 合成，本身不是 AGENT_MAP 的鍵）
    去掉字尾後再查一次。回傳 (cls, is_variant)；is_variant=True 表示一律優先讀
    arena .1 存檔（不存在則 fallback 讀原始存檔），不受 mode 是否為 train/cross 影響。
    """
    if name in AGENT_MAP:
        return AGENT_MAP[name], False
    if name.endswith("_1") and name[:-2] in AGENT_MAP:
        return AGENT_MAP[name[:-2]], True
    raise KeyError(name)


def _load_agent(name, pid, mode, frozen=False):
    cls, is_variant = _resolve(name)
    cross = (mode == "cross")

    if frozen:
        # 凍結側：跟 eval 模式一樣盡量用純推論模式，且不吃 .1 訓練變體——
        # 固定參照對手要讀「原始」存檔，才不會因為之前 arena 訓練過而漂移。
        player_mode = _supports_player_mode(cls)
        ag = cls(pid, player_mode=player_mode) if player_mode else cls(pid)
        if hasattr(ag, "load"):
            orig_path = SAVE_FILES.get(cls.__name__)
            if orig_path and os.path.exists(orig_path):
                ag.load(orig_path)
        return ag

    if cross:
        capturing_cls = type(f"_C{cls.__name__}", (_TrajCaptureMixin, cls), {})
        ag = capturing_cls(pid)
    else:
        player_mode = (mode == "eval") and _supports_player_mode(cls)
        ag = cls(pid, player_mode=player_mode) if player_mode else cls(pid)

    if not hasattr(ag, "load"):
        return ag

    cross_path = CROSS_SAVES.get(cls.__name__)
    arena_path = ARENA_SAVES.get(cls.__name__)
    orig_path  = SAVE_FILES.get(cls.__name__)

    if cross and cross_path and os.path.exists(cross_path):
        ag.load(cross_path)
    elif (is_variant or mode in ("train", "cross")) and arena_path and os.path.exists(arena_path):
        ag.load(arena_path)
    elif orig_path and os.path.exists(orig_path):
        ag.load(orig_path)
    return ag


if __name__ == "__main__":
    name_a, name_b, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    frozen_arg = sys.argv[4] if len(sys.argv) > 4 else ""
    frozen_sides = set(s for s in frozen_arg.split(",") if s)  # {"a"} / {"b"} / {"a","b"} / {}

    a = _load_agent(name_a, 0, mode, frozen=("a" in frozen_sides))
    b = _load_agent(name_b, 1, mode, frozen=("b" in frozen_sides))

    result = run_game(a, b)

    # 存檔（一律以底層類別名定位 .1/.2 存檔路徑；_1 變體與基底版本共用同一份存檔）
    # 凍結側即使 mode="train" 也不存檔——這場它不是訓練對象，是固定參照對手。
    if mode == "train":
        for ag, name, side in [(a, name_a, "a"), (b, name_b, "b")]:
            if side in frozen_sides:
                continue
            cls, _ = _resolve(name)
            p = ARENA_SAVES.get(cls.__name__)
            if p and hasattr(ag, "save"):
                ag.save(p)
    elif mode == "cross":
        for ag, name, side in [(a, name_a, "a"), (b, name_b, "b")]:
            if side in frozen_sides:
                continue
            cls, _ = _resolve(name)
            p = CROSS_SAVES.get(cls.__name__)
            if p and hasattr(ag, "save"):
                ag.save(p)

    winner = name_a if result == 0 else (name_b if result == 1 else None)
    out = {"winner": winner, "name_a": name_a, "name_b": name_b, "frozen": sorted(frozen_sides)}

    # 輸出軌跡 blob（cross 模式）
    if mode == "cross":
        traj_data = {
            "entries_a": getattr(a, "_traj", []),
            "entries_b": getattr(b, "_traj", []),
        }
        out["traj_blob"] = base64.b64encode(_pkl.dumps(traj_data)).decode()

    print(json.dumps(out), flush=True)
