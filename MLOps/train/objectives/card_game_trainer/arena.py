"""
多智能體競技場 — MMR 匹配制；每場對局為獨立子進程

eval 模式（預設）：每場皆重新載入原始存檔，不存檔，N = 總局數
  python arena.py                              # 300 局（預設）
  python arena.py 500                          # 500 局
  python arena.py --agents MCTV4 MCCFRV4 MCTV5 # 指定參賽者（可用縮寫）
  python arena.py --cpu 80                      # CPU 上限 80%（預設 60）
  python arena.py --no-save                    # 不更新 ELO / 矩陣 JSON
  python arena.py --reset                      # 重設 ELO + 矩陣
  python arena.py --leaderboard                # 只顯示排名，不跑比賽
  python arena.py -v                           # 詳細進度（每局）

quick 模式（V4~V5 快捷循環賽，N = 每對局數）：
  python arena.py --quick                      # 每對 20 局
  python arena.py --quick 30                   # 每對 30 局

train 模式：同 eval 但存檔至 .1 路徑（不覆蓋原始實驗存檔），N = 總局數
  python arena.py --train                      # 500 局
  python arena.py --train 1000                 # 1000 局
  python arena.py --train --agents MCTV4 MCCFRV4 MCTV5 MCCFRV5

filling 模式：同 eval（不存檔），補滿對局數 < N 的配對至 N 局，N 預設 30
  python arena.py --filling                    # 補滿所有配對至 30 局
  python arena.py --filling 50                 # 補滿至 50 局
  python arena.py --filling --agents MCTV4 MCCFRV4 MCTV5

balance 模式：平衡各模型的訓練對局數，補至 N 局（存檔至 .1 路徑），N 預設 330
  python arena.py --balance                    # 補平至 330 局
  python arena.py --balance 400                # 補平至 400 局
  python arena.py --balance 350 --agents MCTV4 MCCFRV4 MCTV5 MCTV5_1
  優先配對對局數不足的模型，實現訓練數據的均衡分配

_1 / .1 變體（任何模式下 --agents 皆可用，V1~V4 專用；V5 已有獨立的 V5_1 類別）：
  在版本名後加 _1 或 .1（例：MCTV3_1、MCCFRV2.1）即可把「arena 訓練過的 .1 存檔」
  當成獨立參賽者加入，沒有 .1 存檔時自動 fallback 讀原始存檔；其 ELO / 勝負矩陣
  與不帶 _1 的基底版本分開記錄，方便比較訓練前後差異。
  python arena.py --agents MCTV3 MCTV3_1       # 原始 V3 vs 練過的 V3.1，互相對打
"""
from __future__ import annotations
import base64, json, math, os, pickle, random, re as _re, shutil, sys, itertools, subprocess, time, unicodedata, uuid
from experiment import run_game, AGENT_MAP, SAVE_FILES

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

CPU_TARGET = 60.0   # 不超過此 CPU% 才允許生成新子進程
from agents.mct_agent_v5    import MCTAgentV5
from agents.mccfr_agent_v5  import MCCFRAgentV5
from agents.mct_agent_v5_1  import MCTAgentV5_1
from agents.mccfr_agent_v5_1 import MCCFRAgentV5_1

# 競技場存檔路徑：不覆蓋原始實驗存檔
# e.g. mct_v3_state.pkl → mct_v3_1_state.pkl
# 特例：mct_v5_state.pkl → mct_v5_1_state.pkl（即 MCTAgentV5_1 的存檔路徑，此為預期行為）
ARENA_SAVE_FILES = {
    name: path.replace("_state.pkl", "_1_state.pkl")
    for name, path in SAVE_FILES.items()
}
# Cross 模式（--cross）存檔路徑：.2 後綴
CROSS_SAVE_FILES = {
    name: path.replace("_state.pkl", "_2_state.pkl")
    for name, path in SAVE_FILES.items()
}

_BASE             = os.path.dirname(os.path.abspath(__file__))
CROSS_BUFFER_FILE = os.path.join(_BASE, "cross_buffer.pkl")
_CONSUME_WORKER   = os.path.join(_BASE, "consume_worker.py")

_WORKER = os.path.join(_BASE, "game_worker.py")

ELO_FILE    = "arena_elo.json"
MATRIX_FILE = "arena_matrix.json"
DEFAULT_ELO = 1500.0
K_ELO       = 32.0

DEFAULT_ROSTER = [
    "MCTV2_1",    "MCCFRV2_1",
    "MCTV3_1",    "MCCFRV3_1",
    "MCTV4_1",    "MCCFRV4_1",
    "MCTV5_1",    "MCCFRV5_1",
    # V6 已棄用：架構設計失敗，BFS/DFS 二層加權平均機制被完全破壞
    # 詳見：git log（V6 重寫 _mcts_search 時的分析）
]

QUICK_ROSTER = [
    "MCTAgentV4",    "MCCFRAgentV4",
    "MCTAgentV5",    "MCCFRAgentV5",
    "MCTAgentV5_1",  "MCCFRAgentV5_1",
]


# ─── 名稱解析（含泛用 _1/.1 變體）───────────────────────────────────────────────

def _resolve_agent(name):
    """解析使用者輸入的單個 agent 名稱。

    真實獨立類別（如 'MCTV5_1' → MCTAgentV5_1）直接命中 AGENT_MAP，
    identity 即 cls.__name__。

    泛用 _1/.1 變體（如 'MCTV3_1'／'MCTV3.1' → MCTAgentV3 的 .1 存檔版本）
    不會直接命中 AGENT_MAP，去掉字尾後再查一次；identity 為基底類別名 + '_1'，
    與基底版本（不帶 _1）分開記錄 ELO / 勝負矩陣。實際載入的權重由 game_worker.py
    決定：優先讀 arena .1 存檔，不存在則 fallback 讀原始存檔。

    回傳 (identity, cls)；查無則回傳 (None, None)。
    """
    if name in AGENT_MAP:
        cls = AGENT_MAP[name]
        return cls.__name__, cls
    if name.endswith(("_1", ".1")):
        base = AGENT_MAP.get(name[:-2])
        if base:
            return base.__name__ + "_1", base
    return None, None


def _resolve_roster(tokens):
    """解析一組 agent 名稱，回傳 (去重後的 identity 列表, 無法解析的原始輸入列表)。"""
    identities, bad = [], []
    for r in tokens:
        identity, _ = _resolve_agent(r)
        (bad if identity is None else identities).append(r if identity is None else identity)
    return list(dict.fromkeys(identities)), bad


# ─── I/O ─────────────────────────────────────────────────────────────────────

def _load(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── ELO ─────────────────────────────────────────────────────────────────────

def _elo_expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

def _elo_update(elo, na, nb, score_a, skip_a=False, skip_b=False):
    """skip_a/skip_b：該側本場是凍結參照對手（未被訓練），不更新它的 ELO——
    但仍計算 ea/score_a，另一側的評分照樣依照對戰結果正常調整。"""
    ra = elo.setdefault(na, DEFAULT_ELO)
    rb = elo.setdefault(nb, DEFAULT_ELO)
    ea = _elo_expected(ra, rb)
    if not skip_a:
        elo[na] = ra + K_ELO * (score_a - ea)
    if not skip_b:
        elo[nb] = rb + K_ELO * ((1.0 - score_a) - (1.0 - ea))


# ─── 矩陣 ────────────────────────────────────────────────────────────────────

def _mkey(na, nb):
    """回傳 (canonical_key, na_is_lexfirst)。wins_a 永遠對應字典序較小的 agent。"""
    if na < nb:
        return f"{na}||{nb}", True
    return f"{nb}||{na}", False

def _record(matrix, winner, na, nb):
    key, na_is_a = _mkey(na, nb)
    rec = matrix.setdefault(key, {"wins_a": 0, "wins_b": 0, "draws": 0})
    if winner is None:
        rec["draws"] += 1
    elif (winner == na) == na_is_a:
        rec["wins_a"] += 1
    else:
        rec["wins_b"] += 1

def _h2h(matrix, na, nb):
    """回傳 (wins_na, wins_nb, draws)，從 na 視角。"""
    key, na_is_a = _mkey(na, nb)
    rec = matrix.get(key, {"wins_a": 0, "wins_b": 0, "draws": 0})
    wa, wb, d = rec["wins_a"], rec["wins_b"], rec["draws"]
    return (wa, wb, d) if na_is_a else (wb, wa, d)


# ─── Eval 模式：循環賽 ────────────────────────────────────────────────────────

def _mk_agent(cls, pid):
    ag = cls(pid)
    p = SAVE_FILES.get(cls.__name__)
    if p and os.path.exists(p) and hasattr(ag, "load"):
        ag.load(p)
    return ag

def _mk_frozen_agent(cls, pid):
    """載入存檔但凍結 DQN（player_mode=True），作為固定強力對手使用。"""
    ag = cls(pid, player_mode=True)
    p = SAVE_FILES.get(cls.__name__)
    if p and os.path.exists(p) and hasattr(ag, "load"):
        ag.load(p)
    del ag._tgt_net
    return ag

def run_eval_pair(cls_a, cls_b, n, elo, matrix, verbose):
    """A vs B 跑 n 局，前半 A 為 player 0，後半 A 為 player 1，確保兩種索引皆有覆蓋。不更新 Agent 權重。"""
    na, nb = cls_a.__name__, cls_b.__name__
    half = n // 2
    done = 0

    def _play(p0ag, p1ag, p0n, p1n):
        nonlocal done
        r = run_game(p0ag, p1ag)
        w = p0n if r == 0 else (p1n if r == 1 else None)
        _elo_update(elo, na, nb, 1.0 if w == na else 0.5 if w is None else 0.0)
        _record(matrix, w, na, nb)
        done += 1
        if verbose and done % 5 == 0:
            wa, wb, d = _h2h(matrix, na, nb)
            t = wa + wb + d
            print(f"    [{done}/{n}] {na}: {wa}W  {nb}: {wb}W  D:{d}", flush=True)

    a0, b1 = _mk_agent(cls_a, 0), _mk_agent(cls_b, 1)
    for _ in range(half):
        _play(a0, b1, na, nb)

    b0, a1 = _mk_agent(cls_b, 0), _mk_agent(cls_a, 1)
    for _ in range(n - half):
        _play(b0, a1, nb, na)


def _split_name(name: str):
    """'MCTAgentV3' → ('MCT','V3')  |  'MCCFRAgentV5_1' → ('CFR','V5.1')"""
    series = "CFR" if name.startswith("MCCFR") else "MCT"
    m = _re.search(r'V(\d+)(?:_(\d+))?', name)
    if m:
        ver = f"V{m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")
    else:
        ver = "V1"
    return series, ver


def _visual_width(s: str) -> int:
    """終端機視覺寬度：全角／CJK／Ambiguous（含 ═─ 等框線字元，中文終端機慣例渲染為全角）算 2，其餘算 1。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F", "A") else 1 for ch in s)


def _term_rows(line: str, width: int) -> int:
    """一行文字在終端機裡實際佔用幾個物理行（視覺寬度超過終端機寬度會自動換行）。"""
    return max(1, -(-_visual_width(line) // width))


# ─── MMR 匹配競技場（subprocess + ANSI 原地刷新）──────────────────────────────

class MatchmakingArena:
    """MMR 匹配制競技場：每場對局為獨立子進程，ANSI 原地更新排名表與矩陣。

    eval    — 每場從原始存檔載入，不存檔
    train   — 從 .1 存檔載入（fallback 原始），存回 .1
    filling — 同 eval（不存檔），但從固定的補局隊列（fill_pairs）取出配對，
              用於把對局數不足 N 局的配對補滿，而非 MMR 自動配對
    balance — 同 train（存回 .1），動態選擇對局數不足的模型及其對手，
              用 ELO 加權選擇對手，實現訓練對局數 + ELO 平衡
    """
    _CW = 8   # 矩陣欄寬

    def __init__(self, names, elo, matrix, total, mode="eval",
                 cpu_target=CPU_TARGET, reset_buf=False, fill_pairs=None, balance_target=None,
                 frozen_names=frozenset(), quiet=False, on_harvest=None):
        self._names        = list(names)
        self._elo          = elo
        self._matrix       = matrix
        self._total        = total
        self._mode         = mode
        self._cpu_target   = cpu_target
        self._done         = 0
        self._queue: list[str]                       = []
        self._fill_queue: list[tuple[str, str]]       = list(fill_pairs) if fill_pairs else []
        self._running: dict[subprocess.Popen, tuple] = {}
        # 固定參照對手：這一輪不訓練、不存檔、ELO 不更新（矩陣勝負紀錄仍照記）。
        # 用途：roster 裡放一個已知強度的基準對手，讓其他成員練，同時不汙染
        # 它自己的存檔或評分。見 _elo_update()/_launch()/_harvest()。
        self._frozen_names = set(frozen_names)
        # quiet：關掉自己的 ANSI 即時刷新排行榜（由外部呼叫端接管進度顯示，
        # 例如 MLOps 的 run.py callback/顯示模式，避免兩套進度輸出互相干擾）。
        # on_harvest：選填 callback(na, nb, winner, frozen_sides, done, total)，
        # 每局結束、_harvest() 處理完一場對局後呼叫一次，讓外部能同步進度。
        self._quiet      = quiet
        self._on_harvest = on_harvest

        # 訓練模式安全檢查：禁止非 .1 版本（凍結參照對手例外——它本來就該讀
        # 原始存檔、不受這條「會覆蓋 .1 訓練資料」的風險影響）
        if mode in ("train", "cross", "balance"):
            non_variant = [n for n in self._names
                           if not n.endswith("_1") and n not in self._frozen_names]
            if non_variant:
                print(f"\n[ERROR] 訓練模式禁止使用非 .1 版本的智能體！")
                print(f"[ERROR] 找到: {', '.join(non_variant)}")
                print(f"[ERROR] 這些模型會覆蓋掉對應 .1 版本的訓練數據")
                print(f"[ERROR] 請改用: {', '.join(n + '_1' if not n.endswith('_1') else n for n in non_variant)}")
                sys.exit(1)

        # Balance 模式：動態補局
        self._balance_target = balance_target
        self._balance_needs: dict[str, int] = {}

        # 轮次制（train 模式）：每轮每模型最多配对一次
        self._round = 0
        self._round_matched: set[str] = set()  # 当前轮已配对的模型
        self._last_opp: dict[str, str] = {}   # 上一個對手，避免 train/cross 立刻重打同一人
        self._display_lines = 0
        self._start_time    = 0.0
        self._last_render   = 0.0
        self._cpu_pct       = 0.0
        # 爬坡自適應並行上限（僅 eval 模式）
        n_cores = os.cpu_count() or 8
        self._dynamic_max  = max(1, int(n_cores * 0.4))
        self._last_ramp    = 0.0
        self._ramp_up_thr  = cpu_target * 0.82
        self._ramp_dn_thr  = cpu_target
        # Cross 模式：共用 buffer + consume 工作進程
        self._cross = (mode == "cross")
        if self._cross:
            # 所有 agent（MCCFR 用 regret update，MCT 用 UCB 累積）
            self._learner_names: set[str] = set(names)
            self._cross_buffer: list[dict]              = []
            self._consume_procs: dict[subprocess.Popen, str] = {}
            self._consuming: set[str]                   = set()
            if not reset_buf and os.path.exists(CROSS_BUFFER_FILE):
                try:
                    with open(CROSS_BUFFER_FILE, "rb") as _f:
                        loaded = pickle.load(_f)
                    # 只保留未被所有 learner 消費的條目
                    self._cross_buffer = [
                        e for e in loaded
                        if not e.get("consumed_by", set()) >= self._learner_names
                    ]
                except Exception:
                    self._cross_buffer = []

    # ── public ────────────────────────────────────────────────────────────────

    def run(self):
        if sys.platform == "win32":
            os.system("")           # 啟用 Windows ANSI 支援
        if _psutil:
            _psutil.cpu_percent(interval=0.5)   # 預熱，第一次呼叫永遠回 0

        self._queue      = list(self._names)
        self._start_time = time.perf_counter()
        if not self._quiet:
            self._render()

        while self._done < self._total or self._running or (self._cross and self._consume_procs):
            before = self._done
            self._harvest()
            if self._cross:
                self._harvest_consume()
            if _psutil:
                self._cpu_pct = _psutil.cpu_percent(interval=None)
                if self._mode in ("eval", "filling", "balance"):
                    self._adjust_max()
            if self._done < self._total:
                self._spawn()
            if self._cross:
                self._spawn_consume()
            now = time.perf_counter()
            if not self._quiet and (self._done != before or now - self._last_render >= 5.0):
                self._render()
                self._last_render = now
            time.sleep(0.05)

        if not self._quiet:
            self._render()
            sys.stdout.write("\n")
            sys.stdout.flush()

    # ── internal ──────────────────────────────────────────────────────────────

    def _harvest(self):
        finished = [(p, ab) for p, ab in list(self._running.items())
                    if p.poll() is not None]
        for proc, (na, nb, frozen_sides) in finished:
            del self._running[proc]
            winner = None
            traj_blob = None
            try:
                out = proc.stdout.read().strip()
                parsed = json.loads(out)
                winner    = parsed["winner"]
                traj_blob = parsed.get("traj_blob")
            except Exception:
                pass
            sa = 1.0 if winner == na else 0.5 if winner is None else 0.0
            # 矩陣（勝負紀錄）不管凍結與否都照記；ELO 只有非凍結側才更新。
            _elo_update(self._elo, na, nb, sa,
                        skip_a=("a" in frozen_sides), skip_b=("b" in frozen_sides))
            _record(self._matrix, winner, na, nb)
            if self._on_harvest is not None:
                self._on_harvest(na, nb, winner, frozen_sides, self._done + 1, self._total)
            self._done += 1
            if self._mode in ("train", "cross", "balance") and self._done < self._total:
                self._queue += [na, nb]
            if self._cross and traj_blob:
                self._ingest_traj(traj_blob)

    def _spawn(self):
        n_cores = os.cpu_count() or 8
        if self._mode == "eval":
            self._spawn_eval(n_cores)
        elif self._mode == "cross":
            self._spawn_cross(n_cores)
        elif self._mode == "filling":
            self._spawn_filling(n_cores)
        elif self._mode == "balance":
            self._spawn_balance(n_cores)
        else:
            self._spawn_train(n_cores)

    def _pair_cap(self, n_cores):
        """Train/Cross/Balance 並行上限：

        Train/Balance 搭配輪次制：每模型每輪最多一場，允許全模型同時運行（n//2 對）
        Cross 模式：保留人手閒置，避免固定配對（n//2 - 1 對）
        """
        n = len(self._names)
        if self._mode in ("train", "balance"):
            # 輪次制：全模型運行，但每模型最多一場 → n//2 對
            return min(n_cores, n // 2)
        else:  # cross
            # Cross 仍保留人手閒置
            return min(n_cores, max(1, n // 2 - 1))

    def _elo_weight(self, elo_a, elo_b):
        """ELO 加權函數：差距越小，權重越高。指數衰減係數 200.0"""
        elo_diff = abs(elo_a - elo_b)
        return math.exp(-elo_diff / 200.0)

    def _pick_pair(self, pool):
        """配對選擇：
        - Train/Balance 輪次制：只從本輪未配對的模型中選擇
        - 加權隨機：ELO 差距小的配對概率高
        """
        # 輪次制（train/balance 模式）
        if self._mode in ("train", "balance"):
            # 篩選本輪未配對的模型
            available = [n for n in pool if n not in self._round_matched]
            if len(available) < 2:
                # 本輪全完成，進入下一輪
                self._round += 1
                self._round_matched.clear()
                available = pool

            pool = available

        # 加權隨機配對（考慮 ELO 差距）
        if len(pool) < 2:
            return None, None  # 無法配對

        # 生成所有可能的配對及其權重
        pairs = []
        weights = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                na, nb = pool[i], pool[j]
                elo_a = self._elo.get(na, DEFAULT_ELO)
                elo_b = self._elo.get(nb, DEFAULT_ELO)
                weight = self._elo_weight(elo_a, elo_b)
                pairs.append((na, nb))
                weights.append(weight)

        # 加權隨機抽樣
        if not pairs:
            return None, None

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        na, nb = random.choices(pairs, weights=weights, k=1)[0]

        return na, nb

    def _spawn_train(self, n_cores):
        """Train 模式：輪次制，每輪每模型最多一場對局。"""
        cap = self._pair_cap(n_cores)

        # 輪次開始時，洗牌隊列避免順序固定
        if not self._round_matched and self._queue:
            random.shuffle(self._queue)

        while len(self._queue) >= 2 and len(self._running) < cap:
            if _psutil and self._cpu_pct >= self._cpu_target:
                break
            na, nb = self._pick_pair(self._queue)
            if na is None or nb is None:
                break
            self._queue.remove(na)
            self._queue.remove(nb)
            # 記錄本輪已配對
            self._round_matched.add(na)
            self._round_matched.add(nb)
            self._last_opp[na] = nb
            self._last_opp[nb] = na
            self._launch(na, nb)

    def _spawn_cross(self, n_cores):
        """Cross 模式：與 train 相同隊列邏輯，但排除正在 consume 的 agent。"""
        cap = self._pair_cap(n_cores)
        available = [q for q in self._queue if q not in self._consuming]
        while len(available) >= 2 and len(self._running) < cap:
            if _psutil and self._cpu_pct >= self._cpu_target:
                break
            na, nb = self._pick_pair(available)
            self._queue.remove(na)
            self._queue.remove(nb)
            available.remove(na)
            available.remove(nb)
            self._last_opp[na] = nb
            self._last_opp[nb] = na
            self._launch(na, nb)

    def _ingest_traj(self, blob: str) -> None:
        """解碼軌跡 blob，將兩個視角的條目加入 cross_buffer。"""
        try:
            traj = pickle.loads(base64.b64decode(blob))
        except Exception:
            return
        for entries in (traj.get("entries_a", []), traj.get("entries_b", [])):
            for e in entries:
                e["id"]          = str(uuid.uuid4())
                e["consumed_by"] = set()
                self._cross_buffer.append(e)

    def _save_cross_buffer(self) -> None:
        try:
            with open(CROSS_BUFFER_FILE, "wb") as f:
                pickle.dump(self._cross_buffer, f)
        except Exception:
            pass

    def _spawn_consume(self) -> None:
        """為每個閒置的 learner agent 啟動 consume 子進程。"""
        if not self._cross_buffer:
            return
        self._save_cross_buffer()
        for name in self._learner_names:
            if name in self._consuming:
                continue
            if name not in self._queue:
                continue   # 正在對局，避免存檔衝突
            has_work = any(
                name not in e.get("consumed_by", set())
                for e in self._cross_buffer
            )
            if not has_work:
                continue
            proc = subprocess.Popen(
                [sys.executable, _CONSUME_WORKER, name, CROSS_BUFFER_FILE],
                # stderr=DEVNULL（不是 PIPE）：_harvest_consume() 只讀 stdout，
                # 從不讀 stderr。用 PIPE 又不讀會讓子行程寫滿 OS pipe 緩衝區後
                # 卡死在 write()——這正是造成長時間「跑不完」的成因，見 _launch()
                # 同樣的修正與說明。
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
            )
            self._consume_procs[proc] = name
            self._consuming.add(name)

    def _harvest_consume(self) -> None:
        """收集已完成的 consume 進程結果，標記已消費條目，清理 buffer。"""
        done = [(p, n) for p, n in list(self._consume_procs.items())
                if p.poll() is not None]
        for proc, name in done:
            del self._consume_procs[proc]
            self._consuming.discard(name)
            try:
                out = proc.stdout.read().strip()
                result = json.loads(out)
                for eid in result.get("consumed_ids", []):
                    for e in self._cross_buffer:
                        if e["id"] == eid:
                            e["consumed_by"].add(name)
                            break
            except Exception:
                pass
        if done:
            self._cross_buffer = [
                e for e in self._cross_buffer
                if not e.get("consumed_by", set()) >= self._learner_names
            ]

    def _spawn_eval(self, n_cores):
        """Eval 模式：不存檔，同一 agent 可同時參與多場，上限由爬坡自適應。
        每次主迴圈最多新開 2 個進程，避免批量啟動造成 CPU 瞬間爆衝。"""
        limit = self._dynamic_max if _psutil else n_cores
        spawned = 0
        while len(self._running) < limit and spawned < 2:
            if self._done + len(self._running) >= self._total:
                break
            na, nb = self._pick_eval_pair()
            self._launch(na, nb)
            spawned += 1

    def _spawn_filling(self, n_cores):
        """Filling 模式：不存檔，並行邏輯同 eval，但配對來自固定的補局隊列。"""
        limit = self._dynamic_max if _psutil else n_cores
        spawned = 0
        while len(self._running) < limit and spawned < 2 and self._fill_queue:
            na, nb = self._fill_queue.pop()
            self._launch(na, nb)
            spawned += 1

    def _spawn_balance(self, n_cores):
        """Balance 模式：動態補局 + 加權隨機選擇對手（考慮 ELO）。"""
        cap = self._pair_cap(n_cores)

        # Balance 模式：動態計算補局需求
        if self._balance_target and not self._balance_needs:
            for name in self._names:
                g = 0
                for other in self._names:
                    if other == name:
                        continue
                    wa, wb, d = _h2h(self._matrix, name, other)
                    g += wa + wb + d
                need = max(0, self._balance_target - g)
                if need > 0:
                    self._balance_needs[name] = need

        # 輪次開始時，洗牌隊列避免順序固定
        if not self._round_matched and self._queue:
            random.shuffle(self._queue)

        spawned = 0
        while len(self._running) < cap and spawned < 2:
            if _psutil and self._cpu_pct >= self._cpu_target:
                break

            na = None
            nb = None

            # Balance 模式：優先選對局數不足的模型
            if self._balance_needs:
                # 從需要補局的模型中選一個（加權隨機）
                need_agents = [n for n, need in self._balance_needs.items() if need > 0]
                if need_agents:
                    # 優先補需求多的
                    weights = [self._balance_needs[n] for n in need_agents]
                    na = random.choices(need_agents, weights=weights, k=1)[0]

                    # 為其選擇對手（加權隨機，考慮 ELO）
                    candidates = [n for n in self._names if n != na and n not in self._round_matched]
                    if candidates:
                        # 生成配對及其權重
                        pairs = []
                        weights = []
                        for c in candidates:
                            elo_a = self._elo.get(na, DEFAULT_ELO)
                            elo_c = self._elo.get(c, DEFAULT_ELO)
                            weight = self._elo_weight(elo_a, elo_c)
                            pairs.append(c)
                            weights.append(weight)
                        nb = random.choices(pairs, weights=weights, k=1)[0]

            # 備選：從隊列選（輪次制）
            elif len(self._queue) >= 2:
                na, nb = self._pick_pair(self._queue)
                if na is None or nb is None:
                    break
                self._queue.remove(na)
                self._queue.remove(nb)
                self._round_matched.add(na)
                self._round_matched.add(nb)
            else:
                break

            if na and nb:
                self._last_opp[na] = nb
                self._last_opp[nb] = na
                self._launch(na, nb)

                # 更新補局需求
                if self._balance_needs:
                    self._balance_needs[na] = max(0, self._balance_needs.get(na, 0) - 1)
                    self._balance_needs[nb] = max(0, self._balance_needs.get(nb, 0) - 1)
                    # 清理完成的
                    self._balance_needs = {n: need for n, need in self._balance_needs.items() if need > 0}

                spawned += 1

    def _pick_eval_pair(self):
        """ELO 最近鄰配對，加小幅隨機擾動避免永遠只選同一對。"""
        noise = 80.0
        scored = sorted(
            self._names,
            key=lambda n: self._elo.get(n, DEFAULT_ELO) + random.gauss(0, noise),
        )
        best_i = min(
            range(len(scored) - 1),
            key=lambda i: abs(self._elo.get(scored[i],   DEFAULT_ELO) -
                              self._elo.get(scored[i+1], DEFAULT_ELO)),
        )
        return scored[best_i], scored[best_i + 1]

    def _launch(self, na, nb):
        # filling 對 worker 而言行為應與 eval 完全一致（不存檔、凍結 player_mode）
        # balance 對 worker 而言行為應與 train 一致（存檔至 .1 路徑）
        if self._mode == "filling":
            worker_mode = "eval"
        elif self._mode == "balance":
            worker_mode = "train"
        else:
            worker_mode = self._mode

        frozen_sides = set()
        if na in self._frozen_names:
            frozen_sides.add("a")
        if nb in self._frozen_names:
            frozen_sides.add("b")

        cmd = [sys.executable, _WORKER, na, nb, worker_mode]
        if frozen_sides:
            cmd.append(",".join(sorted(frozen_sides)))
        proc = subprocess.Popen(
            cmd,
            # stderr=DEVNULL（不是 PIPE）：_harvest() 只在 proc 結束後讀
            # proc.stdout，從不讀 proc.stderr。兩邊都開 PIPE 卻只讀一邊，一旦
            # 子行程往 stderr 寫的量超過 OS pipe 緩衝區（例如 PyTorch/numpy 每
            # 步都重複觸發同一個 warning，跑幾百步就會累積超過），子行程會卡在
            # write() 系統呼叫上永遠等不到人讀，看起來就是「跑不完、也連不上」
            # ——這才是先前對局長時間卡住、且無法從別的 session 終止的真正成因，
            # 不是遊戲邏輯本身跑了很多回合（回合數上限見 experiment.py 的
            # MAX_ROUNDS，那是另一個獨立、真實存在但這次沒踩到的已知問題）。
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
        self._running[proc] = (na, nb, frozen_sides)

    # ── ANSI 原地刷新 ─────────────────────────────────────────────────────────

    def _adjust_max(self):
        """每 5 秒根據 CPU 爬坡調整 eval 模式的並行上限。"""
        if not _psutil:
            return
        now = time.perf_counter()
        if now - self._last_ramp < 5.0:
            return
        self._last_ramp = now
        n_cores = os.cpu_count() or 8
        if self._cpu_pct < self._ramp_up_thr:
            self._dynamic_max = min(n_cores, self._dynamic_max + 1)
        elif self._cpu_pct > self._ramp_dn_thr:
            self._dynamic_max = max(1, self._dynamic_max - 1)

    def _render(self):
        lines = self._build_display()
        term_w = shutil.get_terminal_size((100, 24)).columns
        if self._display_lines:
            os.system("cls" if os.name == "nt" else "clear")
        for line in lines:
            sys.stdout.write(f"{line}\n")
        sys.stdout.flush()
        self._display_lines = sum(_term_rows(line, term_w) for line in lines)

    def _build_display(self):
        CW  = 6    # 矩陣資料欄寬
        RL  = 10   # 行標籤寬（"MCT V5.1 "）

        elapsed = time.perf_counter() - self._start_time
        mm, ss  = divmod(int(elapsed), 60)
        hh, mm  = divmod(mm, 60)
        t_str   = f"{hh}h{mm:02d}m{ss:02d}s" if hh else f"{mm}m{ss:02d}s"
        n_cores = os.cpu_count() or 8
        cpu_str = (f"{self._cpu_pct:.0f}%/{self._cpu_target:.0f}%"
                   if _psutil else "N/A")
        if self._mode == "eval" and _psutil:
            par_ceil = self._dynamic_max
            arrow = ("↑" if self._cpu_pct < self._ramp_up_thr
                     else "↓" if self._cpu_pct > self._ramp_dn_thr
                     else "=")
            par_str = f"{len(self._running)}/{par_ceil}{arrow}"
        else:
            par_str = f"{len(self._running)}/{n_cores}"

        n  = len(self._names)
        W  = max(54, RL + 2 + CW * n)
        sep  = "═" * W
        thin = "─" * W

        lines = [sep]
        lines.append(
            f"  Arena {self._mode.capitalize()}  {n}智能體  "
            f"{self._done}/{self._total}局  {t_str}  "
            f"並行:{par_str}  CPU:{cpu_str}"
        )
        lines.append(sep)

        # ── 排名表：系列 / 版本 / ELO / G / W / W% ──
        lines.append(f"  {'#':>2}  {'系列':<4}  {'版本':<5}  "
                     f"{'ELO':>6}  {'G':>5}  {'W':>5}  {'W%':>6}")
        lines.append(f"  {'─'*(W-2)}")
        order = sorted(self._names,
                       key=lambda n: self._elo.get(n, DEFAULT_ELO), reverse=True)
        for rank, name in enumerate(order, 1):
            series, ver = _split_name(name)
            g = w = 0
            for other in self._names:
                if other == name:
                    continue
                wa, wb, dd = _h2h(self._matrix, name, other)
                g += wa + wb + dd
                w += wa
            wr = f"{w/g:.1%}" if g else "—"
            lines.append(
                f"  {rank:>2}  {series:<4}  {ver:<5}  "
                f"{self._elo.get(name, DEFAULT_ELO):>6.0f}  "
                f"{g:>5}  {w:>5}  {wr:>6}"
            )
        lines.append(sep)

        # ── 勝率矩陣：雙行表頭（系列 / 版本）──
        splits  = [_split_name(n) for n in self._names]
        pad     = " " * (RL + 2)
        lines.append("  頭對頭勝率（行 vs 列）")
        lines.append(pad + "".join(f"{s:>{CW}}" for s, _ in splits))
        lines.append(pad + "".join(f"{v:>{CW}}" for _, v in splits))
        lines.append(f"  {thin}")
        for i, na in enumerate(self._names):
            sr, vr = splits[i]
            label  = f"{sr} {vr}"
            row    = f"  {label:<{RL}}"
            for j, nb in enumerate(self._names):
                if na == nb:
                    row += f"{'—':>{CW}}"
                else:
                    wa, wb, d = _h2h(self._matrix, na, nb)
                    t = wa + wb + d
                    row += f"{wa/t:>{CW}.0%}" if t else f"{'?':>{CW}}"
            lines.append(row)
        lines.append(sep)

        return lines


# ─── 顯示 ────────────────────────────────────────────────────────────────────

def display_leaderboard(elo, matrix, roster):
    rows = []
    for name in roster:
        g = w = d = 0
        for other in roster:
            if other == name:
                continue
            wa, wb, dd = _h2h(matrix, name, other)
            g += wa + wb + dd
            w += wa
            d += dd
        rows.append((elo.get(name, DEFAULT_ELO), name, g, w, d))
    rows.sort(reverse=True)

    print(f"\n{'='*64}\n  排名榜\n{'='*64}")
    print(f"  {'#':>3}  {'Agent':<18}  {'ELO':>6}  {'G':>5}  {'W':>5}  {'D':>4}  {'W%':>6}")
    print(f"  {'-'*58}")
    for rank, (e, name, g, w, d) in enumerate(rows, 1):
        if g:
            print(f"  {rank:>3}  {name:<18}  {e:>6.0f}  {g:>5}  {w:>5}  {d:>4}  {w/g:>6.1%}")
        else:
            print(f"  {rank:>3}  {name:<18}  {e:>6.0f}  {'—':>5}  {'—':>5}  {'—':>4}  {'—':>6}")


def display_matrix(matrix, roster):
    W = 9
    ab = [n[:W-1] for n in roster]
    width = 12 + W * len(roster)
    print(f"\n{'='*width}\n  頭對頭勝率（列方對欄方）\n{'='*width}")
    print(f"  {'':10}" + "".join(f"{a:>{W}}" for a in ab))
    print(f"  {'-'*(10 + W * len(roster))}")
    for i, na in enumerate(roster):
        row = f"  {ab[i]:<10}"
        for nb in roster:
            if na == nb:
                row += f"{'—':>{W}}"
            else:
                wa, wb, d = _h2h(matrix, na, nb)
                t = wa + wb + d
                row += f"{wa/t:>{W-1}.1%} " if t else f"{'?':>{W}}"
        print(row)


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    # --help
    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    TRAIN      = "--train"       in args
    CROSS      = "--cross"       in args
    QUICK      = "--quick"       in args
    FILLING    = "--filling"     in args
    BALANCE    = "--balance"     in args
    NO_SAVE    = "--no-save"     in args
    RESET      = "--reset"       in args
    BOARD_ONLY = "--leaderboard" in args
    VERBOSE    = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--train","--cross","--quick","--filling","--balance","--no-save","--reset","--leaderboard","--verbose","-v")]

    # --cpu N（CPU 使用率上限，預設 60）
    CPU_LIMIT = CPU_TARGET
    if "--cpu" in args:
        idx = args.index("--cpu")
        CPU_LIMIT = float(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    # --agents（僅 eval 模式使用）
    roster = None
    if "--agents" in args:
        idx = args.index("--agents")
        roster, j = [], idx + 1
        while j < len(args) and not args[j].startswith("-"):
            roster.append(args[j])
            j += 1
        args = args[:idx] + args[j:]

    _default_n = 500 if (TRAIN or CROSS) else (30 if FILLING else (330 if BALANCE else 300))
    N = int(next((a for a in args if not a.startswith("-")), _default_n))


    elo    = {} if RESET else _load(ELO_FILE, {})
    matrix = {} if RESET else _load(MATRIX_FILE, {})

    # ── --quick：V4~V5 快捷循環賽 ──────────────────────────────────────────
    if QUICK:
        roster = QUICK_ROSTER
        v5_names = {
            MCTAgentV5.__name__, MCCFRAgentV5.__name__,
            MCTAgentV5_1.__name__, MCCFRAgentV5_1.__name__,
        }
        pairs = [(na, nb) for na, nb in itertools.combinations(roster, 2)
                 if na in v5_names or nb in v5_names]
        print(f"\n{'='*60}")
        print(f"  Arena Quick 模式  V4~V5  {len(pairs)} 對  每對 {N} 局")
        print(f"  （N={N} 為每個 V5 對各對手的對局數）")
        print(f"{'='*60}")
        for i, (na, nb) in enumerate(pairs, 1):
            print(f"\n  [{i}/{len(pairs)}] {na}  vs  {nb}")
            run_eval_pair(AGENT_MAP[na], AGENT_MAP[nb], N, elo, matrix, VERBOSE)
            wa, wb, d = _h2h(matrix, na, nb)
            t = wa + wb + d
            print(f"         {na}: {wa}W ({wa/t:.1%})  {nb}: {wb}W ({wb/t:.1%})  Draw: {d}")
        display_leaderboard(elo, matrix, roster)
        display_matrix(matrix, roster)
        if not NO_SAVE:
            _dump(ELO_FILE, elo)
            _dump(MATRIX_FILE, matrix)
            print(f"\n  [已儲存 → {ELO_FILE}, {MATRIX_FILE}]")
        sys.exit(0)

    # ── --leaderboard：只顯示 ───────────────────────────────────────────────
    if BOARD_ONLY:
        disp_roster, _ = _resolve_roster(roster or DEFAULT_ROSTER)
        display_leaderboard(elo, matrix, disp_roster)
        display_matrix(matrix, disp_roster)
        sys.exit(0)

    # ── --filling：補滿對局數 < N 的配對至 N 局（不支援 --no-save，存檔是其唯一目的）──
    if FILLING:
        fill_roster = roster or DEFAULT_ROSTER
        fill_names, bad = _resolve_roster(fill_roster)
        if bad:
            print(f"  [錯誤] 不認識的 agent：{bad}")
            sys.exit(1)

        deficits, fill_queue = [], []
        for na, nb in itertools.combinations(fill_names, 2):
            wa, wb, d = _h2h(matrix, na, nb)
            total = wa + wb + d
            if total < N:
                deficit = N - total
                deficits.append((na, nb, total, deficit))
                fill_queue += [(na, nb)] * deficit

        print(f"\n{'='*60}")
        print(f"  Arena Filling 模式  補滿對局數 < {N} 的配對")
        print(f"{'='*60}")
        if not deficits:
            print(f"  所有配對皆已達 {N} 局，無需補滿。")
        else:
            print(f"  {len(deficits)} 對需補滿，共 {len(fill_queue)} 局")
            for na, nb, total, deficit in deficits:
                print(f"    {na}  vs  {nb}：目前 {total} 局 → 補 {deficit} 局")
            arena = MatchmakingArena(fill_names, elo, matrix, len(fill_queue),
                                      mode="filling", cpu_target=CPU_LIMIT,
                                      fill_pairs=fill_queue)
            arena.run()

        display_leaderboard(elo, matrix, fill_names)
        display_matrix(matrix, fill_names)
        _dump(ELO_FILE, elo)
        _dump(MATRIX_FILE, matrix)
        print(f"\n  [已儲存 → {ELO_FILE}, {MATRIX_FILE}]")
        sys.exit(0)

    # ── --balance：平衡各模型的訓練對局數 ────────────────────────────────────
    if BALANCE:
        balance_roster = roster or DEFAULT_ROSTER
        balance_names, bad = _resolve_roster(balance_roster)
        if bad:
            print(f"  [錯誤] 不認識的 agent：{bad}")
            sys.exit(1)

        # 計算各模型的當前總對局數
        model_counts = {}
        for name in balance_names:
            g = 0
            for other in balance_names:
                if other == name:
                    continue
                wa, wb, d = _h2h(matrix, name, other)
                g += wa + wb + d
            model_counts[name] = g

        min_count = min(model_counts.values()) if model_counts else 0
        max_count = max(model_counts.values()) if model_counts else 0

        print(f"\n{'='*60}")
        print(f"  Arena Balance 模式  補平訓練對局數至 {N} 局")
        print(f"{'='*60}")
        print(f"  當前最少: {min_count} 局  最多: {max_count} 局")

        # 計算補局總數
        total_balance_games = sum(max(0, N - model_counts[name]) for name in balance_names)

        if total_balance_games == 0:
            print(f"  所有模型皆已達 {N} 局，無需補平。")
        else:
            print(f"  約需 {total_balance_games} 場補局")
            # 不預生成隊列，改為動態選擇（在 arena 中）
            arena = MatchmakingArena(balance_names, elo, matrix, total_balance_games,
                                      mode="balance", cpu_target=CPU_LIMIT,
                                      balance_target=N, reset_buf=RESET)
            arena.run()

        display_leaderboard(elo, matrix, balance_names)
        display_matrix(matrix, balance_names)
        _dump(ELO_FILE, elo)
        _dump(MATRIX_FILE, matrix)
        print(f"\n  [已儲存 → {ELO_FILE}, {MATRIX_FILE}]")
        sys.exit(0)

    # ── Train / Cross 模式 ──────────────────────────────────────────────────
    if TRAIN or CROSS:
        train_roster = roster or DEFAULT_ROSTER
        train_names, bad = _resolve_roster(train_roster)
        if bad:
            print(f"  [錯誤] 不認識的 agent：{bad}")
            sys.exit(1)
        mode = "cross" if CROSS else "train"
        if CROSS:
            print(f"\n  [Cross 模式] 跨視角共用 buffer，全體 agent 互學")
            print(f"  MCCFR → regret update  |  MCT → UCB 累積")
            print(f"  存檔路徑：.2（*_2_state.pkl）")
        arena = MatchmakingArena(train_names, elo, matrix, N, mode=mode,
                                 cpu_target=CPU_LIMIT, reset_buf=RESET)
        arena.run()
        display_leaderboard(elo, matrix, train_names)
        display_matrix(matrix, train_names)
        if not NO_SAVE:
            _dump(ELO_FILE, elo)
            _dump(MATRIX_FILE, matrix)
            print(f"\n  [已儲存 → {ELO_FILE}, {MATRIX_FILE}]")
        sys.exit(0)

    # ── Eval 模式：MMR 匹配賽 ─────────────────────────────────────────────
    if roster is None:
        roster = DEFAULT_ROSTER
    roster, bad = _resolve_roster(roster)
    if bad:
        print(f"  [錯誤] 不認識的 agent：{bad}")
        print(f"  可用：{sorted(set(AGENT_MAP))}")
        print(f"  （V1~V4 名稱可加 _1 或 .1 字尾，代表讀取 arena 訓練的 .1 存檔，與基底版本分開計分）")
        sys.exit(1)

    arena = MatchmakingArena(roster, elo, matrix, N, mode="eval", cpu_target=CPU_LIMIT)
    arena.run()

    display_leaderboard(elo, matrix, roster)
    display_matrix(matrix, roster)

    if not NO_SAVE:
        _dump(ELO_FILE, elo)
        _dump(MATRIX_FILE, matrix)
        print(f"\n  [已儲存 → {ELO_FILE}, {MATRIX_FILE}]")
