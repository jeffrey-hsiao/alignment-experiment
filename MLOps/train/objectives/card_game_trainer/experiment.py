"""
實驗腳本：比較 MCT vs MCCFR

量測指標：
  1. 對 Random 的勝率（各跑 N 局）
  2. MCT vs MCCFR 直接對戰勝率
  3. 可利用性近似：best-response（全知 MCT）對各智能體的勝率
"""
from __future__ import annotations
import os
import random
import sys
from game_engine import GameState, Character
# 回合數安全上限與 self-play 共用同一個定義（原因見 game_runner.py 的說明）
from game_runner import MAX_ROUNDS
from agents.random_agent import RandomAgent
from agents.mct_agent import MCTAgent
from agents.mct_agent_v2 import MCTAgentV2
from agents.mct_agent_v3 import MCTAgentV3
from agents.mct_agent_v4 import MCTAgentV4
from agents.mct_agent_v5 import MCTAgentV5
from agents.mct_agent_v5_1 import MCTAgentV5_1
from agents.mccfr_agent_v5_1 import MCCFRAgentV5_1
from agents.mccfr_agent_v2 import MCCFRAgentV2
from agents.mccfr_agent_v3 import MCCFRAgentV3
from agents.mccfr_agent_v4 import MCCFRAgentV4
from agents.mccfr_agent_v5 import MCCFRAgentV5
from agents.mccfr_agent import MCCFRAgent
# ⚠️ V6 已棄用：架構設計失敗（詳見 arena.py 註記）
from agents.mct_agent_v6 import MCTAgentV6
from agents.mccfr_agent_v6 import MCCFRAgentV6

SAVE_FILES = {
    "MCTAgent":     "mct_state.pkl",
    "MCCFRAgent":   "mccfr_state.pkl",
    "MCTAgentV2":   "mct_v2_state.pkl",
    "MCTAgentV3":   "mct_v3_state.pkl",
    "MCTAgentV4":   "mct_v4_state.pkl",
    "MCTAgentV5":    "mct_v5_state.pkl",
    "MCTAgentV5_1":  "mct_v5_1_state.pkl",
    "MCCFRAgentV5_1":"mccfr_v5_1_state.pkl",
    "MCCFRAgentV2":  "mccfr_v2_state.pkl",
    "MCCFRAgentV3": "mccfr_v3_state.pkl",
    "MCCFRAgentV4": "mccfr_v4_state.pkl",
    "MCCFRAgentV5": "mccfr_v5_state.pkl",
    # ⚠️ V6 已棄用
    "MCTAgentV6":   "mct_v6_state.pkl",
    "MCCFRAgentV6": "mccfr_v6_state.pkl",
}


def _fmt(hand): return "[" + " ".join(str(c) for c in sorted(hand)) + "]"


def run_game(agent0, agent1, seed=None, show: bool = False, max_rounds: int = MAX_ROUNDS) -> int:
    """回傳勝者 player_id（0 或 1），平局或撞到 max_rounds 回傳 -1。"""
    gs = GameState(seed=seed)
    for agent in (agent0, agent1):
        if hasattr(agent, 'new_game'):
            agent.new_game()
    n0 = type(agent0).__name__
    n1 = type(agent1).__name__

    if show:
        print(f"\n{'='*60}")
        print(f"  {n0}  vs  {n1}")
        print(f"{'='*60}")

    while not gs.is_terminal() and gs.round_num < max_rounds:
        p0, p1 = gs.players
        if show:
            print(f"\n  -- Round {gs.round_num + 1} --")
            print(f"  {n0:<15} HP:{p0.hp:>3}  DEF:{p0.defense}  hand:{_fmt(p0.hand)}")
            print(f"  {n1:<15} HP:{p1.hp:>3}  DEF:{p1.defense}  hand:{_fmt(p1.hand)}")

        a0 = agent0.choose_action(gs)
        a1 = agent1.choose_action(gs)
        result = gs.apply_round(a0, a1)
        agent0.notify_result(gs)
        agent1.notify_result(gs)

        if show:
            print(f"  {n0} plays: {sorted(a0)} (sum={sum(a0)})")
            print(f"  {n1} plays: {sorted(a1)} (sum={sum(a1)})")
            if result.winner is None:
                tag = f"DRAW  diff={result.difference}"
            elif result.winner == 0:
                tag = f"{n0} wins  diff={result.difference}  reward=[{result.reward_card}]"
            else:
                tag = f"{n1} wins  diff={result.difference}  reward=[{result.reward_card}]"
            print(f"  >> {tag}  HP: {n0}={result.p0_hp_after}  {n1}={result.p1_hp_after}")

    winner = gs.winner() if gs.winner() is not None else -1
    if show:
        name = n0 if winner == 0 else (n1 if winner == 1 else "Draw")
        print(f"\n  WINNER: {name}  ({gs.round_num} rounds)")
    return winner


def run_match(agent0_cls, agent1_cls, n_games: int, verbose: bool = False,
              selfplay: bool = False) -> dict:
    a0 = agent0_cls(0)
    a1 = agent1_cls(1, is_target=True) if selfplay else agent1_cls(1)
    for agent, cls in [(a0, agent0_cls), (a1, agent1_cls)]:
        path = SAVE_FILES.get(cls.__name__)
        if path and os.path.exists(path) and hasattr(agent, 'load'):
            agent.load(path)
    if selfplay and hasattr(a0, 'set_target_agent'):
        a0.set_target_agent(a1)

    wins = [0, 0, 0]  # [agent0 wins, agent1 wins, draws]
    for i in range(n_games):
        result = run_game(a0, a1)

        if result == 0:
            wins[0] += 1
        elif result == 1:
            wins[1] += 1
        else:
            wins[2] += 1

        if selfplay and hasattr(a1, 'soft_update_from'):
            a1.soft_update_from(a0)

        if verbose and (i + 1) % 10 == 0:
            extra = ""
            if selfplay and hasattr(a0, '_last_loss'):
                extra = (f"  loss={a0._last_loss:.4f}"
                         f"  ε={a0._epsilon:.3f}")
            print(f"  {i+1}/{n_games}  {agent0_cls.__name__} {wins[0]}W  "
                  f"{agent1_cls.__name__} {wins[1]}W  Draw {wins[2]}{extra}", flush=True)

    # selfplay 時只存 online agent（a0），target 的權重由軟更新衍生，不需單獨保存
    save_list = [(a0, agent0_cls)] if selfplay else [(a0, agent0_cls), (a1, agent1_cls)]
    for agent, cls in save_list:
        path = SAVE_FILES.get(cls.__name__)
        if path and hasattr(agent, 'save'):
            agent.save(path)

    return {
        "agent0": agent0_cls.__name__,
        "agent1": agent1_cls.__name__,
        "n_games": n_games,
        "wins_0": wins[0],
        "wins_1": wins[1],
        "draws": wins[2],
        "winrate_0": wins[0] / n_games,
        "winrate_1": wins[1] / n_games,
    }


def run_winrate_over_time(agent_cls, opponent_cls, n_games: int, window: int = 10) -> list[float]:
    """回傳每 window 局的滾動勝率，用於收斂曲線。"""
    results = []
    for i in range(n_games):
        a = agent_cls(0)
        opp = opponent_cls(1)
        result = run_game(a, opp)
        results.append(1 if result == 0 else 0)

    smoothed = []
    for i in range(len(results)):
        start = max(0, i - window + 1)
        smoothed.append(sum(results[start:i+1]) / (i - start + 1))
    return smoothed


def run_rounds_match(agent0_cls, agent1_cls, n_games: int, verbose: bool = False) -> dict:
    """回傳各局回合數的統計（min / avg / median / max）。"""
    a0 = agent0_cls(0)
    a1 = agent1_cls(1)
    for agent, cls in [(a0, agent0_cls), (a1, agent1_cls)]:
        path = SAVE_FILES.get(cls.__name__)
        if path and os.path.exists(path) and hasattr(agent, 'load'):
            agent.load(path)

    rounds_list = []
    wins = [0, 0, 0]
    for i in range(n_games):
        gs = GameState()
        for agent in (a0, a1):
            if hasattr(agent, 'new_game'):
                agent.new_game()
        while not gs.is_terminal():
            a0_act = a0.choose_action(gs)
            a1_act = a1.choose_action(gs)
            gs.apply_round(a0_act, a1_act)
            a0.notify_result(gs)
            a1.notify_result(gs)
        rounds_list.append(gs.round_num)
        result = gs.winner() if gs.winner() is not None else -1
        if result == 0:   wins[0] += 1
        elif result == 1: wins[1] += 1
        else:             wins[2] += 1
        if verbose and (i + 1) % 10 == 0:
            cur_avg = sum(rounds_list) / len(rounds_list)
            print(f"  {i+1}/{n_games}  avg_rounds={cur_avg:.1f}  "
                  f"{agent0_cls.__name__} {wins[0]}W  {agent1_cls.__name__} {wins[1]}W  Draw {wins[2]}", flush=True)

    for agent, cls in [(a0, agent0_cls), (a1, agent1_cls)]:
        path = SAVE_FILES.get(cls.__name__)
        if path and hasattr(agent, 'save'):
            agent.save(path)
    rounds_list.sort()
    n = len(rounds_list)
    return {
        "agent0": agent0_cls.__name__,
        "agent1": agent1_cls.__name__,
        "n_games": n_games,
        "wins_0": wins[0], "wins_1": wins[1], "draws": wins[2],
        "winrate_0": wins[0] / n_games, "winrate_1": wins[1] / n_games,
        "min": rounds_list[0], "max": rounds_list[-1],
        "avg": sum(rounds_list) / n,
        "median": rounds_list[n // 2],
        "p25": rounds_list[n // 4],
        "p75": rounds_list[3 * n // 4],
    }


def print_rounds(r: dict):
    print(f"  {r['agent0']} vs {r['agent1']}  ({r['n_games']} games)")
    print(f"  {r['agent0']:15s}  勝率 {r['winrate_0']:.1%}  ({r['wins_0']} wins)")
    print(f"  {r['agent1']:15s}  勝率 {r['winrate_1']:.1%}  ({r['wins_1']} wins)")
    print(f"  平局 {r['draws']}")
    print(f"  rounds: min={r['min']}  p25={r['p25']}  median={r['median']}  "
          f"avg={r['avg']:.1f}  p75={r['p75']}  max={r['max']}")


def print_result(r: dict):
    print(f"\n  {r['agent0']} vs {r['agent1']}  ({r['n_games']} games)")
    print(f"  {r['agent0']:15s}  勝率 {r['winrate_0']:.1%}  ({r['wins_0']} wins)")
    print(f"  {r['agent1']:15s}  勝率 {r['winrate_1']:.1%}  ({r['wins_1']} wins)")
    print(f"  平局 {r['draws']}")


AGENT_MAP = {
    "MCTAgent":     MCTAgent,
    "MCT":          MCTAgent,
    "MCCFRAgent":   MCCFRAgent,
    "MCCFR":        MCCFRAgent,
    "MCTAgentV2":   MCTAgentV2,
    "MCTV2":        MCTAgentV2,
    "MCTAgentV3":   MCTAgentV3,
    "MCTV3":        MCTAgentV3,
    "MCTAgentV4":   MCTAgentV4,
    "MCTV4":        MCTAgentV4,
    "MCTAgentV5":    MCTAgentV5,
    "MCTV5":         MCTAgentV5,
    "MCTAgentV5_1":  MCTAgentV5_1,
    "MCTV5_1":       MCTAgentV5_1,
    "MCCFRAgentV2":  MCCFRAgentV2,
    "MCCFRV2":      MCCFRAgentV2,
    "MCCFRAgentV3": MCCFRAgentV3,
    "MCCFRV3":      MCCFRAgentV3,
    "MCCFRAgentV4": MCCFRAgentV4,
    "MCCFRV4":      MCCFRAgentV4,
    "MCCFRAgentV5":   MCCFRAgentV5,
    "MCCFRV5":        MCCFRAgentV5,
    "MCCFRAgentV5_1": MCCFRAgentV5_1,
    "MCCFRV5_1":      MCCFRAgentV5_1,
    # ⚠️ V6 已棄用：保留用於歷史查詢，但不在 arena 預設名單中
    "MCTAgentV6":    MCTAgentV6,
    "MCTV6":         MCTAgentV6,
    "MCCFRAgentV6":  MCCFRAgentV6,
    "MCCFRV6":       MCCFRAgentV6,
    "RandomAgent":    RandomAgent,
    "Random":       RandomAgent,
}


if __name__ == "__main__":
    args = sys.argv[1:]
    SHOW = "--show" in args
    RESTART = "--restart" in args
    ROUNDS = "--rounds" in args
    RANDOM = "--random" in args
    SELFPLAY = "--selfplay" in args
    args = [a for a in args if a not in ("--show", "--restart", "--rounds", "--random", "--selfplay")]

    # --match A B：指定對戰組合
    MATCH = None
    if "--match" in args:
        idx = args.index("--match")
        if idx + 2 <= len(args) - 1:
            a_name, b_name = args[idx + 1], args[idx + 2]
            if a_name not in AGENT_MAP or b_name not in AGENT_MAP:
                known = ", ".join(AGENT_MAP)
                print(f"  [錯誤] 不認識的 agent 名稱。可用：{known}")
                sys.exit(1)
            MATCH = (AGENT_MAP[a_name], AGENT_MAP[b_name])
            args = args[:idx] + args[idx + 3:]
        else:
            print("  [錯誤] --match 需要兩個 agent 名稱，例如：--match MCTAgent V2")
            sys.exit(1)

    N = int(args[0]) if args else 100

    if RESTART:
        for f in SAVE_FILES.values():
            if os.path.exists(f):
                os.remove(f)
                print(f"  [已刪除存檔：{f}]")
            dqn_f = f.replace('.pkl', '_dqn.pt')
            if os.path.exists(dqn_f):
                os.remove(dqn_f)
                print(f"  [已刪除存檔：{dqn_f}]")

    if SELFPLAY:
        cls = MATCH[0] if MATCH else MCTAgentV5
        print(f"=== 自我對戰：{cls.__name__}（online）vs {cls.__name__}（target），{N} 局 ===")
        print_result(run_match(cls, cls, N, verbose=True, selfplay=True))
        sys.exit(0)

    if MATCH:
        cls_a, cls_b = MATCH
        label = f"{cls_a.__name__} vs {cls_b.__name__}"
        if SHOW:
            print(f"=== --show 模式：{label} ===")
            a0 = cls_a(0)
            a1 = cls_b(1)
            for agent, cls in [(a0, cls_a), (a1, cls_b)]:
                path = SAVE_FILES.get(cls.__name__)
                if path and os.path.exists(path) and hasattr(agent, 'load'):
                    agent.load(path)
            run_game(a0, a1, show=True)
        elif ROUNDS:
            print(f"=== 回合數統計：{label}，{N} 局 ===")
            print_rounds(run_rounds_match(cls_a, cls_b, N, verbose=True))
        else:
            print(f"=== {label}，{N} 局 ===")
            print_result(run_match(cls_a, cls_b, N, verbose=True))
        sys.exit(0)

    if ROUNDS:
        print(f"=== 回合數統計，每組 {N} 局 ===")
        if RANDOM:
            print("\n[1] MCTAgent vs Random")
            print_rounds(run_rounds_match(MCTAgent, RandomAgent, N, verbose=True))
            print("\n[2] MCCFRAgent vs Random")
            print_rounds(run_rounds_match(MCCFRAgent, RandomAgent, N, verbose=True))
        print("\n[3] MCTAgent vs MCCFRAgent")
        print_rounds(run_rounds_match(MCTAgent, MCCFRAgent, N, verbose=True))
        sys.exit(0)

    if SHOW:
        # --show 模式：每組只跑 1 局，完整顯示過程
        print("=== --show 模式：各跑 1 局 ===")
        if RANDOM:
            print("\n[1] MCTAgent vs Random")
            run_game(MCTAgent(0), RandomAgent(1), show=True)
            print("\n[2] MCCFRAgent vs Random")
            run_game(MCCFRAgent(0), RandomAgent(1), show=True)
        print("\n[3] MCTAgent vs MCCFRAgent")
        run_game(MCTAgent(0), MCCFRAgent(1), show=True)
    else:
        print(f"=== 實驗開始，每組 {N} 局 ===")

        if RANDOM:
            print("\n[1] MCTAgent vs Random")
            r1 = run_match(MCTAgent, RandomAgent, N, verbose=True)
            print_result(r1)

            print("\n[2] MCCFRAgent vs Random")
            r2 = run_match(MCCFRAgent, RandomAgent, N, verbose=True)
            print_result(r2)

        print("\n[3] MCTAgent vs MCCFRAgent")
        r3 = run_match(MCTAgent, MCCFRAgent, N, verbose=True)
        print_result(r3)

        if RANDOM:
            print("\n[4] MCTAgentV2 vs Random")
            r4 = run_match(MCTAgentV2, RandomAgent, N, verbose=True)
            print_result(r4)

        print("\n[5] MCTAgentV2 vs MCTAgent")
        r5 = run_match(MCTAgentV2, MCTAgent, N, verbose=True)
        print_result(r5)

        print("\n結果摘要")
        if RANDOM:
            print(f"  MCT   對 Random 勝率：{r1['winrate_0']:.1%}")
            print(f"  MCCFR 對 Random 勝率：{r2['winrate_0']:.1%}")
        print(f"  MCT   對 MCCFR  勝率：{r3['winrate_0']:.1%}")
        if RANDOM:
            print(f"  MCTv2 對 Random 勝率：{r4['winrate_0']:.1%}")
        print(f"  MCTv2 對 MCT    勝率：{r5['winrate_0']:.1%}")
