#!/usr/bin/env python3
"""
LLMAgent V1 訓練腳本

阶段 1：格式錯誤即時 SFT
阶段 2：遊戲級經驗回放 RL

對手：MCTAgentV2（穩定基準）
"""
from __future__ import annotations
import os
import sys
import time
from datetime import datetime
from game_engine import GameState
from agents.llm_agent import LLMAgentV1
from agents.mct_agent_v2 import MCTAgentV2

SAVE_FILES = {
    "LLMAgentV1": "llm_v1_state",
    "MCTAgentV2": "mct_v2_state.pkl",
}

LOG_FILE = "llm_v1_training.log"


def write_log(message: str, max_retries: int = 3, retry_delay: float = 0.1) -> bool:
    """
    寫入日誌文件，帶重試機制以處理文件鎖定

    Args:
        message: 要寫入的日誌訊息
        max_retries: 最大重試次數
        retry_delay: 重試延遲（秒）

    Returns:
        是否成功寫入
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"

    for attempt in range(max_retries):
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_message)
            return True
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[WARNING] Failed to write log after {max_retries} attempts: {e}")
                return False
    return False


def _fmt(hand):
    return "[" + " ".join(str(c) for c in sorted(hand)) + "]"


def run_game(llm_agent: LLMAgentV1, mct_agent: MCTAgentV2, seed=None, show: bool = False) -> int:
    """運行單局遊戲，返回獲勝者 (0=LLM, 1=MCT, -1=平局)"""
    gs = GameState(seed=seed)

    for agent in (llm_agent, mct_agent):
        if hasattr(agent, 'new_game'):
            agent.new_game()

    if show:
        print(f"\n{'='*70}")
        print(f"  LLMAgentV1  vs  MCTAgentV2")
        print(f"{'='*70}")

    while not gs.is_terminal():
        p0, p1 = gs.players

        if show:
            print(f"\n  -- Round {gs.round_num + 1} --")
            print(f"  LLM                HP:{p0.hp:>3}  DEF:{p0.defense}  hand:{_fmt(p0.hand)}")
            print(f"  MCT                HP:{p1.hp:>3}  DEF:{p1.defense}  hand size:{len(p1.hand)}")

        a0 = llm_agent.choose_action(gs)
        a1 = mct_agent.choose_action(gs)

        result = gs.apply_round(a0, a1)
        llm_agent.notify_result(gs)
        mct_agent.notify_result(gs)

        if show:
            print(f"  LLM plays:  {sorted(a0)} (sum={sum(a0)})")
            print(f"  MCT plays:  {sorted(a1)} (sum={sum(a1)})")
            if result.winner is None:
                tag = f"DRAW  diff={result.difference}"
            elif result.winner == 0:
                tag = f"LLM wins  diff={result.difference}"
            else:
                tag = f"MCT wins  diff={result.difference}"
            print(f"  >> {tag}  HP: LLM={result.p0_hp_after}  MCT={result.p1_hp_after}")

    winner = gs.winner() if gs.winner() is not None else -1

    if show:
        name = "LLM" if winner == 0 else ("MCT" if winner == 1 else "Draw")
        print(f"\n  WINNER: {name}  ({gs.round_num} rounds)")
        stats = llm_agent.get_stats()
        print(f"  LLM Stats: {stats}")

    return winner


def create_agents(use_qwen: bool = True, use_lora: bool = False, use_int8: bool = False, log: bool = True):
    """
    初始化 LLMAgentV1 與 MCTAgentV2。

    這是訓練程序與外部診斷/測試工具（debug_tools/ 底下）共用的唯一初始化入口，
    確保診斷工具測到的 agent 跟正式訓練用的 agent 是同一套初始化邏輯（同樣不
    load 舊的 LLM checkpoint、同樣載入 mct_v2_state.pkl），不會因為兩邊各自維
    護一份邏輯而漂移不一致。

    Args:
        log: 是否寫入 llm_v1_training.log。診斷工具應傳 False，避免把測試紀錄
             跟正式訓練紀錄混在一起。
    """
    if use_qwen:
        try:
            llm_agent = LLMAgentV1(0, model_name="Qwen/Qwen2.5-0.5B-Instruct", use_lora=use_lora, use_int8=use_int8)
            extra_modes = []
            if use_lora:
                extra_modes.append("LoRA")
            if use_int8:
                extra_modes.append("INT8")
            mode_str = f" + {' + '.join(extra_modes)}" if extra_modes else ""
            print(f"[OK] Loaded Qwen2.5-0.5B-Instruct{mode_str}")
            if log:
                write_log(f"Model loaded: Qwen2.5-0.5B-Instruct{mode_str}")
        except Exception as e:
            print(f"[FAIL] Failed to load Qwen model: {e}")
            print(f"  Falling back to dummy LLM...")
            if log:
                write_log(f"Failed to load Qwen model: {e}, using dummy LLM")
            llm_agent = LLMAgentV1(0)
    else:
        llm_agent = LLMAgentV1(0)
        print(f"[OK] Using Dummy LLM for testing")
        if log:
            write_log(f"Using Dummy LLM for testing")

    mct_agent = MCTAgentV2(1)

    # 載入已有的存檔
    mct_path = SAVE_FILES.get("MCTAgentV2")
    if mct_path and os.path.exists(mct_path) and hasattr(mct_agent, 'load'):
        try:
            mct_agent.load(mct_path)
            print(f"[OK] Loaded MCTAgentV2 from {mct_path}\n")
            if log:
                write_log(f"Loaded MCTAgentV2 from {mct_path}")
        except Exception as e:
            print(f"[FAIL] Failed to load MCTAgentV2: {e}\n")
            if log:
                write_log(f"Failed to load MCTAgentV2: {e}")

    return llm_agent, mct_agent


def save_agents(llm_agent: LLMAgentV1, mct_agent: MCTAgentV2, tag: str = "") -> None:
    """保存 MCT / LLM 存檔。tag 只用於 print/log 訊息（例如標明是第幾局的中途存檔）。"""
    prefix = f"[Checkpoint @ {tag}] " if tag else ""

    if hasattr(mct_agent, 'save'):
        mct_path = SAVE_FILES["MCTAgentV2"]
        try:
            mct_agent.save(mct_path)
            print(f"{prefix}[OK] Saved MCTAgentV2 to {mct_path}")
            write_log(f"{prefix}Saved MCTAgentV2 to {mct_path}")
        except Exception as e:
            print(f"{prefix}[FAIL] Failed to save MCTAgentV2: {e}")
            write_log(f"{prefix}Failed to save MCTAgentV2: {e}")

    if hasattr(llm_agent, 'save'):
        llm_path = SAVE_FILES["LLMAgentV1"]
        try:
            llm_agent.save(llm_path)
            print(f"{prefix}[OK] Saved LLMAgentV1 to {llm_path}_*")
            write_log(f"{prefix}Saved LLMAgentV1 to {llm_path}_*")
        except Exception as e:
            print(f"{prefix}[FAIL] Failed to save LLMAgentV1: {e}")
            write_log(f"{prefix}Failed to save LLMAgentV1: {e}")


def run_training(n_games: int = 30, show: bool = False, use_qwen: bool = True, use_lora: bool = False,
                  use_int8: bool = False, checkpoint_every: int = 20):
    """
    訓練 LLMAgentV1

    Args:
        checkpoint_every: 每隔多少局中途存檔一次（覆蓋同一份存檔，不是每次都留新檔）。
            設 0 停用中途存檔，只在全部跑完後存一次（舊行為）。中途存檔是為了避免
            長時間訓練中途中斷（例如手動中止、單局異常耗時過久）時進度全部流失。
    """
    print(f"\n{'='*70}")
    print(f"LLMAgent V1 Training vs MCTAgentV2")
    print(f"Model: {'Qwen2.5-0.5B-Instruct' if use_qwen else 'Dummy'}")
    print(f"Games: {n_games}")
    print(f"Precision: {'INT8' if use_int8 else 'FP32'}")
    print(f"{'='*70}\n")

    # 記錄訓練開始
    write_log(f"=== Training Start ===")
    mode_str = f"LoRA: {use_lora}, INT8: {use_int8}"
    write_log(f"Games: {n_games}, Model: {'Qwen2.5-0.5B-Instruct' if use_qwen else 'Dummy'}, {mode_str}")

    llm_agent, mct_agent = create_agents(use_qwen=use_qwen, use_lora=use_lora, use_int8=use_int8, log=True)

    wins = [0, 0, 0]  # [LLM wins, MCT wins, draws]

    for i in range(n_games):
        print(f"[Game {i+1:2d}/{n_games}]", end=" ")
        sys.stdout.flush()

        try:
            result = run_game(llm_agent, mct_agent, show=show)
        except Exception as e:
            print(f"\n[ERROR] Game {i+1} failed: {e}")
            write_log(f"Game {i+1}/{n_games}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            raise

        if result == 0:
            wins[0] += 1
            status = "LLM WIN"
        elif result == 1:
            wins[1] += 1
            status = "MCT WIN"
        else:
            wins[2] += 1
            status = "DRAW"

        # 顯示統計
        stats = llm_agent.get_stats()
        output = f"{status:8s}  | Valid: {stats['valid_llm_outputs']:2d}/{stats['total_actions']:2d} " \
                 f"| Fmt-Err: {stats['format_errors']:2d}  Ill-Card: {stats['illegal_cards']:2d} " \
                 f"| Len(原→修): {stats['avg_original_length']:>5}→{stats['avg_corrected_length']:<5}"
        print(output)

        # 記錄到日誌
        write_log(f"Game {i+1}/{n_games}: {status} | Valid: {stats['valid_llm_outputs']}/{stats['total_actions']} "
                  f"| Fmt-Err: {stats['format_errors']} | Ill-Card: {stats['illegal_cards']} "
                  f"| Len(原→修): {stats['avg_original_length']}→{stats['avg_corrected_length']}")

        # 中途存檔：避免長時間訓練中途中斷時進度全部流失
        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            save_agents(llm_agent, mct_agent, tag=f"Game {i+1}/{n_games}")

    # 匯總統計
    print(f"\n{'='*70}")
    print(f"Training Results ({n_games} games):")
    print(f"  LLM Wins:  {wins[0]} ({wins[0]/n_games*100:5.1f}%)")
    print(f"  MCT Wins:  {wins[1]} ({wins[1]/n_games*100:5.1f}%)")
    print(f"  Draws:     {wins[2]} ({wins[2]/n_games*100:5.1f}%)")

    final_stats = llm_agent.get_stats()
    print(f"\nLLM Overall Stats:")
    print(f"  Total Actions:     {final_stats['total_actions']}")
    print(f"  Valid LLM Outputs: {final_stats['valid_llm_outputs']}")
    print(f"  Format Errors:     {final_stats['format_errors']}")
    print(f"  Illegal Cards:     {final_stats['illegal_cards']}")
    print(f"  Fallback Count:    {final_stats['fallback_count']}")
    print(f"  Valid Rate:        {final_stats['valid_rate']}")
    print(f"  Avg Original Len:  {final_stats['avg_original_length']} chars")
    print(f"  Avg Corrected Len: {final_stats['avg_corrected_length']} chars")

    # 記錄總結統計
    write_log(f"=== Training Complete ===")
    write_log(f"Results: LLM Wins {wins[0]} ({wins[0]/n_games*100:.1f}%), "
              f"MCT Wins {wins[1]} ({wins[1]/n_games*100:.1f}%), "
              f"Draws {wins[2]} ({wins[2]/n_games*100:.1f}%)")
    write_log(f"Stats: Total Actions {final_stats['total_actions']}, "
              f"Valid {final_stats['valid_llm_outputs']}, "
              f"Fmt-Err {final_stats['format_errors']}, "
              f"Ill-Card {final_stats['illegal_cards']}, "
              f"Valid Rate {final_stats['valid_rate']}, "
              f"AvgLen(原→修) {final_stats['avg_original_length']}→{final_stats['avg_corrected_length']}")

    # 保存最終存檔
    print()
    save_agents(llm_agent, mct_agent)

    print(f"{'='*70}\n")
    write_log(f"Log file saved to {LOG_FILE}")


def cleanup_resources() -> None:
    """清理 GPU 資源（在訓練完成後調用）"""
    print("\n[Cleanup] Releasing GPU memory...")
    try:
        import torch
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        write_log("[Cleanup] GPU memory released, process exiting normally")
        print("[Cleanup] GPU memory released")
    except Exception as e:
        write_log(f"[Cleanup] Error during cleanup: {e}")
        print(f"[Cleanup] Error during cleanup: {e}")


if __name__ == "__main__":
    import argparse

    # ✅ 模組可導入的入口點
    # 其他程序可以調用: from run_llm_v1_training import run_training
    parser = argparse.ArgumentParser(description="LLMAgent V1 Training")
    parser.add_argument("--games", type=int, default=30, help="Number of games to play")
    parser.add_argument("--show", action="store_true", help="Show detailed game output")
    parser.add_argument("--dummy", action="store_true", help="Use Dummy LLM for testing")
    parser.add_argument("--lora", action="store_true", help="Use LoRA adaptation for faster training")
    parser.add_argument("--INT8", action="store_true", help="Use INT8 quantization for memory efficiency")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                         help="Save a checkpoint every N games (0 to disable mid-training checkpoints)")

    args = parser.parse_args()

    try:
        # ✅ 現在可以被其他程序導入調用
        run_training(
            n_games=args.games,
            show=args.show,
            use_qwen=not args.dummy,
            use_lora=args.lora,
            use_int8=args.INT8,
            checkpoint_every=args.checkpoint_every
        )
    finally:
        cleanup_resources()
        print("[Info] Training process completed. Exiting...")
        sys.exit(0)
