"""
V5 DQN 網路定義

HandEncoder
  - 每張牌用 3 個 5-level 值編碼（5^3=125 種組合），共 8 個槽
  - 8 槽共用相同的 Linear 層：PyTorch Linear 廣播到最後一維，天然實現 per-slot 共用參數
  - 輸入 (..., HAND_SLOTS, CARD_DIMS) → 輸出 (..., HAND_SLOTS, CARD_EMBED)

DQNNet
  - HandEncoder 壓縮 8 槽 → 展平後接其他遊戲特徵，進入主 MLP
  - 輸出 ACTION_SIZE=256 個 Q 值（all_subsets 的最大動作數）
  - 預設編碼器為最小可還原形狀：hidden=(5,5), embed=6
"""
from __future__ import annotations
import torch
import torch.nn as nn

HAND_SLOTS  = 8
CARD_DIMS   = 3    # 5-level 3-bit encoding per card
CARD_EMBED  = 6    # per-card embedding output dim（最小可還原形狀 5x2）
OTHER_FEATS = 16   # 非手牌特徵：hp×2×2, def×2×2, hand_sz×1, 7-numbers×7
ACTION_SIZE = 256  # 2^8，8 張手牌的最大唯一子集數

ENCODER_HIDDEN  = (5, 5)   # 最小可還原編碼器隱藏層
ENCODER_PRETRAIN_PATH = "hand_encoder_pretrained.pt"

_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def encode_card(v: int) -> list[float]:
    """card value v → 3 元素 5-level 編碼
    v=0  : 空位 padding → (-1,-1,-1)
    v=1..124 : 實際牌值
    v>124 : 截斷到 124
    """
    idx = max(0, min(124, v))
    return [_LEVELS[idx // 25], _LEVELS[(idx // 5) % 5], _LEVELS[idx % 5]]


class HandEncoder(nn.Module):
    """單張牌嵌入 MLP；8 個槽共用相同參數。

    PyTorch nn.Linear 對輸入的最後一維運算，
    故直接對 (..., HAND_SLOTS, CARD_DIMS) 調用即可實現 per-slot 共用。

    Parameters
    ----------
    in_dim       : 輸入維度，固定 CARD_DIMS=3
    hidden_sizes : 中間隱藏層尺寸序列，例如 (5,5)
    embed_dim    : 輸出嵌入維度
    use_norm     : 是否在每個隱藏層後加 LayerNorm
    """

    def __init__(
        self,
        in_dim      : int        = CARD_DIMS,
        hidden_sizes: tuple[int] = ENCODER_HIDDEN,
        embed_dim   : int        = CARD_EMBED,
        use_norm    : bool       = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if use_norm:
                layers.append(nn.LayerNorm(h))
            prev = h
        layers += [nn.Linear(prev, embed_dim), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., HAND_SLOTS, CARD_DIMS)  →  (..., HAND_SLOTS, embed_dim)
        return self.net(x)


class DQNNet(nn.Module):
    """主 Q 網路。

    Parameters
    ----------
    hand_slots           : 手牌槽數（固定 8）
    encoder_hidden_sizes : HandEncoder 隱藏層（預設最小可還原形狀 (5,5)）
    card_embed           : HandEncoder 輸出維度（預設 6）
    other_dim            : 非手牌特徵維度
    action_size          : Q 值輸出數（= ACTION_SIZE）
    """

    def __init__(
        self,
        hand_slots          : int        = HAND_SLOTS,
        encoder_hidden_sizes: tuple[int] = ENCODER_HIDDEN,
        card_embed          : int        = CARD_EMBED,
        other_dim           : int        = OTHER_FEATS,
        action_size         : int        = ACTION_SIZE,
    ):
        super().__init__()
        self.hand_encoder = HandEncoder(CARD_DIMS, encoder_hidden_sizes, card_embed)
        in_dim = hand_slots * card_embed + other_dim   # 8×6 + 16 = 64
        self.main = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(
        self,
        hand_feats : torch.Tensor,   # (..., HAND_SLOTS, CARD_DIMS)
        other_feats: torch.Tensor,   # (..., OTHER_FEATS)
    ) -> torch.Tensor:               # (..., ACTION_SIZE)
        embedded = self.hand_encoder(hand_feats)          # (..., HAND_SLOTS, CARD_EMBED)
        flat     = embedded.flatten(start_dim=-2)         # (..., HAND_SLOTS * CARD_EMBED)
        combined = torch.cat([flat, other_feats], dim=-1) # (..., 64)
        return self.main(combined)
