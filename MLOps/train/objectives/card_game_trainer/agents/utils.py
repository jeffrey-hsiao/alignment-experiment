class _SubsetCache:
    # Level 1: 位置索引的 power set，按手牌數 n 索引，最多 9 個 entry (n=0..8)
    # _pos[n] = list of index-tuples，e.g. _pos[2] = [(), (0,), (1,), (0,1)]
    _pos: list[list[tuple]] = [[()]]

    # Level 2: 實際手牌 sorted-tuple → 去重後的牌值 subset list
    _hand: dict[tuple, list[tuple]] = {}

    @classmethod
    def _ensure_pos(cls, n: int) -> list[tuple]:
        """確保 _pos[n] 已建好；用 n-1 的結果推出 n 的結果。"""
        while len(cls._pos) <= n:
            prev = cls._pos[-1]
            k = len(cls._pos) - 1  # 新加入的位置索引
            cls._pos.append(prev + [s + (k,) for s in prev])
        return cls._pos[n]

    @classmethod
    def get(cls, hand: list[int]) -> list[tuple]:
        key = tuple(sorted(hand))
        if key in cls._hand:
            return cls._hand[key]

        pos_subs = cls._ensure_pos(len(key))

        # 把位置索引映射到實際牌值；key 已排序，索引遞增 → mapped 天然有序
        # 重複牌值造成的重複 subset 用 seen 去除
        seen: set[tuple] = set()
        result: list[tuple] = []
        for s in pos_subs:
            mapped = tuple(key[i] for i in s)
            if mapped not in seen:
                seen.add(mapped)
                result.append(mapped)

        cls._hand[key] = result
        return result


def all_subsets(hand: list[int]) -> list[tuple[int, ...]]:
    return _SubsetCache.get(hand)


def regret_match(regrets: dict, actions: list) -> dict:
    positive = {a: max(0.0, regrets.get(a, 0.0)) for a in actions}
    total = sum(positive.values())
    if total > 0:
        return {a: positive[a] / total for a in actions}
    n = len(actions)
    return {a: 1.0 / n for a in actions}


def sample_from(strategy: dict):
    r, cumulative = __import__('random').random(), 0.0
    for a, p in strategy.items():
        cumulative += p
        if r <= cumulative:
            return a
    return list(strategy.keys())[-1]
