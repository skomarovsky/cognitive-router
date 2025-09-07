from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Any, Tuple, List

Runner = Callable[[str, dict], Tuple[bool, Any, dict]]

def single_winner(query: str, winner_name: str, runners: Dict[str, Runner], gen_kwargs: dict):
    res = []
    ok=False; out=None; meta={}
    if winner_name in runners:
        ok, out, meta = runners[winner_name](query, gen_kwargs)
        res.append((winner_name, ok, out, meta))
    return res, winner_name

def cascade_fallback(query: str, ranked_names: List[str], runners: Dict[str, Runner], gen_kwargs: dict):
    res = []
    winner = None
    for name in ranked_names:
        if name not in runners:
            continue
        ok, out, meta = runners[name](query, gen_kwargs)
        res.append((name, ok, out, meta))
        if ok:
            winner = name
            break
    return res, (winner or (ranked_names[0] if ranked_names else ""))

def parallel_k_of_n(query: str, names: List[str], runners: Dict[str, Runner], gen_kwargs: dict, max_workers: int = 4):
    res = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for name in names:
            if name not in runners:
                continue
            futs[ex.submit(runners[name], query, gen_kwargs)] = name
        for fut in as_completed(futs):
            name = futs[fut]
            ok, out, meta = fut.result()
            res.append((name, ok, out, meta))
    return res, ""
