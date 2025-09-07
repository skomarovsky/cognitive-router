from __future__ import annotations
from typing import Dict, List, Tuple, Any, Callable
import numpy as np

Candidate = Tuple[str, bool, Any, dict]

def take_first(cands: List[Candidate]) -> Candidate | None:
    for c in cands:
        if c[1]:
            return c
    return cands[0] if cands else None

def rrf(rankings: Dict[str, Dict[str,int]], k: int = 60) -> List[tuple[str, float]]:
    scores: Dict[str,float] = {}
    for system, ranks in rankings.items():
        for ans_id, r in ranks.items():
            scores[ans_id] = scores.get(ans_id, 0.0) + 1.0 / (k + r)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def semantic_agreement(texts: Dict[str, str], embed_fn: Callable[[List[str]], np.ndarray]) -> List[tuple[str,float]]:
    keys = list(texts.keys())
    if not keys:
        return []
    vecs = embed_fn([texts[k] for k in keys])
    sims = vecs @ vecs.T  # cosine if normalized
    centrality = sims.mean(axis=1)
    return sorted(zip(keys, centrality), key=lambda x: x[1], reverse=True)
