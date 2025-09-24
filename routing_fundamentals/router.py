from __future__ import annotations
import re, argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Union
import numpy as np
import yaml

@dataclass
class Route:
    name: str
    centroid: np.ndarray
    kw_patterns: List[str]
    prior: float
    bias: float
    cost_norm: float
    latency_norm: float
    health: int
    dispatch: dict | None = None
    aggregation: dict | None = None

def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def kw_bonus(q: str, patterns: List[str]) -> float:
    hits = sum(1 for p in patterns if re.search(p, q, flags=re.I))
    return min(1.0, 0.1*hits)

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_routes(path: str) -> List[Route]:
    raw = load_yaml(path)
    routes = []
    for r in raw['routes']:
        c = np.array(r.get('centroid', []), dtype=np.float32)
        if c.size:
            c = l2norm(c)
        routes.append(Route(
            name=r['name'],
            centroid=c,
            kw_patterns=r.get('kw_patterns', []),
            prior=float(r.get('prior', 0.0)),
            bias=float(r.get('bias', 0.0)),
            cost_norm=float(r.get('cost_norm', 0.0)),
            latency_norm=float(r.get('latency_norm', 0.0)),
            health=int(r.get('health', 1)),
            dispatch=r.get('dispatch', None),
            aggregation=r.get('aggregation', None),
        ))
    return routes

def _align_for_similarity(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter vector with zeros so we can take a dot product.

    Real routes may ship centroids that were created with a slightly different
    embedding dimensionality.  When that happens numpy's ``@`` would raise a
    ``ValueError`` even though the extra dimensions are effectively unused.  To
    keep routing working we zero-pad whichever vector is shorter so that both
    are comparable without mutating the originals.
    """

    if a.shape == b.shape:
        return a, b

    target = max(a.shape[-1], b.shape[-1])

    def _pad(vec: np.ndarray) -> np.ndarray:
        if vec.shape[-1] == target:
            return vec
        pad_width = [(0, 0)] * (vec.ndim - 1) + [(0, target - vec.shape[-1])]
        return np.pad(vec, pad_width, mode="constant")

    return _pad(a), _pad(b)


def route_score_components(q_vec: np.ndarray, q_text: str, route: Route, W: Dict[str, float]) -> Dict[str, float]:
    if route.centroid.size:
        q_vec_aligned, centroid = _align_for_similarity(q_vec, route.centroid)
        sim = float(q_vec_aligned @ centroid)
    else:
        sim = 0.0
    kw  = kw_bonus(q_text, route.kw_patterns)
    prior = route.prior
    cost  = route.cost_norm
    lat   = route.latency_norm
    health= route.health
    score = (W['sim']*sim + W['kw']*kw + W['prior']*prior
             - W['cost']*cost - W['lat']*lat + W['health']*health + route.bias)
    return dict(score=score, sim=sim, kw=kw, prior=prior, cost=cost, latency=lat, health=health, bias=route.bias)

def softmax(xs: Union[List[float], np.ndarray], tau: float) -> np.ndarray:
    xs = np.array(xs) / max(tau, 1e-6)
    xs = xs - xs.max()
    e = np.exp(xs)
    return e / (e.sum() + 1e-12)

def rank_routes(query: str, routes: List[Route], embed_fn: Callable[[List[str]], np.ndarray], W: Dict[str,float], top_k:int=3) -> List[Tuple[Route, Dict[str,float]]]:
    q_vec = embed_fn([query])[0]
    scored = []
    for r in routes:
        comps = route_score_components(q_vec, query, r, W)
        scored.append((r, comps))
    scored.sort(key=lambda x: x[1]['score'], reverse=True)
    return scored[:top_k]

def pick_route(scored_topk: List[Tuple[Route, Dict[str,float]]], tau: float=0.35, margin_threshold: float=0.06, sample_when_close: bool=True) -> Route:
    if not scored_topk:
        raise ValueError("No routes scored")
    if len(scored_topk) == 1 or not sample_when_close:
        return scored_topk[0][0]
    m = scored_topk[0][1]['score'] - scored_topk[1][1]['score']
    if m >= margin_threshold:
        return scored_topk[0][0]
    probs = softmax([c['score'] for _, c in scored_topk], tau)
    idx = np.random.choice(range(len(scored_topk)), p=probs)
    return scored_topk[idx][0]
