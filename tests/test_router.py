import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from routing_fundamentals.router import Route, route_score_components


def _make_route(dim: int) -> Route:
    centroid = np.ones(dim, dtype=np.float32)
    centroid /= np.linalg.norm(centroid)
    return Route(
        name="test",
        centroid=centroid,
        kw_patterns=[],
        prior=0.0,
        bias=0.0,
        cost_norm=0.0,
        latency_norm=0.0,
        health=1,
        dispatch=None,
        aggregation=None,
    )


def test_route_score_handles_dimension_mismatch() -> None:
    # Query embedding is shorter than the centroid exported in routes.yaml.
    q_vec = np.ones(25, dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)
    route = _make_route(27)

    weights = {"sim": 1.0, "kw": 0.0, "prior": 0.0, "cost": 0.0, "lat": 0.0, "health": 0.0}

    comps = route_score_components(q_vec, "hello", route, weights)

    # We expect the dot product to be taken after zero-padding the shorter vector.
    expected = float(np.pad(q_vec, (0, 2)) @ route.centroid)
    assert comps["sim"] == expected
    assert comps["score"] == expected
