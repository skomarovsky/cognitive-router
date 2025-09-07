from __future__ import annotations
import yaml, argparse, numpy as np
import sys, pathlib

# --- Make this script runnable from any cwd ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from routing_fundamentals.qwen_loader import build_embedder
except Exception as e:
    raise RuntimeError(
        "Failed to import routing_fundamentals. "
        "Run from the project root with: 'poetry run python tools/build_centroids.py ...' "
        "or use module mode: 'poetry run python -m tools.build_centroids ...'"
    ) from e

def l2norm(v):
    import numpy as np
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--examples", default="examples/routes_examples.yaml")
    ap.add_argument("--routes", default="routes.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ex = yaml.safe_load(open(args.examples))
    routes = yaml.safe_load(open(args.routes))

    embed_cfg = cfg["embedding"]
    embed_fn = build_embedder(embed_cfg["model"], normalize=embed_cfg.get("normalize", True),
                              batch_size=embed_cfg.get("batch_size", 64))

    for r in routes["routes"]:
        positives = ex.get(r["name"], {}).get("positives", [])
        if not positives:
            print(f"[warn] no positives for route {r['name']}")
            continue
        vecs = embed_fn(positives)
        centroid = l2norm(vecs.mean(axis=0))
        r["centroid"] = centroid.tolist()
        print(f"[ok] {r['name']}: N={len(positives)}")

    yaml.safe_dump(routes, open(args.routes, "w"), sort_keys=False)
    print(f"[done] wrote centroids into {args.routes}")

if __name__ == "__main__":
    main()
