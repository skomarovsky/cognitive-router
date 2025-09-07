from __future__ import annotations
import argparse, yaml
from .qwen_loader import build_embedder, load_generator
from .router import load_routes, rank_routes, pick_route, route_score_components

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--routes", default="routes.yaml")
    ap.add_argument("--deterministic", action="store_true", help="Disable sampling when scores are close")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rank_p = sub.add_parser("rank", help="Rank routes for a query")
    rank_p.add_argument("query", type=str)
    rank_p.add_argument("--explain", action="store_true", help="Show score components per route")

    pick_p = sub.add_parser("pick", help="Pick a winner route for a query")
    pick_p.add_argument("query", type=str)

    chat_p = sub.add_parser("chat", help="Route and (if generative) call Qwen")
    chat_p.add_argument("query", type=str)

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    routes = load_routes(args.routes)
    emb_cfg = cfg['embedding']
    embed_fn = build_embedder(emb_cfg['model'], normalize=emb_cfg.get('normalize', True))
    W = cfg['router']['weights']
    TOP_K = cfg['router']['top_k']
    TEMP  = cfg['router']['temperature']

    query = getattr(args, "query", None)

    if args.cmd == "rank":
        scored = rank_routes(query, routes, embed_fn, W, top_k=TOP_K)
        for i,(r,c) in enumerate(scored,1):
            print(f"{i}. {r.name} -> score={c['score']:.3f}")
        if args.explain:
            q_vec = embed_fn([query])[0]
            print("\n[explain] components:")
            for r,_ in scored:
                comps = route_score_components(q_vec, query, r, W)
                print(f"- {r.name}: "
                      f"sim={comps['sim']:.3f}, kw={comps['kw']:.2f}, prior={comps['prior']:.2f}, "
                      f"cost={comps['cost']:.2f}, lat={comps['latency']:.2f}, health={comps['health']}, "
                      f"bias={comps['bias']:.2f} -> score={comps['score']:.3f}")
        return

    if args.cmd == "pick":
        scored = rank_routes(query, routes, embed_fn, W, top_k=TOP_K)
        winner = pick_route(scored, tau=TEMP, sample_when_close=not args.deterministic)
        print(winner.name)
        return

    if args.cmd == "chat":
        scored = rank_routes(query, routes, embed_fn, W, top_k=TOP_K)
        winner = pick_route(scored, tau=TEMP, sample_when_close=not args.deterministic)
        gen = load_generator(cfg['generator'])
        gen_kwargs = dict(max_new_tokens=cfg['generator']['max_new_tokens'],
                          temperature=cfg['generator']['temperature'],
                          top_p=cfg['generator']['top_p'],
                          top_k=cfg['generator']['top_k'],
                          repetition_penalty=cfg['generator']['repetition_penalty'])
        if winner.name == "code.gen":
            out = gen.generate(query, gen_kwargs)
            print(out)
        else:
            print(f"{winner.name}: (demo) retrieved 3 passages with citations.")

if __name__ == "__main__":
    main()
