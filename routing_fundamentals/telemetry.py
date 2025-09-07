from __future__ import annotations
import json, time

class JSONLLogger:
    def __init__(self, path: str = "router_logs.jsonl"):
        self.path = path
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record):
        rec = dict(ts=time.time(), **record)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
