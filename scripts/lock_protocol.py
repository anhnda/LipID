#!/usr/bin/env python3
"""
lock_protocol.py — write or verify a protocol lock (STAT-01 and other pre-specified v1.2 experiments).

    # after editing protocols/v12_preregistration.json and committing everything:
    python scripts/lock_protocol.py lock   --experiment STAT-01 --out protocols/locks/STAT-01.lock.json
    python scripts/lock_protocol.py verify --lock protocols/locks/STAT-01.lock.json     # exit code 1 if changed

The lock hashes the protocol and the files listed under experiments.<id>.lock_files (default: all code), records the
git commit, timestamps and the seeds' RNG stream keys, and refuses to run on a dirty tree or to overwrite a lock.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lipid.lock import verify_lock, write_lock  # noqa: E402

DEFAULT_PROTOCOL = os.path.join(ROOT, "protocols", "v12_preregistration.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lock")
    a.add_argument("--experiment", required=True)
    a.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    a.add_argument("--out", required=True)
    a.add_argument("--allow-dirty", action="store_true")
    a.add_argument("--note", default="")
    v = sub.add_parser("verify")
    v.add_argument("--lock", required=True)
    args = ap.parse_args()
    if args.cmd == "lock":
        with open(args.protocol) as fh:
            proto = json.load(fh)
        exp = proto["experiments"][args.experiment]
        files = exp.get("lock_files")
        if files:
            missing = [f for f in files if not os.path.isfile(os.path.join(ROOT, f))]
            if missing:
                raise SystemExit(f"lock_files missing: {missing}")
        lock = write_lock(args.protocol, args.out, experiment=args.experiment, seeds=exp.get("seeds"),
                          code_files=files, allow_dirty=args.allow_dirty, note=args.note)
        print(json.dumps({k: lock[k] for k in ("experiment", "created_local", "protocol_sha256", "seeds")}, indent=1))
        print(f"git commit at lock: {lock['git'].get('commit')}  dirty={lock['git'].get('dirty')}")
        print(f"wrote {args.out}")
    else:
        res = verify_lock(args.lock)
        print(json.dumps(res, indent=1))
        sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
