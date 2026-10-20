"""
lipid/lock.py — protocol locks for the v1.2 confirmatory batch (STAT-01) and other pre-specified experiments.

A lock records, BEFORE any confirmatory output exists:
  * the sha256 of the protocol file (protocols/v12_preregistration.json) and of every analysis / runner source file;
  * the git commit and dirty flag, UTC and local timestamps, the seeds and their RNG stream keys;
  * a uniqueness check: the new seeds' stream keys (crc32 of the key tuples used by phase1.gen_of) must not collide
    with each other nor with the keys of the historical seeds 1-11.
`verify_lock` re-hashes the same files: any change is reported (never silently accepted). The analysis scripts record
the verification result next to their outputs, so a post-lock change appears in the report as a deviation.
This is a time-stamped self-registration, not a third-party registration; the paper must say so.
"""
from __future__ import annotations

import glob
import json
import os

import lipid
from lipid.provenance import git_info, now, sha256

ROOT = lipid.ROOT
HISTORICAL_SEEDS = tuple(range(1, 12))
KEY_PREFIXES = ("nested2", "nested2-C", "nested2-P", "nested2-M")


def default_code_files() -> list[str]:
    pats = ["lipid/*.py", "scripts/*.py", "scripts/*.sh", "nested_anchors_v2.py", "phase1.py", "envelope.py",
            "bounds.py", "t6_directional.py"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(ROOT, p))
    return sorted(set(os.path.relpath(f, ROOT) for f in out))


def stream_keys(seeds, dims=(3, 4, 5, 6), rhos=(0.0, 0.6, 0.9, 0.99)) -> dict:
    import phase1 as P
    return {f"{pre}|{s}|{d}|{r}": P.seed_of(pre, s, d, r) for pre in KEY_PREFIXES for s in seeds for d in dims
            for r in rhos}


def key_collisions(new_seeds, dims=(3, 4, 5, 6), rhos=(0.0, 0.6, 0.9, 0.99)) -> list:
    keys = stream_keys(tuple(new_seeds) + HISTORICAL_SEEDS, dims, rhos)
    inv = {}
    for k, v in keys.items():
        inv.setdefault(v, []).append(k)
    return [ks for ks in inv.values() if len(ks) > 1]


def locked_files_git_state(files) -> dict:
    """Only the LOCKED files matter: each must be tracked by git and identical to HEAD. Other dirty paths (untracked
    outputs, unrelated edits) are recorded for information but do not block the lock."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    untracked = [f for f in files if git("ls-files", "--error-unmatch", f).returncode != 0]
    st = git("status", "--porcelain", "--", *files)
    modified = [ln[3:] for ln in st.stdout.splitlines() if ln and not ln.startswith("??")] if st.returncode == 0 else ["<git error>"]
    other = [ln for ln in git("status", "--porcelain").stdout.splitlines()]
    return dict(untracked=untracked, modified=modified, n_other_dirty_paths=len(other), other_dirty_paths=other[:200])


def write_lock(protocol_path: str, out_path: str, *, experiment: str, seeds=None, code_files=None,
               allow_dirty: bool = False, note: str = "") -> dict:
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists: a lock is never overwritten (write a new versioned lock instead)")
    g = git_info()
    code_files = code_files or default_code_files()
    prot_rel = os.path.relpath(protocol_path, ROOT)
    state = locked_files_git_state(sorted(set(code_files) | {prot_rel}))
    if (state["untracked"] or state["modified"] or g.get("commit") is None) and not allow_dirty:
        raise RuntimeError("locked files must be committed and unmodified at HEAD "
                           f"(untracked: {state['untracked']}, modified: {state['modified']}, commit: {g.get('commit')}). "
                           "Untracked outputs (runs/, results_v12/, logs/) do not matter. "
                           "--allow-dirty only if the paper says so.")
    lock = dict(experiment=experiment, created_local=now(), protocol=prot_rel,
                protocol_sha256=sha256(protocol_path), git=g, locked_files_git_state=state,
                seeds=list(seeds or []), note=note,
                code_sha256={f: sha256(os.path.join(ROOT, f)) for f in code_files if os.path.isfile(os.path.join(ROOT, f))})
    if seeds:
        col = key_collisions(seeds)
        lock["rng_key_collisions"] = col
        if col:
            raise RuntimeError(f"RNG stream-key collisions: {col}")
        overlap = sorted(set(seeds) & set(HISTORICAL_SEEDS))
        if overlap:
            raise RuntimeError(f"seeds overlap the historical seeds 1-11: {overlap}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(lock, fh, indent=1)
    return lock


def verify_lock(lock_path: str) -> dict:
    with open(lock_path) as fh:
        lock = json.load(fh)
    changed, missing = [], []
    p = os.path.join(ROOT, lock["protocol"])
    if not os.path.isfile(p):
        missing.append(lock["protocol"])
    elif sha256(p) != lock["protocol_sha256"]:
        changed.append(lock["protocol"])
    for f, h in lock["code_sha256"].items():
        fp = os.path.join(ROOT, f)
        if not os.path.isfile(fp):
            missing.append(f)
        elif sha256(fp) != h:
            changed.append(f)
    return dict(ok=not changed and not missing, lock=os.path.relpath(lock_path, ROOT), experiment=lock["experiment"],
                created_local=lock["created_local"], changed=changed, missing=missing, checked_at=now(),
                git_at_lock=lock["git"].get("commit"))


__all__ = ["write_lock", "verify_lock", "key_collisions", "stream_keys", "default_code_files"]
