"""
lipid/io.py — table I/O. Parquet (pyarrow, zstd) when available, otherwise gzip CSV (same columns).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

STR_COLS = {"model", "mode", "anchor_law", "arm", "chain_id", "Dbin", "L_kind", "protocol", "method", "metric",
            "family", "level", "policy", "stratum"}


def have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def to_frame(cols: dict) -> pd.DataFrame:
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c in STR_COLS or df[c].dtype == object or str(df[c].dtype).startswith("<U"):
            df[c] = df[c].astype(str).astype("category")
    return df


def write_table(df: pd.DataFrame, base: str) -> str:
    """Writes <base>.parquet (or <base>.csv.gz). Returns the path written."""
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    if have_parquet():
        path = base + ".parquet"
        df.to_parquet(path, index=False, compression="zstd")
    else:
        path = base + ".csv.gz"
        df.to_csv(path, index=False, compression="gzip", float_format="%.17g")
    return path


def _path(base: str) -> str | None:
    for ext in (".parquet", ".csv.gz"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def read_table(base_or_path: str, columns=None, filters=None) -> pd.DataFrame:
    p = base_or_path if os.path.exists(base_or_path) else _path(base_or_path)
    if p is None:
        raise FileNotFoundError(base_or_path)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p, columns=columns, filters=filters)
    else:
        df = pd.read_csv(p, usecols=columns)
        if filters:
            for col, op, val in filters:
                if op == "==":
                    df = df[df[col] == val]
                elif op == "in":
                    df = df[df[col].isin(val)]
                else:
                    raise ValueError(op)
    for c in df.columns:
        if c in STR_COLS and str(df[c].dtype) != "category":
            df[c] = df[c].astype(str).astype("category")
    return df


def seeds_in(run_dir: str, kind: str = "records") -> list[int]:
    out = set()
    for p in glob.glob(os.path.join(run_dir, f"{kind}_seed*")):
        m = re.search(rf"{kind}_seed(\d+)\.", os.path.basename(p))
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def read_run(run_dir: str, kind: str = "records", columns=None, filters=None, seeds=None) -> pd.DataFrame:
    seeds = seeds_in(run_dir, kind) if seeds is None else seeds
    if not seeds:
        raise FileNotFoundError(f"no {kind}_seed* tables in {run_dir}")
    parts = [read_table(os.path.join(run_dir, f"{kind}_seed{s}"), columns=columns, filters=filters) for s in seeds]
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:
        if c in STR_COLS and str(df[c].dtype) != "category":
            df[c] = df[c].astype(str).astype("category")
    return df


def diag_columns(df: pd.DataFrame) -> dict:
    """pandas -> dict of numpy arrays in the form nested_anchors_v2.table() expects."""
    out = {}
    for c in df.columns:
        v = df[c].to_numpy()
        if c in STR_COLS:
            v = v.astype(str)
        out[c] = v
    return out


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def fmt_range(v, nd=2) -> str:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "--"
    lo, hi = np.min(v), np.max(v)
    f = lambda t: (f"{t:.{nd}f}"[1:] if 0 <= t < 1 and nd > 0 else
                   (f"-{f'{-t:.{nd}f}'[1:]}" if -1 < t < 0 else f"{t:.{nd}f}"))
    return f"{f(lo)}--{f(hi)}" if f(lo) != f(hi) else f(lo)
