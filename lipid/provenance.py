"""
lipid/provenance.py — environment, source and output checksums for manifests (REP-03).
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

import lipid

ROOT = lipid.ROOT


def _ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return None


def environment() -> dict:
    env = dict(python=sys.version.replace("\n", " "), executable=sys.executable, platform=platform.platform(),
               machine=platform.machine(), processor=platform.processor(), hostname=socket.gethostname(),
               cpu_count=os.cpu_count())
    for mod in ("numpy", "scipy", "torch", "pandas", "pyarrow", "matplotlib", "sklearn"):
        env[mod] = _ver(mod)
    try:
        import torch
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_version"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        env["cuda_devices"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                               if torch.cuda.is_available() else [])
        env["torch_default_dtype"] = str(torch.get_default_dtype())
        env["torch_num_threads"] = torch.get_num_threads()
    except Exception:  # pragma: no cover
        pass
    return env


def git_info(root: str = ROOT) -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    head = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return dict(commit=head, branch=run("git", "rev-parse", "--abbrev-ref", "HEAD"),
                dirty=(None if status is None else bool(status)),
                dirty_files=(None if not status else status.splitlines()[:200]))


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_checksums(root: str = ROOT) -> dict:
    files = sorted(glob.glob(os.path.join(root, "*.py")) + glob.glob(os.path.join(root, "lipid", "*.py"))
                   + glob.glob(os.path.join(root, "scripts", "*.py")) + glob.glob(os.path.join(root, "scripts", "*.sh"))
                   + glob.glob(os.path.join(root, "tests", "*.py")))
    return {os.path.relpath(f, root): sha256(f) for f in files}


def output_checksums(paths) -> dict:
    return {os.path.relpath(p, ROOT) if os.path.isabs(p) else p: dict(sha256=sha256(p), bytes=os.path.getsize(p))
            for p in sorted(paths) if os.path.isfile(p)}


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def base_manifest(argv=None, **extra) -> dict:
    return dict(protocol=lipid.PROTOCOL, created=now(), argv=list(sys.argv if argv is None else argv),
                cwd=os.getcwd(), git=git_info(), environment=environment(), source_sha256=source_checksums(), **extra)


def to_jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (o != o):
        return None
    return o


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(to_jsonable(obj), fh, indent=1, sort_keys=False)


def _yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    return json.dumps(s)


def to_yaml(obj, indent: int = 0) -> str:
    """Minimal YAML emitter (dicts, lists, scalars) so that no PyYAML dependency is needed."""
    pad = "  " * indent
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)) and len(v):
                lines.append(f"{pad}{k}:")
                lines.append(to_yaml(v, indent + 1))
            elif isinstance(v, (list, tuple)):
                lines.append(f"{pad}{k}: []")
            elif isinstance(v, dict):
                lines.append(f"{pad}{k}: {{}}")
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
    elif isinstance(obj, (list, tuple)):
        if all(not isinstance(v, (dict, list, tuple)) for v in obj):
            return pad + "[" + ", ".join(_yaml_scalar(v) for v in obj) + "]"
        for v in obj:
            lines.append(f"{pad}-")
            lines.append(to_yaml(v, indent + 1))
    else:
        lines.append(pad + _yaml_scalar(obj))
    return "\n".join(lines)


def write_yaml(path: str, obj) -> None:
    with open(path, "w") as fh:
        fh.write(to_yaml(to_jsonable(obj)) + "\n")
