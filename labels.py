import csv, json, sys
import numpy as np
T, F, S = 0.8, 0.5, 0.2          # same thresholds as config.json

def U(rows, lev):
    rho, sg = [], []
    for r in rows:
        e = r["levels"].get(lev)
        if not e:
            continue
        for j in r["active"]:
            rho.append(e["rho"][j])
            sg.append(e["sign"][j] != 0)
    if not rho:
        return None
    fr, sr = float(np.mean(np.array(rho) <= T)), float(np.mean(sg))
    return fr, sr, fr >= F and sr >= S

w = csv.writer(sys.stdout)
w.writerow(["run", "d", "model", "mode", "Lmin_frac", "Lmin_sign", "LV_frac", "LV_sign",
            "Ltrue_frac", "Ltrue_sign", "label_hybrid", "label_prior", "prior_implies_hybrid"])
for path in sys.argv[1:]:
    rows = [r for r in map(json.loads, open(path)) if not r.get("degenerate")]
    keys = sorted({(r["d"], r["model"], r["mode"]) for r in rows}) \
        + sorted({(r["d"], r["model"], "all") for r in rows}) \
        + sorted({(r["d"], "all", "all") for r in rows})
    for d, mo, md in keys:
        g = [r for r in rows if r["d"] == d and mo in ("all", r["model"]) and md in ("all", r["mode"])]
        lm, lv = U(g, "Lmin"), U(g, "LV")
        lt = U(g, "Ltrue") if all(r["levels"].get("Ltrue") for r in g) else None
        hyb = "S3" if lv[2] else ("S2" if lm[2] else "S1")
        pri = "-" if lt is None else ("S3" if lt[2] else "no")
        ok = "-" if lt is None else str((not lt[2]) or lv[2])
        f = lambda t, k: "" if t is None else f"{t[k]:.3f}"
        w.writerow([path, d, mo, md, f(lm, 0), f(lm, 1), f(lv, 0), f(lv, 1), f(lt, 0), f(lt, 1), hyb, pri, ok])