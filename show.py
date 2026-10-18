import json, sys
S = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "phase1_out/summary.json"))
O = S["optimistic"]

def row(name, g):
    out = [name]
    for lev in ("Lmin", "LV", "Ltrue"):
        e = g.get(lev)
        if e:
            out.append(f"{lev}: med={e['rho_quantiles_10_25_50_75_90'][2]:.3f} "
                       f"le0.8={e.get('rho_frac_le_target', float('nan')):.2f} "
                       f"sign={e['cert_sign_rate']:.3f} topk={(e['cert_topk_rate'] or 0):.3f} "
                       f"out={e['frozen_outside_rate']:.3f}")
    out.append(f"rho_inf med={g['rho_inf']['quantiles_10_25_50_75_90'][2]:.3f}")
    out.append(f"kV med={g['kappa_V_quantiles_10_25_50_75_90'][2]:.2f}")
    print(" | ".join(out))

for sec in ("by_d", "by_d_mode", "by_d_model"):
    print(f"== {sec}")
    for k, g in O[sec].items():
        if g.get("n_instances"):
            row(k, g)