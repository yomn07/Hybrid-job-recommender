"""
explain.py — Explications lisibles d'une recommandation (utilisé par la démo).

L'explication combine les trois sources de signal :
  * contenu       : compétences communes et compétences manquantes (« skill gap ») ;
  * connaissances : contraintes satisfaites / violées / inconnues ;
  * collaboratif  : candidats au profil proche ou co-positionnés sur la même offre.
Le score affiché est un score de correspondance, PAS une probabilité d'embauche.
"""
from __future__ import annotations

import numpy as np

from .engines import KB_CONSTRAINTS


def explain(ctx, comp: dict, pool: np.ndarray, idx: int, u: int, t: int) -> dict:
    p = ctx.prof
    j = int(pool[idx])
    c_sk = set(p.tokens("c", "sk", u))
    j_sk = set(p.tokens("j", "sk", j))
    matched = sorted(c_sk & j_sk)
    missing = sorted(j_sk - c_sk)
    row = comp["kb_matrix"][idx]
    ok = [KB_CONSTRAINTS[i] for i, v in enumerate(row) if v == 1]
    ko = [KB_CONSTRAINTS[i] for i, v in enumerate(row) if v == 0]
    unk = [KB_CONSTRAINTS[i] for i, v in enumerate(row) if v == 0.5]
    n_app = len(ctx.hist.users_of(j, t))

    reasons = []
    if matched:
        reasons.append(f"{len(matched)}/{len(j_sk)} compétences demandées présentes dans votre profil "
                       f"({', '.join(matched[:5])}{'…' if len(matched) > 5 else ''}).")
    if ok:
        reasons.append("Critères respectés : " + ", ".join(ok) + ".")
    if comp["meta"][idx] > 0:
        reasons.append("Des candidats au profil proche du vôtre ont été positionnés sur cette offre.")
    if comp["cf"][idx] > 0:
        reasons.append("Des candidats positionnés sur les mêmes offres que vous l'ont aussi été sur celle-ci.")
    if comp["pop"][idx] > 0:
        reasons.append(f"Offre active : {int(comp['pop'][idx])} candidature(s) ces {ctx.pop_window} derniers jours.")
    warnings = []
    if ko:
        warnings.append("Critères non respectés : " + ", ".join(ko) + ".")
    if missing:
        warnings.append("Compétences à acquérir : " + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "") + ".")
    if unk:
        warnings.append("Information manquante pour : " + ", ".join(unk) + ".")
    return {"job": j, "matched": matched, "missing": missing, "ok": ok, "ko": ko,
            "unknown": unk, "n_applicants_before": n_app, "reasons": reasons, "warnings": warnings}
