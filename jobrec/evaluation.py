"""
evaluation.py — Protocole d'évaluation hors-ligne.

Pour chaque requête (candidat u, date t, offres réellement obtenues) :
  1. l'espace de recherche = offres actives à la date t, moins les offres sur
     lesquelles u a déjà été positionné ;
  2. chaque méthode classe toutes ces offres ;
  3. on mesure la position des offres vraies dans le top-K.

Métriques de précision : Precision@K, Recall@K, NDCG@K, HR@K, MRR.
Métriques au-delà de la précision : couverture du catalogue, diversité
intra-liste, nouveauté. Groupes : tous / nouveaux candidats (aucun historique
avant t) / candidats connus ; et HR@K sur les offres nouvelles (aucune
interaction avant t), c.-à-d. le cold-start côté offre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hybrids import components

TIE_SEED = 12345


def make_pool_fn(ctx, pool):
    """Offres actives à t, privées de celles déjà vues par u."""
    def pool_fn(u, t):
        p = pool.at(int(t))
        past = ctx.hist.items_of(int(u), int(t))
        return np.setdiff1d(p, past, assume_unique=False) if past else p
    return pool_fn


def precompute(ctx, queries: pd.DataFrame, pool_fn, progress=None,
               hide_user_history: bool = False) -> list:
    """Calcule une fois les composantes de chaque requête (réutilisées par toutes
    les méthodes et par le réglage des hyper-paramètres).

    hide_user_history=True : on fait comme si le candidat n'avait aucun
    historique (simulation de nouvel utilisateur sur des candidats connus).
    """
    out = []
    it = queries.itertuples(index=False)
    for row in (progress(it, total=len(queries)) if progress else it):
        u, t = int(row.u), int(row.t)
        pool = pool_fn(u, t)  # calculé AVANT masquage : on exclut toujours ses offres passées
        ctx.hist.hidden_user = u if hide_user_history else None
        if len(pool) == 0:
            continue
        comp = components(ctx, u, t, pool)
        comp["item_count_before"] = ctx.hist.item_counts(pool, t)
        ctx.hist.hidden_user = None
        out.append((u, t, pool, list(row.true_items), comp))
    return out


def rank_order(scores: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Ordre décroissant ; ex-aequo départagés par une clé pseudo-aléatoire fixe
    (pour ne pas avantager l'ordre du fichier)."""
    tie = (pool * 2654435761 + TIE_SEED) % 1000003
    return np.lexsort((tie, -scores))


def ndcg_for(order, pool, true, k):
    rel = np.isin(pool[order[:k]], true).astype(float)
    dcg = (rel / np.log2(np.arange(2, len(rel) + 2))).sum()
    ideal = min(len(true), k)
    idcg = (1 / np.log2(np.arange(2, ideal + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def quick_ndcg(cached, score_fn, k=10) -> float:
    """NDCG@k moyen — utilisé pour régler les hybrides sur la validation."""
    vals = []
    for u, t, pool, true, comp in cached:
        vals.append(ndcg_for(rank_order(score_fn(comp), pool), pool, true, k))
    return float(np.mean(vals)) if vals else 0.0


def evaluate(cached, methods: dict, ctx, ks=(5, 10), extra=None) -> pd.DataFrame:
    """Évalue chaque méthode ; renvoie une ligne par (requête, méthode).

    methods : nom -> fonction(comp, u, t, pool) -> scores
    """
    J = ctx.prof.J_tfidf
    kmax = max(ks)
    rows = []
    for qi, (u, t, pool, true, comp) in enumerate(cached):
        true_arr = np.array(true)
        cold_user = comp["n_past"] == 0
        cnt = comp["item_count_before"]
        n_before = ctx.hist.inter.t.searchsorted(t)  # nb d'interactions avant t
        cold_true = [j for j in true if j in set(pool[cnt == 0])]
        for name, fn in methods.items():
            s = fn(comp, u, t, pool)
            order = rank_order(s, pool)
            top = pool[order[:kmax]]
            hits = np.isin(pool[order], true_arr)
            first = np.flatnonzero(hits)
            rec = {"q": qi, "method": name, "u": u, "t": t, "cold_user": cold_user,
                   "n_true": len(true), "mrr": 1 / (first[0] + 1) if len(first) else 0.0,
                   "top": top}
            for k in ks:
                h = hits[:k].sum()
                rec[f"P@{k}"] = h / k
                rec[f"R@{k}"] = h / len(true)
                rec[f"HR@{k}"] = float(h > 0)
                rec[f"NDCG@{k}"] = ndcg_for(order, pool, true_arr, k)
            # Cold-start côté offre : les offres vraies sans aucune interaction avant t
            rec["n_cold_true"] = len(cold_true)
            rec[f"cold_item_hits@{kmax}"] = int(np.isin(cold_true, top).sum())
            # Diversité intra-liste : 1 - similarité cosinus moyenne entre paires
            V = J[top]
            S = (V @ V.T).toarray()
            n = len(top)
            rec["diversity"] = 1 - (S.sum() - np.trace(S)) / (n * (n - 1)) if n > 1 else 0.0
            # Nouveauté : -log2 de la popularité (lissée) des offres recommandées
            c_top = cnt[order[:kmax]]
            p = (c_top + 1) / (n_before + ctx.hist.n_items)
            rec["novelty"] = float(np.mean(-np.log2(p)))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cached, ks=(5, 10), group: str = "all") -> pd.DataFrame:
    if group == "cold":
        df = df[df.cold_user]
    elif group == "warm":
        df = df[~df.cold_user]
    kmax = max(ks)
    all_pool = set()
    for _, _, pool, _, _ in cached:
        all_pool.update(pool.tolist())
    out = {}
    for name, g in df.groupby("method", sort=False):
        r = {}
        for k in ks:
            for m in ("P", "R", "NDCG", "HR"):
                r[f"{m}@{k}"] = g[f"{m}@{k}"].mean()
        r["MRR"] = g["mrr"].mean()
        rec_items = set(np.concatenate(g["top"].values).tolist())
        r["Couverture"] = len(rec_items) / max(len(all_pool), 1)
        r["Diversité"] = g["diversity"].mean()
        r["Nouveauté"] = g["novelty"].mean()
        nc = g["n_cold_true"].sum()
        r[f"HR@{kmax} offres nouvelles"] = g[f"cold_item_hits@{kmax}"].sum() / nc if nc else np.nan
        r["n_requêtes"] = len(g)
        out[name] = r
    return pd.DataFrame(out).T
