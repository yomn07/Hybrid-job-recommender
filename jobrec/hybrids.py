"""
hybrids.py — Les 7 stratégies d'hybridation (classification de Burke, 2002).

Pour éviter de recalculer les moteurs de base pour chaque hybride, on calcule une
fois par requête un dictionnaire de « composantes » (scores de chaque moteur +
matrice des contraintes) ; chaque hybride est une fonction de ces composantes.

  1. Pondéré (weighted)          : somme pondérée des scores normalisés.
  2. Commutation (switching)     : choix du moteur selon le profil (nouveau / connu).
  3. Mixte (mixed)               : entrelacement de listes issues de 3 moteurs.
  4. Cascade                     : filtre d'éligibilité (connaissances) puis tri (contenu).
  5. Combinaison de features     : un seul modèle (régression logistique) sur des
                                   caractéristiques brutes candidat + offre.
  6. Augmentation de features    : le même modèle, augmenté des SORTIES des autres
                                   moteurs (contenu, CF, méta) comme features.
  7. Méta-niveau                 : le profil appris par le contenu sert d'entrée au
                                   user-kNN collaboratif (voir engines.MetaUserKNN).
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .engines import (CFRec, ContentRec, Context, KnowledgeRec, MetaUserKNN,
                      PopularityRec)

POP, CB, CF, KB = PopularityRec(), ContentRec(), CFRec(), KnowledgeRec()
META = MetaUserKNN(CB)
BASE_KEYS = ["pop", "cb", "cf", "kb", "meta"]


def components(ctx: Context, u: int, t: int, pool: np.ndarray) -> dict:
    """Scores de tous les moteurs de base pour une requête (calculés une fois)."""
    M = KB.constraints(ctx, u, pool)
    return {
        "pop": POP.score(ctx, u, t, pool),
        "cb": CB.score(ctx, u, t, pool),
        "cf": CF.score(ctx, u, t, pool),
        "kb": KB.score(ctx, u, t, pool, M),
        "meta": META.score(ctx, u, t, pool),
        "kb_matrix": M,
        "n_past": len(ctx.hist.items_of(u, t)),
    }


def slice_comp(comp: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in comp.items()}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(initial=0), x.max(initial=0)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x, dtype=float)


def rank_scores(order: np.ndarray, n: int) -> np.ndarray:
    """Transforme un ordre (liste d'indices) en scores décroissants."""
    s = np.zeros(n)
    s[order] = np.arange(n, 0, -1)
    return s


# ---------------------------------------------------------------------------
# 1. Pondéré
# ---------------------------------------------------------------------------
def weighted(comp: dict, w: dict) -> np.ndarray:
    return sum(w.get(k, 0) * minmax(comp[k]) for k in ("pop", "cb", "cf", "kb", "meta"))


def weight_grid(keys=("pop", "cb", "cf", "kb"), step=0.1):
    n = int(round(1 / step))
    for combo in itertools.product(range(n + 1), repeat=len(keys)):
        if sum(combo) == n:
            yield {k: c * step for k, c in zip(keys, combo)}


# ---------------------------------------------------------------------------
# 2. Commutation
# ---------------------------------------------------------------------------
def switching(comp: dict, cfg: dict) -> np.ndarray:
    """cfg = {'n_min': int, 'warm': clé moteur, 'cold': clé moteur} — appris sur la
    validation. Le CF n'est utilisable que si le candidat a un historique."""
    warm = comp["n_past"] >= cfg["n_min"] and comp["cf"].max(initial=0) > 0
    main = comp[cfg["warm"] if warm else cfg["cold"]]
    # départage léger par le contenu (évite les ex-aequo massifs)
    return minmax(main) + 1e-3 * minmax(comp["cb"])


# ---------------------------------------------------------------------------
# 3. Mixte
# ---------------------------------------------------------------------------
def mixed(comp: dict, keys=("cb", "meta", "kb")) -> np.ndarray:
    """Entrelacement round-robin : 1er du contenu, 1er des candidats similaires,
    1er des contraintes, 2e du contenu, ... (sans doublons)."""
    n = len(comp["cb"])
    lists = [np.lexsort((-comp["pop"], -comp[k])) for k in keys]  # ex-aequo : popularité
    seen, order = set(), []
    for rank in range(n):
        for lst in lists:
            j = lst[rank]
            if j not in seen:
                seen.add(j)
                order.append(j)
    return rank_scores(np.array(order, dtype=int), n)


def mixed_sources(comp: dict, k: int = 10, keys=("cb", "meta", "kb")) -> list[tuple[int, str]]:
    """Même entrelacement mais en gardant le moteur d'origine (pour la démo)."""
    n = len(comp["cb"])
    lists = [np.lexsort((-comp["pop"], -comp[key])) for key in keys]
    seen, out = set(), []
    for rank in range(n):
        for key, lst in zip(keys, lists):
            j = int(lst[rank])
            if j not in seen:
                seen.add(j)
                out.append((j, key))
            if len(out) >= k:
                return out
    return out


# ---------------------------------------------------------------------------
# 4. Cascade
# ---------------------------------------------------------------------------
def cascade(comp: dict, tau: float, second: dict) -> np.ndarray:
    """Étape 1 : on ne garde que les offres éligibles (score connaissances >= tau).
    Étape 2 : les éligibles sont triées par un second moteur (pondération `second`).
    Les non-éligibles sont reléguées en fin de liste."""
    eligible = comp["kb"] >= tau
    s = weighted(comp, second)
    return s + 10.0 * eligible   # tout éligible passe devant tout non-éligible


# ---------------------------------------------------------------------------
# 5 & 6. Combinaison / augmentation de features (régression logistique)
# ---------------------------------------------------------------------------
RAW_FEATURES = ["jac_skills", "cov_skills", "jac_domain", "jac_soft",
                "c_diplome", "c_experience", "c_seniorite", "c_localisation",
                "c_contrat", "c_langues", "kb_inconnues",
                "log_pop_recente", "log_pop_totale", "age_offre", "log_hist_candidat"]
AUG_FEATURES = RAW_FEATURES + ["score_contenu", "score_cf", "score_meta", "score_kb"]


def pair_features(ctx: Context, u: int, t: int, pool: np.ndarray, comp: dict,
                  augmented: bool) -> np.ndarray:
    p = ctx.prof
    M = comp["kb_matrix"][:, :6]
    unknown = (comp["kb_matrix"] == 0.5).sum(axis=1)
    cols = [
        p.jaccard("sk", u, pool), p.overlap("sk", u, pool),
        p.jaccard("dom", u, pool), p.jaccard("soft", u, pool),
        *[np.nan_to_num(M[:, i], nan=0.5) for i in range(6)], unknown,
        np.log1p(comp["pop"]), np.log1p(ctx.hist.item_counts(pool, t)),
        (t - p.j_create[pool]) / 365.0,
        np.full(len(pool), np.log1p(comp["n_past"])),
    ]
    if augmented:
        cols += [comp["cb"], np.log1p(comp["cf"]), np.log1p(comp["meta"]), comp["kb"]]
    return np.column_stack(cols).astype(np.float32)


class LTRHybrid:
    """Régression logistique point-par-point entraînée sur des couples
    (candidat, offre active) : positifs = offres réelles, négatifs = offres actives
    tirées au hasard au même moment (échantillonnage négatif)."""

    def __init__(self, augmented: bool, C: float = 1.0):
        self.augmented = augmented
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        self.feature_names = AUG_FEATURES if augmented else RAW_FEATURES

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def score_from_features(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)

    def coefficients(self) -> dict:
        lr = self.model[-1]
        return dict(zip(self.feature_names, lr.coef_[0].round(3)))


def build_ltr_dataset(ctx: Context, queries, pool_fn, n_neg: int = 20, seed: int = 0,
                      max_queries: int | None = 6000, progress=None):
    """Construit (X_raw, X_aug, y) à partir de requêtes de la période d'entraînement."""
    rng = np.random.default_rng(seed)
    q = queries
    if max_queries and len(q) > max_queries:
        q = q.sample(max_queries, random_state=seed)
    Xr, Xa, Y = [], [], []
    it = q.itertuples(index=False)
    for row in (progress(it, total=len(q)) if progress else it):
        u, t = int(row.u), int(row.t)
        pool = pool_fn(u, t)
        is_pos = np.isin(pool, row.true_items)
        if not is_pos.any() or is_pos.all():
            continue
        neg = np.flatnonzero(~is_pos)
        neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
        sel = np.concatenate([np.flatnonzero(is_pos), neg])
        # Les composantes sont calculées sur TOUTE l'offre active (comme au test),
        # puis on ne garde que les lignes échantillonnées.
        comp = slice_comp(components(ctx, u, t, pool), sel)
        Xa.append(pair_features(ctx, u, t, pool[sel], comp, augmented=True))
        Y.append(is_pos[sel].astype(int))
    Xa = np.vstack(Xa)
    return Xa[:, : len(RAW_FEATURES)], Xa, np.concatenate(Y)
