"""
engines.py — Les moteurs de recommandation de base.

Tous les moteurs ont la même interface : pour un candidat u, une date t et un
ensemble d'offres actives `pool`, ils renvoient un tableau de scores (plus grand =
plus pertinent), aligné sur `pool`.

  * PopularityRec : baseline — offres ayant reçu le plus de candidatures récemment.
  * ContentRec    : contenu — similarité cosinus TF-IDF profil candidat / offre.
  * CFRec         : filtrage collaboratif item-kNN — « les candidats positionnés
                    sur les mêmes offres que vous l'ont aussi été sur... ».
  * KnowledgeRec  : connaissances — satisfaction des contraintes explicites
                    (diplôme, expérience, lieu, contrat, langues, management).
  * MetaUserKNN   : (utilisé par l'hybride méta-niveau) user-kNN dont les voisins
                    sont trouvés grâce au profil appris par le modèle de contenu.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from .data import History
from .features import Profiles


class Context:
    """Regroupe tout ce dont les moteurs ont besoin."""

    def __init__(self, prof: Profiles, hist: History, pop_window: int = 30,
                 profile_beta: float = 0.5, knn_k: int = 50):
        self.prof, self.hist = prof, hist
        self.pop_window = pop_window      # fenêtre (jours) de la popularité récente
        self.profile_beta = profile_beta  # poids des offres passées dans le profil contenu
        self.knn_k = knn_k                # nb de voisins du user-kNN méta-niveau


# ---------------------------------------------------------------------------
class PopularityRec:
    name = "Popularité"

    def score(self, ctx: Context, u: int, t: int, pool: np.ndarray) -> np.ndarray:
        recent = ctx.hist.item_counts(pool, t, window=ctx.pop_window)
        total = ctx.hist.item_counts(pool, t)
        return recent + 1e-3 * total      # la popularité totale départage les ex-aequo


# ---------------------------------------------------------------------------
class ContentRec:
    name = "Contenu (TF-IDF)"

    def user_vector(self, ctx: Context, u: int, t: int):
        """Profil du candidat = son CV (TF-IDF) enrichi de la moyenne des offres
        sur lesquelles il a déjà été positionné (si utilisateur « chaud »)."""
        v = ctx.prof.C_tfidf[u]
        past = ctx.hist.items_of(u, t)
        if past:
            m = sparse.csr_matrix(ctx.prof.J_tfidf[past].mean(axis=0))
            v = v + ctx.profile_beta * m
        return normalize(v)

    def score(self, ctx: Context, u: int, t: int, pool: np.ndarray) -> np.ndarray:
        v = self.user_vector(ctx, u, t)
        return np.asarray((ctx.prof.J_tfidf[pool] @ v.T).todense()).ravel()


# ---------------------------------------------------------------------------
class CFRec:
    """Item-kNN : sim(i,k) = |U_i ∩ U_k| / sqrt(|U_i|·|U_k|) (cosinus binaire)
    calculée sur l'historique antérieur à t. Score(k) = Σ_{i ∈ passé(u)} sim(i,k)."""
    name = "Collaboratif (item-kNN)"

    def score(self, ctx: Context, u: int, t: int, pool: np.ndarray) -> np.ndarray:
        h = ctx.hist
        past = h.items_of(u, t)
        out = np.zeros(len(pool))
        if not past:
            return out                     # nouvel utilisateur : le CF ne sait rien
        pos = {int(j): k for k, j in enumerate(pool)}
        deg_cache: dict[int, int] = {}

        def deg(j):
            if j not in deg_cache:
                deg_cache[j] = len(h.users_of(j, t))
            return deg_cache[j]

        for i in past:
            co = defaultdict(int)
            for v in h.users_of(i, t):
                if v == u:
                    continue
                for k in h.items_of(v, t):
                    if k in pos and k != i:
                        co[k] += 1
            for k, c in co.items():
                out[pos[k]] += c / np.sqrt(max(deg(i), 1) * max(deg(k), 1))
        return out


# ---------------------------------------------------------------------------
KB_CONSTRAINTS = ["diplôme", "expérience", "séniorité", "localisation", "contrat", "langues", "management"]


class KnowledgeRec:
    """Chaque contrainte vaut 1 (satisfaite), 0 (violée) ou 0.5 (information
    manquante d'un côté) ; NaN si non applicable. Score = moyenne des contraintes
    applicables. Ce moteur ne dépend d'aucune interaction : il fonctionne pour
    les nouveaux candidats et les nouvelles offres."""
    name = "Connaissances (contraintes)"

    def constraints(self, ctx: Context, u: int, pool: np.ndarray) -> np.ndarray:
        p = ctx.prof
        n = len(pool)
        M = np.full((n, len(KB_CONSTRAINTS)), np.nan)

        def tri(known, ok):  # 1 / 0 / 0.5
            return np.where(known, ok.astype(float), 0.5)

        # Diplôme : diplôme du candidat >= diplôme minimal requis
        cd, jd = p.c_dipl[u], p.j_dipl[pool]
        M[:, 0] = tri(~np.isnan(cd) & ~np.isnan(jd), np.nan_to_num(cd) >= np.nan_to_num(jd))
        # Expérience : années du candidat >= années requises - 1 (tolérance)
        cy, jy = p.c_years[u], p.j_years[pool]
        M[:, 1] = tri(~np.isnan(cy) & ~np.isnan(jy), np.nan_to_num(cy) >= np.nan_to_num(jy) - 1)
        # Séniorité : écart d'au plus un niveau
        cs, js = p.c_senior[u], p.j_senior[pool]
        M[:, 2] = tri(~np.isnan(cs) & ~np.isnan(js), np.abs(np.nan_to_num(cs) - np.nan_to_num(js)) <= 1)
        # Localisation : même département, ou télétravail possible
        cz, jz, rem = p.c_zip[u], p.j_zip[pool], p.j_remote[pool]
        known = (cz != "") & (jz != "")
        loc = tri(known, jz == cz)
        M[:, 3] = np.where(rem == 1, 1.0, loc)
        # Contrat : au moins un type de contrat en commun
        cc = p.c_contract[u]
        M[:, 4] = [0.5 if not cc or not p.j_contract[j] else float(bool(cc & p.j_contract[j])) for j in pool]
        # Langues : toutes les langues requises sont parlées
        cl = p.c_lang[u]

        def lang_ok(req):
            if not req or not cl:
                return 0.5
            return float(all(any(r in s or s in r for s in cl) for r in req))
        M[:, 5] = [lang_ok(p.j_lang[j]) for j in pool]
        # Management : applicable seulement si l'offre l'exige
        jm, cm = p.j_mgmt[pool], p.c_mgmt[u]
        M[:, 6] = np.where(jm == 1, 0.5 if np.isnan(cm) else float(cm == 1), np.nan)
        return M

    def score(self, ctx: Context, u: int, t: int, pool: np.ndarray, M=None) -> np.ndarray:
        M = self.constraints(ctx, u, pool) if M is None else M
        with np.errstate(invalid="ignore"):
            s = np.nanmean(M, axis=1)
        return np.nan_to_num(s, nan=0.5)


# ---------------------------------------------------------------------------
class MetaUserKNN:
    """Méta-niveau : le modèle de CONTENU apprend un profil vectoriel pour chaque
    candidat ; ce profil (et non les interactions brutes) sert ensuite d'entrée à
    un filtrage COLLABORATIF user-kNN. Les voisins sont les candidats déjà
    positionnés sur une offre active ; un nouveau candidat a donc des voisins."""
    name = "Méta-niveau (contenu → user-kNN)"

    def __init__(self, content: ContentRec):
        self.content = content

    def score(self, ctx: Context, u: int, t: int, pool: np.ndarray) -> np.ndarray:
        h = ctx.hist
        pos = {int(j): k for k, j in enumerate(pool)}
        applicants = defaultdict(list)          # voisin potentiel -> offres actives
        for j in pool:
            for v in h.users_of(int(j), t):
                if v != u:
                    applicants[v].append(pos[int(j)])
        out = np.zeros(len(pool))
        if not applicants:
            return out
        vs = np.fromiter(applicants.keys(), dtype=int)
        pu = self.content.user_vector(ctx, u, t)
        sims = np.asarray((ctx.prof.C_tfidf[vs] @ pu.T).todense()).ravel()
        top = np.argsort(-sims)[: ctx.knn_k]
        for idx in top:
            if sims[idx] <= 0:
                break
            for k in applicants[vs[idx]]:
                out[k] += sims[idx]
        return out
