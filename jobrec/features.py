"""
features.py — Représentations des candidats et des offres.

On construit, à partir des colonnes JTH :
  1. des ensembles de jetons par « famille » (compétences, domaine, soft skills),
     préfixés pour partager un vocabulaire commun candidats/offres ;
  2. une matrice TF-IDF (moteur à base de contenu) ;
  3. des attributs explicites (diplôme, expérience, lieu, contrat...) pour le
     moteur à base de connaissances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from .data import (diploma_rank, first_number, parse_list, seniority_rank, to_bool)

# Familles de champs (on ignore silencieusement les colonnes absentes)
SKILL_COLS = ["skills", "llm_hard_skills", "llm_programming_languages",
              "llm_tools_technologies", "llm_certifications"]
DOMAIN_COLS = ["expertise_area", "job_category", "llm_expertise_area",
               "llm_job_category", "llm_industry_domains"]
SOFT_COLS = ["llm_soft_skills"]
FAMILIES = {"sk": SKILL_COLS, "dom": DOMAIN_COLS, "soft": SOFT_COLS}


def _tokens(df: pd.DataFrame, cols: list[str], prefix: str) -> list[list[str]]:
    cols = [c for c in cols if c in df.columns]
    if not cols:                       # aucune colonne de cette famille (ex. profil saisi)
        return [[] for _ in range(len(df))]
    out = []
    for row in df[cols].itertuples(index=False):
        toks = set()
        for v in row:
            toks.update(parse_list(v))
        out.append(sorted(f"{prefix}:{t}" for t in toks))
    return out


def _col(df, name, fn):
    return df[name].map(fn).values.astype(float) if name in df.columns else np.full(len(df), np.nan)


def _set_col(df, names):
    names = [n for n in names if n in df.columns]
    res = []
    for row in df[names].itertuples(index=False) if names else [()] * len(df):
        s = set()
        for v in row:
            s.update(parse_list(v))
        res.append(s)
    return res


def _zip(df: pd.DataFrame) -> np.ndarray:
    """Code département normalisé : '75.0' -> '75', '1' -> '01', vide si absent."""
    if "zipcode" not in df:
        return np.array([""] * len(df))
    z = df["zipcode"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return np.where(z == "", "", z.str.zfill(2).str[:2]).astype(str)


class Profiles:
    """Toutes les représentations utiles, calculées une seule fois."""

    def __init__(self, cand: pd.DataFrame, jobs: pd.DataFrame, fit_cutoff_day: int,
                 family_weights: dict | None = None):
        self.family_weights = family_weights or {"sk": 1.0, "dom": 1.0, "soft": 0.3}
        # --- 1. Jetons par famille -----------------------------------------
        self.c_tok = {f: _tokens(cand, cols, f) for f, cols in FAMILIES.items()}
        self.j_tok = {f: _tokens(jobs, cols, f) for f, cols in FAMILIES.items()}

        # Matrices binaires (une colonne par jeton) — utiles pour Jaccard & explications
        self.vec, self.C_bin, self.J_bin = {}, {}, {}
        for f in FAMILIES:
            v = CountVectorizer(analyzer=lambda x: x, binary=True)
            v.fit(self.c_tok[f] + self.j_tok[f])
            self.vec[f] = v
            self.C_bin[f] = v.transform(self.c_tok[f]).tocsr().astype(np.float32)
            self.J_bin[f] = v.transform(self.j_tok[f]).tocsr().astype(np.float32)

        # --- 2. TF-IDF (contenu) --------------------------------------------
        # L'IDF n'est appris que sur les profils existant avant la fin de
        # l'entraînement (pas de fuite de vocabulaire du futur).
        C_all = sparse.hstack([self.C_bin[f] * self.family_weights[f] for f in FAMILIES]).tocsr()
        J_all = sparse.hstack([self.J_bin[f] * self.family_weights[f] for f in FAMILIES]).tocsr()
        c_ok = (cand["create_day"].isna() | (cand["create_day"] <= fit_cutoff_day)).values
        j_ok = (jobs["create_day"] <= fit_cutoff_day).values
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.tfidf.fit(sparse.vstack([C_all[c_ok], J_all[j_ok]]))
        self.C_tfidf = normalize(self.tfidf.transform(C_all)).tocsr()
        self.J_tfidf = normalize(self.tfidf.transform(J_all)).tocsr()

        # --- 3. Attributs explicites (connaissances) -------------------------
        yrs = _col(cand, "llm_years_of_work_experience", first_number)
        yrs_manual = _col(cand, "years_experience", first_number)
        self.c_years = np.where(np.isnan(yrs), yrs_manual, yrs)
        self.c_dipl = _col(cand, "llm_highest_diploma", diploma_rank)
        self.c_senior = _col(cand, "llm_seniority_level", seniority_rank)
        self.c_mgmt = _col(cand, "llm_management_experience", to_bool)
        self.c_zip = _zip(cand)
        self.c_contract = _set_col(cand, ["contract_type"])
        self.c_lang = _set_col(cand, ["llm_languages_spoken"])

        jyrs = _col(jobs, "llm_required_years_of_work_experience", first_number)
        jyrs_manual = _col(jobs, "years_experience", first_number)
        self.j_years = np.where(np.isnan(jyrs), jyrs_manual, jyrs)
        self.j_dipl = _col(jobs, "llm_required_lowest_diploma", diploma_rank)
        self.j_senior = _col(jobs, "llm_seniority_level", seniority_rank)
        self.j_mgmt = _col(jobs, "llm_required_management_experience", to_bool)
        self.j_remote = _col(jobs, "llm_remote_possible", to_bool)
        self.j_zip = _zip(jobs)
        self.j_contract = _set_col(jobs, ["contract_type"])
        self.j_lang = _set_col(jobs, ["llm_required_languages_spoken"])
        self.j_create = jobs["create_day"].values.astype(int)

    # ------------------------------------------------------------------
    def jaccard(self, family: str, u: int, items: np.ndarray) -> np.ndarray:
        """Indice de Jaccard entre les jetons du candidat u et ceux de chaque offre."""
        cu = self.C_bin[family][u]
        J = self.J_bin[family][items]
        inter = np.asarray((J @ cu.T).todense()).ravel()
        size_u = cu.sum()
        size_j = np.asarray(J.sum(axis=1)).ravel()
        union = size_u + size_j - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    def overlap(self, family: str, u: int, items: np.ndarray) -> np.ndarray:
        """Part des jetons requis par l'offre que le candidat possède (couverture)."""
        cu = self.C_bin[family][u]
        J = self.J_bin[family][items]
        inter = np.asarray((J @ cu.T).todense()).ravel()
        size_j = np.asarray(J.sum(axis=1)).ravel()
        return np.divide(inter, size_j, out=np.zeros_like(inter), where=size_j > 0)

    def tokens(self, side: str, family: str, idx: int) -> list[str]:
        toks = (self.c_tok if side == "c" else self.j_tok)[family][idx]
        return [t.split(":", 1)[1] for t in toks]


def add_candidate(prof: Profiles, row: dict) -> int:
    """Ajoute un NOUVEAU candidat (saisi dans la démo) aux représentations existantes
    sans rien réapprendre : on réutilise le vocabulaire et l'IDF déjà appris.
    Renvoie l'indice du nouveau candidat (cold-start utilisateur)."""
    df = pd.DataFrame([row])
    for f, cols in FAMILIES.items():
        toks = _tokens(df, cols, f)
        prof.c_tok[f].append(toks[0])
        prof.C_bin[f] = sparse.vstack([prof.C_bin[f], prof.vec[f].transform(toks).astype(np.float32)]).tocsr()
    new_all = sparse.hstack([prof.C_bin[f][-1] * prof.family_weights[f] for f in FAMILIES]).tocsr()
    prof.C_tfidf = sparse.vstack([prof.C_tfidf, normalize(prof.tfidf.transform(new_all))]).tocsr()
    yrs = first_number(row.get("llm_years_of_work_experience"))
    prof.c_years = np.append(prof.c_years, yrs)
    prof.c_dipl = np.append(prof.c_dipl, diploma_rank(row.get("llm_highest_diploma")))
    prof.c_senior = np.append(prof.c_senior, seniority_rank(row.get("llm_seniority_level")))
    prof.c_mgmt = np.append(prof.c_mgmt, to_bool(row.get("llm_management_experience")))
    prof.c_zip = np.append(prof.c_zip, _zip(df)[0])
    prof.c_contract = prof.c_contract + _set_col(df, ["contract_type"])
    prof.c_lang = prof.c_lang + _set_col(df, ["llm_languages_spoken"])
    return prof.C_tfidf.shape[0] - 1
