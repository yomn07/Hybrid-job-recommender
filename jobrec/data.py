"""
data.py — Chargement, nettoyage et découpage temporel du dataset JTH.

JTH (Job Tracking History, Romero et al., RecSys'26, CC BY-NC 4.0) contient :
  - candidates.csv : profils des candidats (compétences, diplôme, expérience...)
  - jobs.csv       : offres d'emploi (compétences requises, exigences...)
  - history.csv    : une ligne = un couple candidat–offre et les dates des étapes
                     du processus de recrutement (candidature, entretiens, offre...).

Choix de modélisation : chaque ligne de history est une interaction implicite
positive (le candidat a été positionné sur l'offre). La date de l'interaction est
la première date non vide du processus (comme dans le découpage officiel JTH).
"""
from __future__ import annotations

import ast
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Colonnes de dates du processus de recrutement, dans l'ordre de l'entonnoir.
STAGE_COLS = [
    "spontaneous_application_date", "shortlist_date", "qualification_date",
    "resume_sent_to_company_date", "1st_interview_date", "2nd_interview_date",
    "3rd_interview_date", "4th_interview_date", "job_offer_proposed_date",
    "job_offer_accepted_date",
]
EPOCH = pd.Timestamp("1970-01-01")


# ---------------------------------------------------------------------------
# Parsing des champs « liste »
# ---------------------------------------------------------------------------
def parse_list(value) -> list[str]:
    """Transforme un champ texte en liste de jetons normalisés.

    Les champs JTH mélangent plusieurs formats : « a;b;c » (champs manuels),
    « ['a', 'b'] » (champs extraits par LLM) ou « a, b ». On gère les trois,
    on met en minuscules et on supprime la sentinelle `_rare_skill_`.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    items: list
    if s.startswith("["):
        try:
            items = list(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            items = s.strip("[]").split(",")
    elif ";" in s:
        items = s.split(";")
    else:
        items = s.split(",")
    out = []
    for it in items:
        t = str(it).strip().strip("'\"").lower()
        t = re.sub(r"\s+", " ", t)
        if t and not t.startswith("_rare"):
            out.append(t)
    return out


def first_number(value) -> float:
    """Extrait le premier nombre d'un intervalle binné ('3-6 years' -> 3, '+6 years' -> 6)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    m = re.search(r"\d+(\.\d+)?", str(value))
    return float(m.group()) if m else np.nan


def to_bool(value) -> float:
    """Booléen tolérant : True/False/'true'/'1' -> 1.0/0.0, sinon NaN (inconnu)."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "1.0"}:
        return 1.0
    if s in {"false", "0", "no", "0.0"}:
        return 0.0
    return np.nan


DIPLOMA_RANK = [  # (mot-clé, rang) testés dans cet ordre
    ("phd", 5), ("doctor", 5), ("master", 4), ("mba", 4), ("engineer", 4),
    ("bachelor", 3), ("licence", 3), ("associate", 2), ("bts", 2), ("dut", 2),
    ("high school", 1), ("baccalaur", 1), ("none", 0), ("no diploma", 0),
]


def diploma_rank(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).lower()
    for key, r in DIPLOMA_RANK:
        if key in s:
            return float(r)
    return np.nan


SENIORITY_RANK = {"intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "executive": 4}


def seniority_rank(value) -> float:
    s = str(value).strip().lower()
    for k, r in SENIORITY_RANK.items():
        if k in s:
            return float(r)
    return np.nan


def to_days(series: pd.Series) -> pd.Series:
    """Date -> nombre entier de jours depuis 1970 (plus rapide à comparer)."""
    return (pd.to_datetime(series, errors="coerce") - EPOCH).dt.days


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
@dataclass
class JTHData:
    candidates: pd.DataFrame          # index 0..n_cand-1, colonne candidate_id
    jobs: pd.DataFrame                # index 0..n_jobs-1, colonne job_id
    interactions: pd.DataFrame        # colonnes u, j, t (jours), stage_depth, last_stage
    raw_history: pd.DataFrame
    cleaning_log: dict = field(default_factory=dict)


ZENODO_RECORD = 21390581
ZENODO_URL = "https://zenodo.org/api/records/{rec}/files/{name}/content"
JTH_FILES = {  # nom -> empreinte SHA-256 de la version officielle (Zenodo 21390581)
    "candidates.csv": "4328ceb1678757ce19b62105e97cca70759f38c332bef13cf9746badbbf3f692",
    "jobs.csv": "5bcd9e4379e558b98418e2242d321757612a96887b934e0d3ece6eb4a9952628",
    "history.csv": "0348bb0017156a91fe40fdd4346846982587940a0796dae3941da55312ade2a3",
}


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_jth(data_dir: str | Path) -> Path:
    """Télécharge les CSV JTH depuis Zenodo s'ils sont absents (licence CC BY-NC 4.0 :
    les données ne sont pas versionnées dans le dépôt). Vérifie l'empreinte SHA-256."""
    import urllib.request
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in JTH_FILES.items():
        dest = data_dir / name
        if dest.exists() and _sha256(dest) == digest:
            continue
        tmp = dest.with_suffix(".part")
        req = urllib.request.Request(ZENODO_URL.format(rec=ZENODO_RECORD, name=name),
                                     headers={"User-Agent": "hybrid-job-recommender"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        if _sha256(tmp) != digest:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name} : empreinte SHA-256 inattendue après téléchargement")
        tmp.replace(dest)
    return data_dir


def load_jth(data_dir: str | Path) -> JTHData:
    data_dir = Path(data_dir)
    cand = pd.read_csv(data_dir / "candidates.csv", dtype=str)
    jobs = pd.read_csv(data_dir / "jobs.csv", dtype=str)
    hist = pd.read_csv(data_dir / "history.csv", dtype=str)
    log = {"history_rows_raw": len(hist)}

    # Identifiants dupliqués éventuels : on garde la première occurrence.
    cand = cand.drop_duplicates("candidate_id").reset_index(drop=True)
    jobs = jobs.drop_duplicates("job_id").reset_index(drop=True)

    # Dates de création en jours. Une offre sans date est rare (0 % d'après la
    # fiche du dataset) ; on lui impute la date de sa première interaction plus bas.
    cand["create_day"] = to_days(cand["create_date"])
    jobs["create_day"] = to_days(jobs["create_date"])

    # --- Interactions -------------------------------------------------------
    stage_cols = [c for c in STAGE_COLS if c in hist.columns]
    dates = hist[stage_cols].apply(to_days)
    hist["t"] = dates.min(axis=1)                       # 1re étape = date d'interaction
    hist["stage_depth"] = dates.notna().sum(axis=1)     # nb d'étapes franchies
    before = len(hist)
    hist = hist.dropna(subset=["t"])
    log["dropped_no_date"] = before - len(hist)

    # Clés étrangères inconnues
    cid = {c: i for i, c in enumerate(cand["candidate_id"])}
    jid = {c: i for i, c in enumerate(jobs["job_id"])}
    hist["u"] = hist["candidate_id"].map(cid)
    hist["j"] = hist["job_id"].map(jid)
    before = len(hist)
    hist = hist.dropna(subset=["u", "j"])
    log["dropped_unknown_ids"] = before - len(hist)

    # Doublons candidat–offre : on garde la plus ancienne interaction.
    before = len(hist)
    hist = hist.sort_values("t").drop_duplicates(["u", "j"], keep="first")
    log["dropped_duplicates"] = before - len(hist)

    inter = hist[["u", "j", "t", "stage_depth"]].astype(int).copy()
    inter["last_stage"] = hist.get("last_stage_reached", pd.Series("", index=hist.index)).values
    inter = inter.sort_values(["t", "u", "j"]).reset_index(drop=True)

    # Imputation des dates de création manquantes des offres
    first_t = inter.groupby("j")["t"].min()
    miss = jobs["create_day"].isna()
    jobs.loc[miss, "create_day"] = jobs.index[miss].map(first_t)
    jobs["create_day"] = jobs["create_day"].fillna(jobs["create_day"].median())
    log["history_rows_clean"] = len(inter)
    return JTHData(cand, jobs, inter, hist, log)


# ---------------------------------------------------------------------------
# Découpage temporel
# ---------------------------------------------------------------------------
def temporal_split(inter: pd.DataFrame, frac: float = 0.8):
    """Découpage chronologique : les `frac` premières interactions (par date) en
    entraînement, le reste en test. Évite toute fuite d'information du futur."""
    inter = inter.sort_values(["t", "u", "j"]).reset_index(drop=True)
    cut = int(len(inter) * frac)
    return inter.iloc[:cut].copy(), inter.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Historique interrogeable « à la date t »
# ---------------------------------------------------------------------------
class History:
    """Toutes les interactions connues, interrogeables à une date t.

    Pour une requête à la date t, on n'utilise QUE les interactions strictement
    antérieures à t : c'est ce que connaîtrait un vrai système en production le
    jour t. Les paramètres appris (TF-IDF, régression logistique, poids) sont,
    eux, figés à la fin de la période d'entraînement.
    """

    def __init__(self, inter: pd.DataFrame, n_users: int, n_items: int):
        self.inter = inter.sort_values("t").reset_index(drop=True)
        self.n_users, self.n_items = n_users, n_items
        # Candidat dont on masque l'historique (expérience « nouvel utilisateur simulé »)
        self.hidden_user: int | None = None
        self.user_items: dict[int, list[tuple[int, int]]] = {}
        self.item_users: dict[int, list[tuple[int, int]]] = {}
        for u, j, t in zip(self.inter.u.values, self.inter.j.values, self.inter.t.values):
            self.user_items.setdefault(int(u), []).append((int(t), int(j)))
            self.item_users.setdefault(int(j), []).append((int(t), int(u)))
        # Dates triées des interactions de chaque offre (popularité à la date t)
        self.item_times = {j: np.array([t for t, _ in v]) for j, v in self.item_users.items()}

    def items_of(self, u: int, t: int) -> list[int]:
        if u == self.hidden_user:
            return []
        return [j for tt, j in self.user_items.get(u, []) if tt < t]

    def users_of(self, j: int, t: int) -> list[int]:
        return [u for tt, u in self.item_users.get(j, []) if tt < t and u != self.hidden_user]

    def item_counts(self, items: np.ndarray, t: int, window: int | None = None) -> np.ndarray:
        """Nombre d'interactions de chaque offre avant t (dans la fenêtre si donnée)."""
        out = np.zeros(len(items))
        for k, j in enumerate(items):
            ts = self.item_times.get(int(j))
            if ts is None:
                continue
            hi = bisect_left(ts, t)
            lo = bisect_left(ts, t - window) if window else 0
            out[k] = hi - lo
        return out


class JobPool:
    """Offres « actives » à la date t : créées avant t et depuis moins de `window` jours.

    En recrutement, une offre a une durée de vie limitée ; recommander une offre
    pourvue depuis deux ans n'a pas de sens. La fenêtre est estimée sur les données
    d'entraînement (quantile de l'âge des offres au moment des candidatures).
    """

    def __init__(self, create_day: np.ndarray, window: int):
        self.order = np.argsort(create_day, kind="stable")
        self.sorted_days = create_day[self.order]
        self.window = int(window)
        self._cache: dict[int, np.ndarray] = {}

    def at(self, t: int) -> np.ndarray:
        if t not in self._cache:
            lo = np.searchsorted(self.sorted_days, t - self.window, side="left")
            hi = np.searchsorted(self.sorted_days, t, side="right")
            self._cache[t] = np.sort(self.order[lo:hi])
        return self._cache[t]


def estimate_pool_window(inter: pd.DataFrame, job_create_day: np.ndarray, q: float = 0.95) -> int:
    age = inter["t"].values - job_create_day[inter["j"].values]
    age = age[age >= 0]
    return int(np.ceil(np.quantile(age, q))) if len(age) else 180


def build_queries(test: pd.DataFrame) -> pd.DataFrame:
    """Une requête = (candidat, date) avec l'ensemble des offres vraies ce jour-là."""
    q = test.groupby(["u", "t"])["j"].apply(lambda s: sorted(set(s))).reset_index()
    q = q.rename(columns={"j": "true_items"}).sort_values("t").reset_index(drop=True)
    return q
