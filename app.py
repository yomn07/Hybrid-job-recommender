"""
app.py — Démonstration Streamlit du système de recommandation hybride d'offres d'emploi.

Lancement :  streamlit run app.py
Prérequis  :  data/{candidates,jobs,history}.csv  et  models.joblib (produit par le notebook).

Deux modes :
  * « Candidat du jeu de test » : on se place à la date de sa requête de test et on
    indique quelles offres il a réellement obtenues (✓) ;
  * « Nouveau profil » : on saisit un profil → démonstration du cold-start utilisateur.
Chaque recommandation est accompagnée d'une explication (compétences, critères,
signal collaboratif) et d'une analyse des compétences manquantes.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from jobrec import hybrids as H
from jobrec.data import History, JobPool, build_queries, load_jth, temporal_split
from jobrec.engines import Context
from jobrec.evaluation import make_pool_fn, rank_order
from jobrec.explain import explain
from jobrec.features import Profiles, add_candidate

st.set_page_config(page_title="Recommandation d'offres d'emploi", page_icon="💼", layout="wide")

SOURCE_LABEL = {"cb": "compétences (contenu)", "meta": "candidats similaires", "kb": "critères (connaissances)"}


@st.cache_resource(show_spinner="Chargement des données et des modèles…")
def load_all():
    D = load_jth("data")
    cfg = joblib.load("models.joblib")
    train, test = temporal_split(D.interactions, 0.8)
    hist = History(D.interactions, len(D.candidates), len(D.jobs))
    prof = Profiles(D.candidates, D.jobs, fit_cutoff_day=cfg["train_end"])
    ctx = Context(prof, hist, pop_window=30, profile_beta=0.5, knn_k=50)
    pool = JobPool(D.jobs.create_day.values.astype(int), cfg["window"])
    return D, cfg, ctx, pool, build_queries(test)


D, cfg, ctx, pool, q_test = load_all()
jobs = D.jobs


def methods(cfg):
    def ltr(m, aug):
        return lambda c, u, t, p: m.score_from_features(H.pair_features(ctx, u, t, p, c, augmented=aug))
    return {
        "H6 Augmentation de features (meilleur)": ltr(cfg["ltr_aug"], True),
        "H1 Pondéré": lambda c, u, t, p: H.weighted(c, cfg["W_BEST"]),
        "H2 Commutation": lambda c, u, t, p: H.switching(c, cfg["SW"]),
        "H3 Mixte": lambda c, u, t, p: H.mixed(c),
        "H4 Cascade": lambda c, u, t, p: H.cascade(c, cfg["TAU"], cfg["second"]),
        "H5 Combinaison de features": ltr(cfg["ltr_raw"], False),
        "H7 Méta-niveau": lambda c, u, t, p: c["meta"],
        "Contenu seul": lambda c, u, t, p: c["cb"],
        "Collaboratif seul": lambda c, u, t, p: c["cf"],
        "Connaissances seul": lambda c, u, t, p: c["kb"],
        "Popularité (baseline)": lambda c, u, t, p: c["pop"],
    }


METHODS = methods(cfg)


def job_title(j: int) -> str:
    """JTH ne fournit pas d'intitulé : on en compose un à partir des attributs."""
    r = jobs.iloc[j]
    cat = ctx.prof.tokens("j", "dom", j)
    head = (cat[0] if cat else "offre").title()
    sen = r.get("llm_seniority_level") if isinstance(r.get("llm_seniority_level"), str) else ""
    return f"{head} — {sen}".strip(" —")


# ---------------------------------------------------------------------------
st.title("💼 Recommandation hybride d'offres d'emploi")
st.caption("Dataset JTH (Romero et al., RecSys'26, CC BY-NC 4.0). Le score affiché est un score de "
           "correspondance relatif, pas une probabilité d'embauche.")

with st.sidebar:
    mode = st.radio("Profil", ["Candidat du jeu de test", "Nouveau profil (cold-start)"])
    strategy = st.selectbox("Stratégie", list(METHODS))
    k = st.slider("Nombre de recommandations", 5, 20, 10)

true_items = []
if mode == "Candidat du jeu de test":
    with st.sidebar:
        only_warm = st.checkbox("Seulement les candidats ayant un historique", value=False)
        q = q_test
        if only_warm:
            q = q[[len(ctx.hist.items_of(int(u), int(t))) > 0 for u, t in zip(q.u, q.t)]]
        sample = q.sample(min(300, len(q)), random_state=1).sort_values("t")
        labels = [f"{D.candidates.candidate_id.iloc[int(r.u)]} · {pd.to_datetime(int(r.t), unit='D').date()}"
                  for r in sample.itertuples()]
        pick = st.selectbox("Candidat · date de la requête", range(len(sample)), format_func=lambda i: labels[i])
    row = sample.iloc[pick]
    u, t, true_items = int(row.u), int(row.t), list(row.true_items)
else:
    with st.sidebar:
        skills = st.text_input("Compétences (séparées par ;)", "python;sql;machine learning;pandas")
        domain = st.text_input("Domaine / catégorie", "data science")
        diploma = st.selectbox("Diplôme", ["Bachelor", "Master", "PhD", "High School"], index=1)
        years = st.selectbox("Expérience (années)", ["0-2", "2-5", "5-10", "10-15"])
        seniority = st.selectbox("Séniorité", ["Intern", "Junior", "Mid", "Senior"], index=1)
        contract = st.multiselect("Contrat", ["permanent", "freelance", "fixed-term", "internship"], ["permanent"])
        zipcode = st.text_input("Département (ex. 75)", "75")
        langs = st.text_input("Langues (séparées par ;)", "french;english")
        date = st.date_input("Date de la recherche", pd.Timestamp("2024-10-15"))
    if "new_profiles" not in st.session_state:
        st.session_state.new_profiles = {}
    key = (skills, domain, diploma, years, seniority, tuple(contract), zipcode, langs)
    if key not in st.session_state.new_profiles:     # on n'ajoute chaque profil qu'une fois
        st.session_state.new_profiles[key] = add_candidate(ctx.prof, {
            "skills": skills, "llm_hard_skills": skills, "expertise_area": domain, "llm_job_category": domain,
            "llm_highest_diploma": diploma, "llm_years_of_work_experience": years,
            "llm_seniority_level": seniority, "contract_type": ";".join(contract), "zipcode": zipcode,
            "llm_languages_spoken": langs, "llm_management_experience": False})
    u = st.session_state.new_profiles[key]
    t = int((pd.Timestamp(date) - pd.Timestamp("1970-01-01")).days)

# ---------------------------------------------------------------------------
pool_fn = make_pool_fn(ctx, pool)
p = pool_fn(u, t)
if len(p) == 0:
    st.warning("Aucune offre active à cette date.")
    st.stop()
comp = H.components(ctx, u, t, p)
scores = METHODS[strategy](comp, u, t, p)
order = rank_order(scores, p)[:k]
s = scores[order]
match = 100 * (s - scores.min()) / (scores.max() - scores.min() + 1e-12)
sources = dict(H.mixed_sources(comp, k)) if strategy == "H3 Mixte" else {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Date", str(pd.to_datetime(t, unit="D").date()))
c2.metric("Offres actives", len(p))
c3.metric("Historique du candidat", f"{comp['n_past']} offre(s)", "nouveau" if comp["n_past"] == 0 else "connu",
          delta_color="off")
if true_items:
    hits = int(np.isin(p[order], true_items).sum())
    c4.metric("Offres réellement obtenues dans la liste", f"{hits} / {len(true_items)}")

with st.expander("Profil du candidat"):
    st.write("**Compétences :**", ", ".join(ctx.prof.tokens("c", "sk", u)[:25]) or "—")
    st.write("**Domaines :**", ", ".join(ctx.prof.tokens("c", "dom", u)) or "—")
    past = ctx.hist.items_of(u, t)
    if past:
        st.write("**Offres passées :**", " · ".join(job_title(j) for j in past))

st.subheader(f"Top {k} — {strategy}")
for rank, (idx, m) in enumerate(zip(order, match), 1):
    e = explain(ctx, comp, p, idx, u, t)
    j = e["job"]
    r = jobs.iloc[j]
    flag = " ✓ offre réellement obtenue" if j in true_items else ""
    with st.container(border=True):
        a, b = st.columns([4, 1])
        a.markdown(f"**#{rank} · {job_title(j)}**{flag}  \n"
                   f"<small>id {r.job_id} · contrat : {r.get('contract_type', '—')} · dépt : {ctx.prof.j_zip[j] or '—'}"
                   f" · télétravail : {'oui' if ctx.prof.j_remote[j] == 1 else 'non'} · publiée le {r.create_date}</small>",
                   unsafe_allow_html=True)
        b.metric("Correspondance", f"{m:.0f} %")
        if sources:
            a.caption(f"Liste d'origine (mixte) : {SOURCE_LABEL[sources[idx]]}")
        st.markdown("**Pourquoi ?**  \n" + "  \n".join(f"✅ {x}" for x in e["reasons"]) if e["reasons"] else "")
        if e["warnings"]:
            st.markdown("  \n".join(f"⚠️ {x}" for x in e["warnings"]))

with st.expander("Résultats globaux sur le test (calculés dans le notebook)"):
    st.dataframe(cfg["summary"][["P@10", "R@10", "NDCG@10", "HR@10", "MRR", "Couverture", "Diversité", "Nouveauté"]]
                 .astype(float).round(3))
