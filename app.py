"""
app.py — Démonstration Streamlit du système de recommandation hybride d'offres d'emploi.

Lancement :  streamlit run app.py
Prérequis  :  data/{candidates,jobs,history}.csv (téléchargés automatiquement depuis Zenodo)
              et models.joblib (produit par le notebook).

Deux onglets :
  * « Recommandations » — deux modes :
      - « Candidat du jeu de test » : on se place à la date de sa requête de test et on
        indique quelles offres il a réellement obtenues (✓) ;
      - « Nouveau profil » : on saisit un profil → démonstration du cold-start utilisateur.
    Chaque recommandation est accompagnée d'une explication (compétences, critères,
    signal collaboratif) et d'une analyse des compétences manquantes.
  * « Résultats » — les résultats exigés par l'énoncé, calculés dans le notebook :
    comparaison hybrides / moteurs seuls / popularité, métriques de précision et
    au-delà de la précision, intervalles de confiance, cold-start, interprétabilité.
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from jobrec import hybrids as H
from jobrec.data import (History, JobPool, build_queries, ensure_jth, load_jth,
                         temporal_split)
from jobrec.engines import Context
from jobrec.evaluation import make_pool_fn, rank_order
from jobrec.explain import explain
from jobrec.features import Profiles, add_candidate

st.set_page_config(page_title="Recommandation d'offres d'emploi", page_icon="💼", layout="wide")

SOURCE_LABEL = {"cb": "compétences (contenu)", "meta": "candidats similaires", "kb": "critères (connaissances)"}
BEST = "H6 Augmentation de features"
BASE_ENGINES = ["Popularité (baseline)", "Contenu", "Collaboratif", "Connaissances"]

DATA_DIR = Path(__file__).parent / "data"


@st.cache_resource(show_spinner="Téléchargement du dataset JTH depuis Zenodo (premier lancement, ~25 Mo)…")
def get_data_dir():
    return ensure_jth(DATA_DIR)


def as_ltr(model, augmented: bool) -> H.LTRHybrid:
    """models.joblib contient les pipelines scikit-learn ; on les ré-enveloppe dans LTRHybrid.
    (Accepte aussi l'ancien format, où l'objet LTRHybrid lui-même était sauvegardé.)"""
    if isinstance(model, H.LTRHybrid):
        return model
    wrapper = H.LTRHybrid(augmented=augmented)
    wrapper.model = model
    return wrapper


@st.cache_resource(show_spinner="Chargement des données et des modèles…")
def load_all():
    D = load_jth(get_data_dir())
    cfg = joblib.load(Path(__file__).parent / "models.joblib")
    cfg["ltr_raw"] = as_ltr(cfg["ltr_raw"], augmented=False)
    cfg["ltr_aug"] = as_ltr(cfg["ltr_aug"], augmented=True)
    train, test = temporal_split(D.interactions, 0.8)
    hist = History(D.interactions, len(D.candidates), len(D.jobs))
    prof = Profiles(D.candidates, D.jobs, fit_cutoff_day=cfg["train_end"])
    ctx = Context(prof, hist, pop_window=30, profile_beta=0.5, knn_k=50)
    pool = JobPool(D.jobs.create_day.values.astype(int), cfg["window"])
    return D, cfg, ctx, pool, build_queries(test)


D, cfg, ctx, pool, q_test = load_all()
jobs = D.jobs
RES = cfg.get("results")  # absent si models.joblib vient d'une ancienne version du notebook


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


def fmt(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Arrondit les colonnes numériques pour l'affichage."""
    return df.apply(lambda c: c.astype(float).round(digits) if pd.api.types.is_numeric_dtype(c) else c)


def show_table(df: pd.DataFrame, digits: int = 3):
    """Tableau affiché en entier (toutes les lignes visibles, sans défilement)."""
    st.dataframe(fmt(df, digits), width="stretch", height=35 * (len(df) + 1) + 3)


def nb(n) -> str:
    """Entier avec espace comme séparateur de milliers (8216 -> « 8 216 »)."""
    return f"{int(n):,}".replace(",", " ")


FEATURE_LABELS = {  # noms lisibles des caractéristiques du modèle H5 / H6
    "jac_skills": "Compétences communes (Jaccard)", "cov_skills": "Part des compétences requises possédées",
    "jac_domain": "Domaines communs (Jaccard)", "jac_soft": "Soft skills communes (Jaccard)",
    "c_diplome": "Contrainte : diplôme", "c_experience": "Contrainte : expérience",
    "c_seniorite": "Contrainte : séniorité", "c_localisation": "Contrainte : localisation",
    "c_contrat": "Contrainte : contrat", "c_langues": "Contrainte : langues",
    "kb_inconnues": "Nombre d'informations manquantes", "log_pop_recente": "Popularité récente (30 j)",
    "log_pop_totale": "Popularité totale", "age_offre": "Âge de l'offre",
    "log_hist_candidat": "Taille de l'historique du candidat", "score_contenu": "Score du moteur contenu",
    "score_cf": "Score du moteur collaboratif", "score_meta": "Score du méta-niveau",
    "score_kb": "Score du moteur connaissances",
}


# ---------------------------------------------------------------------------
st.title("💼 Recommandation hybride d'offres d'emploi")
st.caption("Dataset JTH (Romero et al., RecSys'26, CC BY-NC 4.0). Le score affiché est un score de "
           "correspondance relatif, pas une probabilité d'embauche.")

tab_reco, tab_res = st.tabs(["🔎 Recommandations", "📊 Résultats"])

# ===========================================================================
# Onglet 1 — Recommandations expliquées
# ===========================================================================
with st.sidebar:
    st.header("Recommandations")
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

with tab_reco:
    pool_fn = make_pool_fn(ctx, pool)
    p = pool_fn(u, t)
    if len(p) == 0:
        # pas de st.stop() ici : il arrêterait aussi l'onglet Résultats
        st.warning("Aucune offre active à cette date.")
    else:
        comp = H.components(ctx, u, t, p)
        scores = METHODS[strategy](comp, u, t, p)
        order = rank_order(scores, p)[:k]
        s = scores[order]
        match = 100 * (s - scores.min()) / (scores.max() - scores.min() + 1e-12)
        sources = dict(H.mixed_sources(comp, k)) if strategy == "H3 Mixte" else {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Date", str(pd.to_datetime(t, unit="D").date()))
        c2.metric("Offres actives", len(p))
        c3.metric("Historique du candidat", f"{comp['n_past']} offre(s)",
                  "nouveau" if comp["n_past"] == 0 else "connu", delta_color="off")
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
                           f"<small>id {r.job_id} · contrat : {r.get('contract_type', '—')} · "
                           f"dépt : {ctx.prof.j_zip[j] or '—'} · "
                           f"télétravail : {'oui' if ctx.prof.j_remote[j] == 1 else 'non'} · "
                           f"publiée le {r.create_date}</small>", unsafe_allow_html=True)
                b.metric("Correspondance", f"{m:.0f} %")
                if sources:
                    a.caption(f"Liste d'origine (mixte) : {SOURCE_LABEL[sources[idx]]}")
                if e["reasons"]:
                    st.markdown("**Pourquoi ?**  \n" + "  \n".join(f"✅ {x}" for x in e["reasons"]))
                if e["warnings"]:
                    st.markdown("  \n".join(f"⚠️ {x}" for x in e["warnings"]))

# ===========================================================================
# Onglet 2 — Résultats de l'évaluation (calculés dans le notebook)
# ===========================================================================
with tab_res:
    summary = (RES or {}).get("summary", cfg["summary"]).astype(float)
    nd = summary["NDCG@10"]
    best_base = nd[BASE_ENGINES].idxmax()
    pop = nd["Popularité (baseline)"]

    st.subheader("Le meilleur hybride")
    st.write(f"Sur **{nb(summary['n_requêtes'].iloc[0])}** requêtes de test (les 20 % d'interactions les plus "
             f"récentes), **{BEST}** obtient le meilleur NDCG@10.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"NDCG@10 — {BEST.split()[0]}", f"{nd[BEST]:.3f}")
    k2.metric(f"Meilleur moteur seul ({best_base})", f"{nd[best_base]:.3f}",
              f"{100 * (nd[BEST] / nd[best_base] - 1):+.0f} % pour H6", delta_color="off")
    k3.metric("Popularité (baseline)", f"{pop:.3f}", f"× {nd[BEST] / pop:.1f} pour H6", delta_color="off")
    k4.metric(f"HR@10 — {BEST.split()[0]}", f"{summary.loc[BEST, 'HR@10']:.3f}",
              "offre réelle dans le top 10", delta_color="off")

    # --- Comparaison principale, avec intervalles de confiance ---------------
    st.subheader("1. Comparaison : hybrides, moteurs seuls et popularité")
    if RES is not None and RES.get("ci") is not None:
        ci = RES["ci"].copy()
        ci.columns = ["ndcg", "bas", "haut"]
        ci = ci.reset_index(names="Méthode")
        ci["Type"] = np.where(ci["Méthode"].isin(BASE_ENGINES), "Moteur seul", "Hybride")
        ci.loc[ci["Méthode"] == BEST, "Type"] = "Meilleur hybride"
        order_m = ci.sort_values("ndcg", ascending=False)["Méthode"].tolist()
        colors = alt.Scale(domain=["Meilleur hybride", "Hybride", "Moteur seul"],
                           range=["#2f6fd1", "#9aa5b1", "#d5dae0"])
        bars = alt.Chart(ci).mark_bar().encode(
            x=alt.X("ndcg:Q", title="NDCG@10 (test)", scale=alt.Scale(domain=[0, 0.6])),
            y=alt.Y("Méthode:N", sort=order_m, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("Type:N", scale=colors, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Méthode", alt.Tooltip("ndcg:Q", format=".3f", title="NDCG@10"),
                     alt.Tooltip("bas:Q", format=".3f", title="IC bas"),
                     alt.Tooltip("haut:Q", format=".3f", title="IC haut")])
        errs = alt.Chart(ci).mark_rule(color="black").encode(
            x="bas:Q", x2="haut:Q", y=alt.Y("Méthode:N", sort=order_m))
        val_labels = alt.Chart(ci).mark_text(align="left", dx=18, fontSize=11).encode(
            x="ndcg:Q", y=alt.Y("Méthode:N", sort=order_m), text=alt.Text("ndcg:Q", format=".3f"))
        rule_pop = alt.Chart(pd.DataFrame({"x": [pop]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x:Q")
        st.altair_chart((bars + errs + val_labels + rule_pop).properties(height=380), width="stretch")
        st.caption("Barre = NDCG@10 moyen · trait noir = intervalle de confiance à 95 % (bootstrap, 300 tirages) · "
                   "pointillé = popularité. Des intervalles qui ne se chevauchent pas indiquent un écart réel.")

    acc_cols = ["P@5", "R@5", "NDCG@5", "HR@5", "P@10", "R@10", "NDCG@10", "HR@10", "MRR"]
    beyond_cols = ["Couverture", "Diversité", "Nouveauté"]
    st.markdown("**Métriques de précision**")
    show_table(summary[acc_cols])
    st.markdown("**Métriques au-delà de la précision**")
    show_table(summary[beyond_cols])
    st.caption("La meilleure valeur de chaque colonne n'est pas toujours celle de H6 : la popularité récente "
               "améliore la précision au prix d'un peu de nouveauté et de couverture ; H3 (mixte) est le plus divers.")

    # --- Cold-start ----------------------------------------------------------
    st.subheader("2. Cold-start : nouveaux candidats et nouvelles offres")
    if RES is not None and RES.get("cold") is not None:
        cs = pd.DataFrame({
            "NDCG@10 nouveaux": RES["cold"]["NDCG@10"],
            "NDCG@10 connus": RES["warm"]["NDCG@10"],
            "HR@10 offres neuves": summary["HR@10 offres nouvelles"],
        })
        if RES.get("hidden") is not None:
            cs["Perte sans historique (%)"] = RES["hidden"]["perte (%)"].round(0)
        n_cold = int(RES["cold"]["n_requêtes"].iloc[0])
        n_warm = int(RES["warm"]["n_requêtes"].iloc[0])
        st.write(f"{nb(n_cold)} requêtes viennent de candidats **sans historique**, {nb(n_warm)} de candidats "
                 f"connus.")
        show_table(cs)
        st.caption("Le collaboratif s'effondre sans historique et ne propose presque jamais d'offre neuve ; "
                   "la popularité ne peut jamais proposer une offre sans candidature. Les hybrides qui utilisent "
                   "le contenu restent robustes. « Perte » : on rejoue les candidats connus en masquant leur historique.")
    else:
        st.info("Relancez le notebook pour enregistrer les résultats cold-start dans models.joblib.")

    # --- Interprétabilité ----------------------------------------------------
    st.subheader("3. Ce que le meilleur modèle a appris")
    coef = (RES or {}).get("coef")
    if coef is None:
        coef = pd.DataFrame({"augmentation (H6)": pd.Series(cfg["ltr_aug"].coefficients())})
    c = coef["augmentation (H6)"].rename("coefficient").rename_axis("caractéristique").reset_index()
    c["caractéristique"] = c["caractéristique"].map(lambda f: FEATURE_LABELS.get(f, f))
    c = c.reindex(c["coefficient"].abs().sort_values(ascending=False).index).head(10)
    coef_chart = alt.Chart(c).mark_bar().encode(
        x=alt.X("coefficient:Q", title="Coefficient standardisé (H6)"),
        y=alt.Y("caractéristique:N", sort=c["caractéristique"].tolist(), title=None,
                axis=alt.Axis(labelLimit=300)),
        color=alt.condition("datum.coefficient > 0", alt.value("#2f9e5b"), alt.value("#d1495b")),
        tooltip=["caractéristique", alt.Tooltip("coefficient:Q", format=".2f")])
    st.altair_chart(coef_chart.properties(height=300), width="stretch")
    st.caption("Vert = augmente la chance d'être positionné, rouge = la diminue. Les caractéristiques étant "
               "standardisées, la longueur de la barre compare directement leur importance. "
               "Le signal le plus fort est temporel : une offre récente est bien plus probable.")
    st_ = (RES or {}).get("settings", {"W_BEST": cfg["W_BEST"], "SW": cfg["SW"], "TAU": cfg["TAU"],
                                       "window": cfg["window"]})
    w_txt = ", ".join(f"{k} {v:.1f}" for k, v in st_["W_BEST"].items())
    st.write(f"**Réglages choisis sur la validation :** poids H1 = {w_txt} · commutation H2 = "
             f"CF dès {st_['SW']['n_min']} offre passée, sinon contenu · seuil de la cascade H4 τ = {st_['TAU']} · "
             f"fenêtre des offres actives W = {st_['window']} jours.")

    # --- Données et protocole ------------------------------------------------
    with st.expander("Données et protocole d'évaluation"):
        if RES is not None and RES.get("stats") is not None:
            stats = RES["stats"].rename("valeur").to_frame()
            stats["valeur"] = stats["valeur"].map(lambda v: f"{v:,.4g}".replace(",", " ") if isinstance(v, (int, float, np.number)) else v)
            show_table(stats)
        if RES is not None and RES.get("split") is not None:
            show_table(RES["split"], 1)
        st.markdown(
            "- **Découpage temporel** : test = 20 % d'interactions les plus récentes ; validation = 10 % les plus "
            "récents de l'entraînement, pour le réglage uniquement.\n"
            "- À la date *t*, les moteurs ne voient que les interactions **antérieures à t** ; seules les offres "
            "encore actives sont recommandées.\n"
            "- Une offre est **pertinente** si le candidat y a été positionné (retour implicite).")

    with st.expander("Que mesure chaque métrique ?"):
        st.markdown(
            "- **P@K** : part des K offres affichées qui sont correctes.\n"
            "- **R@K** : part des offres réellement obtenues que l'on retrouve dans le top K.\n"
            "- **HR@K** : 1 si au moins une offre correcte est dans le top K (moyenne sur les requêtes).\n"
            "- **MRR** : 1 / rang de la première offre correcte.\n"
            "- **NDCG@K** : récompense une offre correcte d'autant plus qu'elle est haut placée (0 à 1).\n"
            "- **Couverture** : part des offres actives recommandées au moins une fois.\n"
            "- **Diversité** : 1 − similarité moyenne entre les offres d'une même liste.\n"
            "- **Nouveauté** : moyenne de −log₂(popularité) des offres recommandées, en bits.")
