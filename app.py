import streamlit as st
import pandas as pd
from utils import (
    load_data,
    get_model,
    get_table_voisins,
    compute_uai_status,
    highlight_uais,
)
from PIL import Image
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(layout="wide")

df_features_colleges, df_features_lycees, annuaire = load_data()

col1, col2 = st.columns([1, 3])
with col1:
    st.image("img/MENESR.png", width=250)
with col2:
    st.title("TBNum - UAIs voisins")
    st.header("Comparaison des modèles")

with st.expander("ℹ️ À propos de cette application"):
    st.markdown(
        """
Cette application permet de **comparer plusieurs algorithmes** pour identifier les établissements scolaires **les plus similaires** à un établissement donné (UAI en entrée). Un UAI de collège en entrée propose des UAI de collèges en sortie et un UAI de lycée propose des UAI de lycées (quelles que soient leurs natures) en sortie.

- `total_eleves` : table `comptage_eleves` (staging)
- `ips` : table `ips` (staging)
- `taux_reussite` : table `reussite_examens` (staging)
- `eloignement` :  API open data
- `rep` : table `extended` (staging, uniquement pour les collèges)
- `tne` : table `extended` (staging)
- `libelle_nature` : table `extended` (staging, uniquement pour les lycées)

Le menu de gauche permet de sélectionner un établissement et de configurer plusieurs modèles à comparer.
                
Chaque modèle proposera *n* voisins qui seront affichés dans un tableau classé par similarité.
Les UAIs proposés par **tous les modèles** sont affichés en `vert`.
"""
    )

st.markdown("---")

# --- Barre latérale ---
st.sidebar.header("Configuration")

type_etab = st.sidebar.radio("Type d'établissement", ["Collège", "Lycée"])
is_college = type_etab == "Collège"

uai_list = (
    annuaire.query("nature == 'COLLEGE'" if is_college else "nature != 'COLLEGE'")[
        "uai"
    ]
    .unique()
    .tolist()
)
selected_uai = st.sidebar.selectbox("Sélectionner un établissement (UAI)", uai_list)

n_voisins = st.sidebar.slider("Nombre de voisins", min_value=1, max_value=10, value=5)
n_modeles = st.sidebar.slider(
    "Nombre de modèles à comparer", min_value=1, max_value=5, value=3
)

model_configs = []
st.sidebar.markdown("---")

for i in range(n_modeles):
    st.sidebar.markdown(f"### Modèle {i+1}")
    algo = st.sidebar.selectbox(f"Algo {i+1}", ["KNN", "KMeans"], key=f"algo_{i}")

    weights = {}
    col1, col2 = st.sidebar.columns(2)
    with col1:
        for feat in ["ips", "taux_reussite", "total_eleves"]:
            weights[feat] = st.number_input(
                f"Poids {feat} (modèle {i+1})",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"w1_{feat}_{i}",
            )
    with col2:
        for feat in ["eloignement", "tne"]:
            weights[feat] = st.number_input(
                f"Poids {feat} (modèle {i+1})",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"w2_{feat}_{i}",
            )
        if is_college:
            weights["rep"] = st.number_input(
                f"Poids rep (modèle {i+1})",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"rep_{i}",
            )
        if not is_college:
            weights["nature"] = st.number_input(
                f"Poids nature (modèle {i+1})",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f"nature_{i}",
            )

    model_configs.append(
        {
            "algo": algo,
            "metric": "euclidean",
            "reduction": "aucun",
            "weights": weights,
        }
    )
    st.sidebar.markdown("---")

# --- Analyse ---
if st.sidebar.button("Lancer l'analyse"):

    df_features = df_features_colleges if is_college else df_features_lycees
    results = []

    for i, config in enumerate(model_configs):
        model = get_model(df_features, config)
        df_result = get_table_voisins(model, selected_uai, annuaire, n=n_voisins)
        results.append(df_result)

    sets_uai = [set(df["uai"]) - {selected_uai} for df in results]
    uai_communs = set.intersection(*sets_uai) if results else set()

    for i, df_result in enumerate(results):
        df_result["distance"] = df_result["distance"].map(lambda x: f"{x:.2f}")
        df_result["total_eleves"] = df_result["total_eleves"].astype(int).astype(str)
        df_result["ips"] = df_result["ips"].map(lambda x: f"{x:.2f}")
        df_result["taux_reussite"] = df_result["taux_reussite"].map(
            lambda x: f"{x:.2f}"
        )
        df_result["eloignement"] = df_result["eloignement"].map(lambda x: f"{x:.2f}")
        st.markdown(f"### Modèle {i + 1}")
        status_map = compute_uai_status(results, selected_uai)
        st.dataframe(
            df_result.style.apply(lambda row: highlight_uais(row, status_map), axis=1),
            use_container_width=True,
        )

    # Marquage des UAI pour la PCA
    df_features_full = df_features.copy()
    df_features_full["uai"] = df_features_full.index
    df_features_full["type"] = "Autre"

    status_map = compute_uai_status(results, selected_uai)
    for uai, status in status_map.items():
        if status == "commun_tous":
            df_features_full.loc[uai, "type"] = "Commun à tous"
        elif status == "commun_2_plus":
            df_features_full.loc[uai, "type"] = "Commun à ≥2 modèles"
        elif status == "un_seul_modele":
            df_features_full.loc[uai, "type"] = "1 seul modèle"
    df_features_full.loc[selected_uai, "type"] = "Sélectionné"

    # PCA 2D
    X = df_features_full.drop(columns=["uai", "type"])
    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2).fit_transform(X_scaled)

    df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df_features_full.index)
    df_plot["uai"] = df_features_full["uai"]
    df_plot["type"] = df_features_full["type"]

    # Enrichir avec toutes les variables utiles
    df_plot = df_plot.reset_index(drop=True)
    df_plot = df_plot.merge(annuaire, on="uai", how="left")

    hover_cols = [
        "uai",
        "nom",
        "region",
        "dpt",
        "commune",
        "nature",
        "rep",
        "tne",
        "total_eleves",
        "ips",
        "taux_reussite",
        "eloignement",
    ]

    # Affichage de la projection PCA
    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="type",
        hover_data=hover_cols,
        color_discrete_map={
            "Sélectionné": "red",
            "Commun à tous": "green",
            "Commun à ≥2 modèles": "orange",
            "1 seul modèle": "yellow",
            "Autre": "lightgrey",
        },
        title="Projection PCA 2D",
        height=600,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.5))
    st.plotly_chart(fig, use_container_width=True)
