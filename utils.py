import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter


def load_data():
    """
    Charge les fichiers Parquet des features et de l'annuaire.
    """
    df_colleges = pd.read_parquet("data_streamlit/features_colleges.parquet")
    df_lycees = pd.read_parquet("data_streamlit/features_lycees.parquet")
    annuaire = pd.read_parquet("data_streamlit/annuaire.parquet")
    return df_colleges, df_lycees, annuaire


class KNNModel:
    """
    Wrapper sur sklearn.NearestNeighbors pour retrouver les établissements similaires.
    """

    def __init__(self, n_neighbors=6, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.uais = None
        self.df = None

    def fit(self, df_features):
        """
        Entraîne le modèle sur le DataFrame fourni.
        """
        self.df = df_features.copy()
        self.uais = df_features.index
        self.model.fit(self.df)

    def get_similars(self, uai, n=5):
        """
        Renvoie les n établissements les plus proches de l’UAI donné.
        """
        idx = list(self.uais).index(uai)
        distances, indices = self.model.kneighbors(
            [self.df.iloc[idx]], n_neighbors=n + 1
        )
        neighbors = self.uais[indices[0]].tolist()
        distances = distances[0].tolist()
        return [(u, d) for u, d in zip(neighbors, distances) if u != uai][:n]


def apply_weights(df, weights):
    """
    Applique les poids aux colonnes du DataFrame.
    Gère aussi les colonnes encodées pour nature (lycées).
    """
    df_copy = df.copy()
    for col, w in weights.items():
        if col == "nature":
            for c in df.columns:
                if c.startswith("nature_"):
                    df_copy[c] *= w
        elif col in df.columns:
            df_copy[col] *= w
    return df_copy


def get_model(df, config):
    """
    Instancie un modèle KNN ou KMeans selon la config.
    Applique les poids avant l’entraînement.
    """
    df_prepared = apply_weights(df, config["weights"])

    if config.get("algo") == "KMeans":
        km = KMeans(n_clusters=30, random_state=42).fit(df_prepared)
        centers = pd.DataFrame(km.cluster_centers_)
        model = NearestNeighbors(n_neighbors=6).fit(centers)
        return KNNWrapper(df_prepared, km.labels_, centers, model)

    model = KNNModel(n_neighbors=6, metric=config["metric"])
    model.fit(df_prepared)
    return model


class KNNWrapper:
    """
    Permet de faire de la recherche de voisins après clustering par KMeans.
    """

    def __init__(self, df, labels, centers, model):
        self.df = df
        self.labels = labels
        self.centers = centers
        self.model = model
        self.uais = df.index

    def get_similars(self, uai, n=5):
        """
        Renvoie les n établissements du même cluster les plus proches de l’UAI donné.
        """
        idx = list(self.uais).index(uai)
        label = self.labels[idx]
        cluster_indices = [
            i for i, l in enumerate(self.labels) if l == label and self.uais[i] != uai
        ]
        distances = (
            ((self.df.iloc[cluster_indices] - self.df.iloc[idx]) ** 2)
            .sum(axis=1)
            .pow(0.5)
        )
        sorted_neighbors = distances.sort_values().head(n)
        return list(zip(sorted_neighbors.index, sorted_neighbors.values))


def get_table_voisins(model, uai, annuaire, n=5):
    """
    Récupère les informations des n voisins depuis l’annuaire.
    """
    voisins = model.get_similars(uai, n)
    table = annuaire[annuaire["uai"].isin([uai] + [v[0] for v in voisins])].copy()
    table.insert(0, "distance", [0.0] + [v[1] for v in voisins])
    table = table.sort_values(by="distance")
    return table


def compute_uai_status(results, selected_uai):
    """
    Détermine pour chaque UAI s’il est proposé par 1, plusieurs ou tous les modèles.
    """
    all_uais = sum(
        [[uai for uai in df["uai"] if uai != selected_uai] for df in results], []
    )
    counts = Counter(all_uais)
    status_map = {}
    for uai in set(all_uais):
        if counts[uai] == len(results):
            status_map[uai] = "commun_tous"
        elif counts[uai] >= 2:
            status_map[uai] = "commun_2_plus"
        else:
            status_map[uai] = "un_seul_modele"
    return status_map


def highlight_uais(row, status_map):
    """
    Applique une couleur selon le statut de l’UAI (tous, plusieurs, un seul modèle).
    """
    uai = row.get("uai", None)
    n_cols = len(row)
    if uai in status_map:
        if status_map[uai] == "commun_tous":
            return ["background-color: #d8f5d4"] * n_cols
        elif status_map[uai] == "commun_2_plus":
            return ["background-color: #ffe8b3"] * n_cols
        elif status_map[uai] == "un_seul_modele":
            return ["background-color: #fffacc"] * n_cols
    return [""] * n_cols
