# semantic_engine.py — Moteur sémantique PRO
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache

# chemin du CSV (assure-toi que data/referentiel_competences.csv existe)
CSV_PATH = "data/referentiel_competences.csv"

# 1) Charger modèle SBERT en cache (performant)
@lru_cache(maxsize=1)
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# 2) Charger référentiel et pré-calculer embeddings
@lru_cache(maxsize=1)
def load_referentiel():
    df = pd.read_csv(CSV_PATH)
    # s'assurer colonnes attendues : ['metier','bloc','competence']
    if "competence" not in df.columns:
        raise ValueError(f"{CSV_PATH} doit contenir une colonne 'competence'.")
    model = load_model()
    # encodage (convert_to_tensor=True pour vitesse GPU/CPU)
    embeddings = model.encode(df["competence"].tolist(), convert_to_tensor=True)
    df = df.reset_index(drop=True)
    df["embedding"] = list(embeddings)
    return df

def normaliser_score(valeur):
    """Convertit [-1,1] -> [0,100]"""
    return round((valeur + 1) * 50, 2)

def analyser_profil(reponses_textuelles):
    """
    reponses_textuelles : liste de chaînes (3 champs typiquement)
    retourne : (scores_blocs_df, scores_metiers_df)
    """
    model = load_model()
    df = load_referentiel()

    texte = " ".join([r for r in reponses_textuelles if isinstance(r, str)])
    if not texte.strip():
        # si vide, renvoyer frames vides structurés
        return pd.DataFrame(columns=["metier", "bloc", "similarite", "score_normalise"]), \
                pd.DataFrame(columns=["metier", "similarite", "score_normalise"])

    emb_profile = model.encode(texte, convert_to_tensor=True)

    # stack embeddings en 2D tensor
    embeddings_comp = torch.stack(df["embedding"].tolist())

    # similarité cosinus
    sims = util.cos_sim(emb_profile, embeddings_comp)[0].cpu().tolist()

    temp = df.copy()
    temp["similarite"] = sims

    # score par bloc
    scores_blocs = temp.groupby(["metier", "bloc"])["similarite"].mean().reset_index()
    scores_blocs["score_normalise"] = scores_blocs["similarite"].apply(normaliser_score)
    scores_blocs = scores_blocs.sort_values("score_normalise", ascending=False)

    # score par métier
    scores_metiers = temp.groupby("metier")["similarite"].mean().reset_index()
    scores_metiers["score_normalise"] = scores_metiers["similarite"].apply(normaliser_score)
    scores_metiers = scores_metiers.sort_values("score_normalise", ascending=False)

    return scores_blocs, scores_metiers

def top_3_metiers(scores_metiers_df):
    return scores_metiers_df.head(3).reset_index(drop=True)

# petit test local (décommenter pour debug)
# if __name__ == "__main__":
#     b,m = analyser_profil(["Python, SQL", "Power BI", "analyse de données"])
#     print(m.head())
