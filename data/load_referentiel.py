import pandas as pd

# Charger le fichier CSV
df = pd.read_csv("data/referentiel_competences.csv")

# Affichage des premières lignes
print(df.head())

# Nombre de métiers
print("\nNombre de métiers :", df["metier"].nunique())

# Vérification : nombre de compétences par métier
verif = df.groupby("metier").count()
print("\nVérification des compétences par métier :")
print(verif)

