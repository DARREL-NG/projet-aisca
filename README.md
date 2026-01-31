# AISCA — Cartographie des compétences & recommandation de métiers

## Résumé
AISCA est un prototype d'orientation professionnelle qui combine :
- un moteur sémantique (SBERT) pour comparer un profil utilisateur à un référentiel métiers,
- une couche de génération (OpenAI) pour produire un plan de progression et une mini‑bio.

## Structure du projet
- `app.py` — Interface Streamlit (moderne & dynamique)
- `semantic_engine.py` — Moteur SBERT (encodage, scoring, top 3)
- `data/referentiel_competences.csv` — Référentiel métiers/blocs/compétences
- `requirements.txt` — dépendances

## Installation (Windows, Python 3.10 recommandé)
1. Installer Python 3.10 et utiliser `py -3.10`.
2. Dans le dossier du projet :
   ```bash
   py -3.10 -m pip install -r requirements.txt
