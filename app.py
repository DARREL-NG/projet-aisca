# app.py — AISCA (version PRO, design: moderne & dynamique)
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from semantic_engine import analyser_profil, top_3_metiers
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Config + Load keys
# -------------------------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.warning("Clé OpenAI non trouvée. Ajoute .env avec OPENAI_API_KEY pour activer l'IA générative.")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

st.set_page_config(page_title="AISCA — Analyse de compétences", layout="wide", page_icon="🤖")

# -------------------------
# Style & Header
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f7fbff 0%, #ffffff 100%); }
    .header { font-size:26px; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("🧠 AISCA — Cartographie des compétences & recommandation de métiers")
st.write("Interface moderne — remplis les 3 champs ci‑dessous, clique sur **Analyser mon profil** puis génère un plan ou une mini‑bio.")

# -------------------------
# Sidebar (navigation + options)
# -------------------------
with st.sidebar:
    st.title("🔧 Paramètres")
    st.write("Design : Modern & Dynamic")
    st.markdown("---")
    st.caption("Prochaines étapes : graphiques, IA, cache, sauvegarde.")
    show_raw = st.checkbox("Afficher les tables brutes", value=False)

# -------------------------
# Formulaire principal
# -------------------------
with st.form("form_profile", clear_on_submit=False):
    c1, c2 = st.columns([2,1])
    with c1:
        rep1 = st.text_area("📝 1) Décris les tâches que tu réalises le plus souvent", value="", height=120)
        rep2 = st.text_area("🛠️ 2) Outils / technologies (ex: Python, SQL, Power BI...)", value="", height=80)
    with c2:
        rep3 = st.text_area("⭐ 3) Type de missions préférées", value="", height=200)
        st.markdown("💡 Astuce : sois naturel — le moteur sémantique comprend le texte libre.")
    submitted = st.form_submit_button("🚀 Analyser mon profil")

# -------------------------
# Analyse (exécutée après submit)
# -------------------------
if submitted:
    # validation minimale
    if not (rep1.strip() or rep2.strip() or rep3.strip()):
        st.error("Merci d'entrer au moins une information dans un des champs.")
    else:
        with st.spinner("Analyse en cours — chargement modèle SBERT si nécessaire..."):
            reponses = [rep1, rep2, rep3]
            blocs, metiers = analyser_profil(reponses)
            top3 = top_3_metiers(metiers)

        # --- résultats résumé + top3
        st.success("Analyse terminée ✅")
        st.markdown("### 🔝 Top 3 métiers recommandés")
        st.table(top3.reset_index(drop=True))

        # layout 2 colonnes : blocs + graphiques
        left, right = st.columns([2,3])

        with left:
            st.markdown("### 🧱 Blocs de compétences (scores normalisés)")
            st.dataframe(blocs.reset_index(drop=True), use_container_width=True if hasattr(st, "use_container_width") else None)

            if show_raw:
                st.markdown("#### 🗂 Toutes les compétences (brutes)")
                st.dataframe(blocs, height=200)

        with right:
            # radar (top 5 blocs)
            st.markdown("### 🧭 Radar — Top blocs")
            top_bl = blocs.head(5)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top_bl["score_normalise"].tolist(),
                theta=top_bl["bloc"].tolist(),
                fill='toself',
                name='Blocs'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=380)
            st.plotly_chart(fig_radar, width="stretch")

            # bar chart métiers
            st.markdown("### 📊 Scores par métier")
            fig_m = px.bar(metiers, x="metier", y="score_normalise", color="score_normalise",
                        color_continuous_scale="Blues", title="Métiers — score normalisé")
            st.plotly_chart(fig_m, width="stretch")

        st.markdown("---")
        st.markdown("### 📚 Vue complète des métiers")
        st.dataframe(metiers, use_container_width=True if hasattr(st, "use_container_width") else None)

        # -------------------------
        # Génération IA (si clé présente)
        # -------------------------
        st.markdown("---")
        st.subheader("🤖 Génération IA — plan & mini‑bio")

        if client is None:
            st.info("Ajoute ton OPENAI_API_KEY dans le fichier .env pour activer cette fonctionnalité.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📚 Générer un plan de progression"):
                    prompt = f"""Tu es un expert en évolution professionnelle.
Compétences et blocs :
{blocs.to_string(index=False)}
Métiers recommandés :
{top3.to_string(index=False)}
Génère un plan structuré sur 30 jours :
- 5 étapes claires avec objectifs quotidiens/hebdo
- outils à maîtriser
- ressources (courses, docs)"""
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "Tu es un expert carrière."},
                                {"role": "user", "content": prompt}
                            ],
                        )
                        content = resp.choices[0].message["content"]
                        st.markdown("#### Plan de progression (IA)")
                        st.write(content)
                    except Exception as e:
                        st.error(f"Erreur API OpenAI : {e}")

            with col_b:
                if st.button("🧑‍💼 Générer une mini‑bio"):
                    bio_prompt = f"""Rédige une mini‑bio professionnelle (max 5 lignes) basée sur ces compétences :
{blocs.to_string(index=False)}"""
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": bio_prompt}]
                        )
                        bio = resp.choices[0].message["content"]
                        st.markdown("#### Mini‑bio")
                        st.info(bio)
                    except Exception as e:
                        st.error(f"Erreur API OpenAI : {e}")

# Footer — petite méthodo
st.markdown("---")
st.caption("AISCA — prototype IA sémantique. SBERT pour embeddings, OpenAI pour génération (si clé fournie).")

