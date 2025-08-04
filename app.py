import streamlit as st
import sqlite3
import pandas as pd
from transformers.pipelines import pipeline

# --- Initialisation de la base de données SQLite ---
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

# Fonction pour sauvegarder un résumé
def save_summary(original: str, summary: str):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO resumes (original, summary) VALUES (?, ?)",
        (original, summary)
    )
    conn.commit()
    conn.close()

# Initialisation de la DB
init_db()

# Chargement du pipeline de résumé (forcé CPU)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

# --- Splash screen + bouton de lancement ---
st.title("🧠 Assistant personnel")

# 1) Initialise le flag
if "started" not in st.session_state:
    st.session_state.started = False

# 2) Tant que l'on n'a pas cliqué, on reste sur le splash
if not st.session_state.started:
    st.write("Bienvenue ! Cliquez pour lancer l’assistant IA.")
    # Si je clique : je passe started à True et je continue CE MÊME run
    if st.button("Lancer l’assistant IA"):
        st.session_state.started = True
    else:
        # Si je n'ai PAS cliqué, j'arrête tout ici
        st.stop()

# → À partir d'ici, session_state.started == True

# 3) Bouton Retour
if st.button("⬅️ Retour"):
    st.session_state.started = False

# → Dès ce point, session_state.started == True
#    on peut afficher les onglets Résumé / Historique
# … après le splash et le bouton Retour …

if st.session_state.started:
    # Création des onglets uniquement quand started == True
    tab1, tab2 = st.tabs(["Résumé", "Historique"])

    # Tout le code qui se réfère à tab1/tab2 doit être indenté ici !
    with tab1:
        st.header("Résumé de texte")
        texte = st.text_area("Colle ici ton texte à résumer 👇")
        if st.button("Résumer le texte"):
            with st.spinner("Chargement du modèle..."):
                summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1
                )
                resultat = summarizer(texte, max_length=100, min_length=25, do_sample=False)
                summary_text = resultat[0]["summary_text"]
                st.success("Résumé :")
                st.write(summary_text)
                save_summary(texte, summary_text)

    with tab2:
        st.header("Historique des résumés")
        conn = sqlite3.connect("history.db")
        df = pd.read_sql_query(
            "SELECT timestamp, summary FROM resumes ORDER BY timestamp DESC",
            conn
        )
        conn.close()
        for _, row in df.iterrows():
            title = row['summary'][:50] + ("..." if len(row['summary']) > 50 else "")
            with st.expander(f"{title} — {row['timestamp']}"):
                st.write(row['summary'])
