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


# Init DB
init_db()

# Chargement du pipeline de résumé (force CPU)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

# --- Écran d'accueil avec bouton pour démarrer ---
st.title("🧠 Résumeur de texte intelligent")
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.write("Bienvenue ! Cliquez ci-dessous pour accéder à l'assistant IA.")
    if st.button("Lancer l'assistant IA"):
        st.session_state.started = True
        st.experimental_rerun()
    st.stop()

# --- Interface principale après démarrage ---
tab1, tab2 = st.tabs(["Résumé", "Historique"])

with tab1:
    st.header("Résumé de texte")
    texte = st.text_area("Colle ici ton texte à résumer 👇")
    if st.button("Résumer le texte"):
        with st.spinner("Je réfléchis... 🤔"):
            resultat = summarizer(texte, max_length=100, min_length=25, do_sample=False)
            summary_text = resultat[0]['summary_text']
            st.success("Résumé :")
            st.write(summary_text)
            save_summary(texte, summary_text)

with tab2:
    st.header("Historique des résumés")
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query(
        "SELECT timestamp, summary FROM resumes ORDER BY timestamp DESC", conn
    )
    conn.close()
    for _, row in df.iterrows():
        title = row['summary'][:50] + ("..." if len(row['summary']) > 50 else "")
        with st.expander(f"{title} — {row['timestamp']}"):
            st.write(row['summary'])
