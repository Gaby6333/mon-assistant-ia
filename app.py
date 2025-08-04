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

# --- Interface Streamlit ---
st.title("Résumer un texte")

# Onglets pour la navigation juste sous le titre
tab1, tab2 = st.tabs(["Résumé", "Historique"])

with tab1:
    st.header("Créer un nouveau résumé")
    texte = st.text_area("Colle ici ton texte à résumer 👇")
    if st.button("Résumer le texte"):
        with st.spinner("Je réfléchis... 🤔"):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            resultat = summarizer(texte, max_length=100, min_length=25, do_sample=False)
            summary_text = resultat[0]['summary_text']
            st.success("Résumé généré !")
            st.write(summary_text)
            save_summary(texte, summary_text)

with tab2:
    st.header("Historique des résumés")
    # Récupération des données
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query(
        "SELECT timestamp, original, summary FROM resumes ORDER BY timestamp DESC", conn
    )
    conn.close()
    # Mise en forme du timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # Affichage sous forme d'expanders pour plus de lisibilité
    for idx, row in df.iterrows():
        with st.expander(f"Résumé du {row['timestamp']}", expanded=False):
            st.markdown(f"**Résumé :**\n{row['summary']}")
            st.markdown(f"**Texte original :**\n> {row['original'][:200]}{'...' if len(row['original'])>200 else ''}")
