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

# Appel de l'initialisation de la base
init_db()

# --- Interface Streamlit ---
st.title("🧠 Résumeur de texte intelligent")

# Navigation dans la sidebar
en_cours = st.sidebar.radio(
    "Navigation", ["Résumé", "Historique"]
)

if en_cours == "Résumé":
    texte = st.text_area("Colle ici ton texte à résumer 👇")
    if st.button("Résumer le texte"):
        with st.spinner("Je réfléchis... 🤔"):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            resultat = summarizer(texte, max_length=100, min_length=25, do_sample=False)
            summary_text = resultat[0]['summary_text']
            st.success("Résumé :")
            st.write(summary_text)
            # Sauvegarde du résumé en base
            save_summary(texte, summary_text)

elif en_cours == "Historique":
    st.subheader("Historique des résumés")
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query(
        "SELECT id, timestamp, summary FROM resumes ORDER BY timestamp DESC",
        conn
    )
    conn.close()
    st.dataframe(df)
