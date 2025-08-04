import streamlit as st
import sqlite3
import pandas as pd
from transformers.pipelines import pipeline

# --- Initialisation de la DB ---
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_summary(original: str, summary: str):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO resumes (original, summary) VALUES (?, ?)",
        (original, summary)
    )
    conn.commit()
    conn.close()

init_db()

# --- SPLASH SCREEN ---
st.title("🧠 Assistant personnel")

# Crée le flag seulement une fois
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.write("Bienvenue ! Cliquez pour lancer l’assistant IA.")
    # → Si l’utilisateur clique, ce run-ci met started à True et continue
    if st.button("Lancer l’assistant IA"):
        st.session_state.started = True
    else:
        # → Si pas cliqué, on coupe tout ici. 
        #     (aucun onglet ni bouton Retour ne s’affichent)
        st.stop()

# À partir d’ici started == True

# Bouton Retour (redevient splash en remettant started à False)
if st.button("⬅️ Retour"):
    st.session_state.started = False
    # Pas de st.stop() : on laisse Streamlit relancer automatiquement
    # et afficher le splash au run suivant

# Onglets, uniquement quand started == True
if st.session_state.started:
    tab1, tab2 = st.tabs(["Résumé", "Historique"])
    
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
                resultat = summarizer(
                    texte, max_length=100, min_length=25, do_sample=False
                )
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
            titre = row["summary"][:50] + ("…" if len(row["summary"]) > 50 else "")
            with st.expander(f"{titre} — {row['timestamp']}"):
                st.write(row["summary"])
