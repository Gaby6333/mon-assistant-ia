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

# --- Assure-toi d'avoir les pipelines chargés ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# --- Interface Streamlit ---
st.title("🧠 Assistant IA personnel")

tab1, tab2, tab3 = st.tabs(["Résumé", "Historique", "Assistant Q&A"])

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
    # Affiche chaque résumé dans un expander avec titre extrait
    for _, row in df.iterrows():
        title = row['summary'][:50] + ("..." if len(row['summary'])>50 else "")
        with st.expander(f"{title} 🤖 — {row['timestamp']}"):
            st.write(row['summary'])

with tab3:
    st.header("Assistant Q&A")
    st.write("Posez une question en fournissant un contexte ci-dessous :")
    context = st.text_area("Contexte (ex. : ton cours, tes notes)", height=200)
    question = st.text_input("Ta question 👇")
    if st.button("Répondre à la question"):
        if context and question:
            with st.spinner("Recherche de la réponse... 🕵️‍♂️"):
                answer = qa_pipeline(question=question, context=context)
                st.success("Réponse :")
                st.write(f"**{answer['answer']}** (score: {answer['score']:.2f})")
        else:
            st.error("Veuillez fournir à la fois un contexte et une question.")
