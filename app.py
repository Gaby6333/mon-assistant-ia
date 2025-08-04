import streamlit as st
import sqlite3
import pandas as pd
from transformers.pipelines import pipeline

# --- Init DB & fonctions ---
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

def init_tasks_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            due_date DATE,
            is_done INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

# --- Fonctions de sauvegarde ---
def save_summary(original: str, summary: str):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO resumes (original, summary) VALUES (?, ?)",
        (original, summary)
    )
    conn.commit()
    conn.close()

def save_task(description: str, due_date):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO tasks (description, due_date) VALUES (?, ?)",
        (description, str(due_date))
    )
    conn.commit()
    conn.close()

def toggle_task(task_id: int):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute(
        "UPDATE tasks SET is_done = 1 - is_done WHERE id = ?",
        (task_id,)
    )
    conn.commit()
    conn.close()

# --- Initialisation des bases ---
init_db()
init_tasks_db()

# --- Interface Streamlit ---
st.title("🧠 Assistant personnel")

# Flag pour splash
if "started" not in st.session_state:
    st.session_state.started = False

# --- Splash screen ---
if not st.session_state.started:
    st.write("Bienvenue ! Cliquez pour lancer l’assistant IA.")
    if st.button("Lancer l’assistant IA", key="start"):
        st.session_state.started = True

# --- UI principale après lancement ---
else:
    # Bouton Retour
    if st.button("⬅️ Retour", key="back"):
        st.session_state.started = False

    # Onglets
    tab1, tab2, tab3 = st.tabs(["Résumé", "Historique", "Tâches"])

    with tab1:
        st.header("Résumé de texte")
        texte = st.text_area("Colle ici ton texte à résumer 👇", key="input")
        if st.button("Résumer le texte", key="summarize"):
            with st.spinner("Chargement du modèle…"):
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
            "SELECT timestamp, summary FROM resumes ORDER BY timestamp DESC", conn
        )
        conn.close()
        for _, row in df.iterrows():
            titre = row["summary"][:50] + ("…" if len(row["summary"]) > 50 else "")
            with st.expander(f"{titre} — {row['timestamp']}"):
                st.write(row["summary"])

    with tab3:
        st.header("Tâches & rappels")
        st.subheader("Ajouter une nouvelle tâche")
        desc = st.text_input("Description de la tâche", key="task_desc")
        due = st.date_input("Date d'échéance", key="task_due")
        if st.button("Ajouter la tâche", key="add_task"):
            if desc:
                save_task(desc, due)
                st.success("Tâche ajoutée !")
            else:
                st.error("Veuillez entrer une description.")

        st.subheader("Liste des tâches")
        conn = sqlite3.connect("history.db")
        df_tasks = pd.read_sql_query(
            "SELECT id, description, due_date, is_done FROM tasks ORDER BY due_date", conn
        )
        conn.close()
        for _, row in df_tasks.iterrows():
            cols = st.columns([0.1, 0.6, 0.2, 0.1])
            checked = bool(row["is_done"])
            new_checked = cols[0].checkbox("", value=checked, key=f"task_{row['id']}")
            if new_checked != checked:
                toggle_task(row["id"])
                st.experimental_rerun()
            cols[1].write(row["description"])
            cols[2].write(row["due_date"])
            status = "✅" if new_checked else "❌"
            cols[3].write(status)
