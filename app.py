import streamlit as st
from transformers.pipelines import pipeline

st.title("🧠 Résumeur de texte intelligent")

texte = st.text_area("Colle ici ton texte à résumer 👇")

if st.button("Résumer le texte"):
    with st.spinner("Je réfléchis... 🤔"):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        resultat = summarizer(texte, max_length=100, min_length=25, do_sample=False)
        st.success("Résumé :")
        st.write(resultat[0]['summary_text'])
