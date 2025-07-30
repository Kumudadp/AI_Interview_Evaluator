import streamlit as st
from evaluator import evaluate

st.title("🎙️ AI Interview Evaluator")

uploaded_file = st.file_uploader("Upload your .wav audio file", type=["wav"])

if uploaded_file:
    transcript, clarity, confidence, final_score = evaluate(uploaded_file)

    st.subheader("Results")
    st.write("📝 Transcript:", transcript)
    st.write("🔊 Clarity Score:", clarity)
    st.write("💬 Confidence Score:", confidence)
    st.write("🏆 Final Score:", final_score)
