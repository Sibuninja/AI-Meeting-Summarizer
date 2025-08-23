import os
import streamlit as st
from transcriber import transcribe_audio
from summarizer import generate_summary

# ensure uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🎤 AI Meeting Summarizer")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save uploaded file in uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  


    st.success(f"File saved: {file_path}")

    if st.button("Transcribe & Summarize"):
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(file_path)
            st.subheader("Transcript")
            st.write(transcript)

        with st.spinner("Summarizing..."):
            summary = generate_summary(transcript)
            st.subheader("Summary")
            st.write(summary)
