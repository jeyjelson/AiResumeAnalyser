import streamlit as st
import PyPDF2
import io
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer + ATS", page_icon="📄", layout="centered")

st.title("AI Resume Critiquer + ATS Scanner")
st.markdown("Upload your resume and get ATS score + AI feedback")

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role (e.g., Backend Developer, Data Analyst)")

analyse = st.button("Analyse Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

def basic_ats_checks(text):
    sections = ["experience", "skills", "education", "projects"]
    found = {sec: sec in text.lower() for sec in sections}
    return found

if analyse and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("File has no content")
            st.stop()

        ats_sections = basic_ats_checks(file_content)

        ats_prompt = f"""
You are an ATS resume scanner.

Compare this resume against the job role: {job_role}

Tasks:
1. Give an ATS compatibility score from 0–100
2. List missing important keywords
3. List strong matched keywords
4. Mention formatting or structure issues
5. Short hiring recommendation

Resume:
{file_content}

Return in clean bullet format.
"""

        feedback_prompt = f"""
You are an expert HR resume reviewer.

Analyze this resume focusing on:
- Impact
- Skills clarity
- Experience strength
- Improvements for {job_role}

Resume:
{file_content}

Respond clearly with sections.
"""

        client = OpenAI(api_key=OPEN_AI_API_KEY)

        ats_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You simulate professional ATS systems."},
                {"role": "user", "content": ats_prompt}
            ],
            temperature=0.3,
            max_tokens=700
        )

        critique_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior HR resume expert."},
                {"role": "user", "content": feedback_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        st.subheader("📊 ATS Section Check")
        for section, present in ats_sections.items():
            st.write(f"{section.title()}: {'✅ Found' if present else '❌ Missing'}")

        st.subheader("🤖 ATS Score & Matching")
        st.markdown(ats_response.choices[0].message.content)

        st.subheader("📝 AI Resume Feedback")
        st.markdown(critique_response.choices[0].message.content)

    except Exception as e:
        st.error(f"Error: {str(e)}")
