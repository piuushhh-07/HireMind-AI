import streamlit as st
from PyPDF2 import PdfReader
import re

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="HireMind AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# DOWNLOAD STOPWORDS
# ---------------------------------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stTextArea textarea {
    background-color: #262730;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🤖 HireMind AI")

st.sidebar.markdown("""
## AI Resume Screening System

### Features
✅ Resume Analysis  
✅ ATS Score Detection  
✅ Skill Extraction  
✅ Candidate Ranking  
✅ Data Visualization  
✅ NLP Processing  

---

### Technologies Used
- Python
- Streamlit
- NLP
- Machine Learning
- Scikit-learn
- Plotly

""")

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.markdown("""
# 🤖 HireMind AI

### Intelligent Resume Screening Platform
""")

st.markdown("---")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "📂 Upload Multiple Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# JOB DESCRIPTION
# ---------------------------------------------------

job_description = st.text_area(
    "📝 Enter Job Description"
)

# ---------------------------------------------------
# CLEAN TEXT FUNCTION
# ---------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    stop_words = set(stopwords.words('english'))

    words = text.split()

    cleaned_words = []

    for word in words:

        if word not in stop_words:
            cleaned_words.append(word)

    cleaned_text = " ".join(cleaned_words)

    return cleaned_text

# ---------------------------------------------------
# SKILLS DATABASE
# ---------------------------------------------------

skills_list = [
    "python",
    "sql",
    "machine learning",
    "power bi",
    "excel",
    "data analysis",
    "numpy",
    "pandas",
    "tensorflow",
    "deep learning",
    "docker",
    "java",
    "c++",
    "statistics",
    "nlp",
    "data visualization",
    "scikit learn",
    "matplotlib",
    "seaborn"
]

# ---------------------------------------------------
# MAIN SYSTEM
# ---------------------------------------------------

if uploaded_files:

    candidate_scores = []

    all_detected_skills = []

    for uploaded_file in uploaded_files:

        st.markdown("---")

        st.subheader(f"📄 Resume: {uploaded_file.name}")

        # ---------------------------------------------------
        # READ PDF
        # ---------------------------------------------------

        pdf_reader = PdfReader(uploaded_file)

        resume_text = ""

        for page in pdf_reader.pages:
            resume_text += page.extract_text()

        # ---------------------------------------------------
        # CLEAN RESUME
        # ---------------------------------------------------

        cleaned_resume = clean_text(resume_text)

        # ---------------------------------------------------
        # DETECT SKILLS
        # ---------------------------------------------------

        detected_skills = []

        for skill in skills_list:

            if skill in cleaned_resume:

                detected_skills.append(skill)

                all_detected_skills.append(skill)

        # ---------------------------------------------------
        # JOB SKILLS
        # ---------------------------------------------------

        job_skills = []

        for skill in skills_list:

            if skill in job_description.lower():
                job_skills.append(skill)

        # ---------------------------------------------------
        # MISSING SKILLS
        # ---------------------------------------------------

        missing_skills = []

        for skill in job_skills:

            if skill not in detected_skills:
                missing_skills.append(skill)

        # ---------------------------------------------------
        # TF-IDF
        # ---------------------------------------------------

        tfidf = TfidfVectorizer()

        vectors = tfidf.fit_transform([
            cleaned_resume,
            job_description
        ])

        # ---------------------------------------------------
        # COSINE SIMILARITY
        # ---------------------------------------------------

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )

        # ---------------------------------------------------
        # ATS SCORE
        # ---------------------------------------------------

        ats_score = round(
            similarity[0][0] * 100,
            2
        )

        candidate_scores.append({
            "name": uploaded_file.name,
            "score": ats_score
        })

        
        # ---------------------------------------------------
        # DASHBOARD METRICS
        # ---------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 ATS Score", f"{ats_score}%")

        with col2:
            st.metric("✅ Skills Found", len(detected_skills))

        with col3:
            st.metric("❌ Missing Skills", len(missing_skills))

        # ---------------------------------------------------
        # ATS SCORE STATUS
        # ---------------------------------------------------

        if ats_score >= 80:
            st.success(f"🔥 Excellent Match: {ats_score}%")

        elif ats_score >= 60:
            st.warning(f"⚡ Moderate Match: {ats_score}%")

        else:
            st.error(f"❌ Low Match: {ats_score}%")

        # ---------------------------------------------------
        # PROGRESS BAR
        # ---------------------------------------------------

        st.progress(int(ats_score))

        # ---------------------------------------------------
        # DETECTED SKILLS
        # ---------------------------------------------------

        st.subheader("✅ Detected Skills")

        if detected_skills:

            skill_cols = st.columns(3)

            for index, skill in enumerate(detected_skills):

                with skill_cols[index % 3]:
                    st.success(skill)

        else:
            st.warning("No Skills Detected")

        # ---------------------------------------------------
        # MISSING SKILLS
        # ---------------------------------------------------

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:
                st.error(skill)

        else:
            st.success("No Missing Skills")

        # ---------------------------------------------------
        # EXPANDERS
        # ---------------------------------------------------

        with st.expander("📄 View Original Resume Text"):
            st.write(resume_text)

        with st.expander("🧹 View Cleaned Resume Text"):
            st.write(cleaned_resume)

    # ---------------------------------------------------
    # FINAL DASHBOARD
    # ---------------------------------------------------

    st.markdown("---")

    st.header("🏆 Candidate Rankings")

    ranked_candidates = sorted(
        candidate_scores,
        key=lambda x: x["score"],
        reverse=True
    )

    rank = 1

    for candidate in ranked_candidates:

        st.write(
            f"{rank}. {candidate['name']} → {candidate['score']}%"
        )

        rank += 1

    # ---------------------------------------------------
    # DATAFRAME
    # ---------------------------------------------------

    df = pd.DataFrame(candidate_scores)

    st.subheader("📊 Candidate Score Table")

    st.dataframe(df)

    # ---------------------------------------------------
    # BAR GRAPH
    # ---------------------------------------------------

    fig = px.bar(
        df,
        x="name",
        y="score",
        text="score",
        title="Candidate ATS Scores"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------

    if all_detected_skills:

        skill_df = pd.DataFrame(
            all_detected_skills,
            columns=["skill"]
        )

        skill_count = (
            skill_df["skill"]
            .value_counts()
            .reset_index()
        )

        skill_count.columns = [
            "skill",
            "count"
        ]

        pie_fig = px.pie(
            skill_count,
            names="skill",
            values="count",
            title="Skill Distribution"
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    # ---------------------------------------------------
    # BEST CANDIDATE
    # ---------------------------------------------------

    if not df.empty:
        top_candidate = df.sort_values(
        by="score",
        ascending=False
        ).iloc[0]

    st.markdown("---")

    st.success(
        f"🏆 Best Candidate: "
        f"{top_candidate['name']} "
        f"with {top_candidate['score']}% ATS Match"
    )
else:
    st.warning("No candidates available.")
# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption("🚀 Built by Rudresh Patil | HireMind AI")
