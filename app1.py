import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


load_dotenv()
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="wide")

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_RESUME_CHARS = 18000
MAX_JOB_CHARS = 12000

DEMO_JOB = """Senior Python Developer

We are looking for a Senior Python Developer to build reliable web applications.
Required skills: Python, Django or FastAPI, REST APIs, PostgreSQL, Git, Docker,
cloud deployment, testing, and strong communication. Experience leading code
reviews and mentoring developers is preferred."""

DEMO_RESUMES = {
    "demo_alex_smith.txt": """Alex Smith
Senior Python Developer with 7 years of experience building Django and FastAPI
services. Developed REST APIs with PostgreSQL, Docker, GitHub Actions, and AWS.
Led code reviews, wrote automated tests, and mentored three junior developers.""",
    "demo_jordan_lee.txt": """Jordan Lee
Backend developer with 4 years of Python experience. Built FastAPI REST APIs,
worked with PostgreSQL and Git, and deployed services using Docker. Familiar
with automated testing and cloud platforms.""",
}


class ResumeMatch(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    summary: str = ""


@st.cache_resource(show_spinner=False)
def get_matcher():
    model_name = os.getenv("RESUME_MATCHER_MODEL", "gemini-3.5-flash-lite")
    llm = init_chat_model(
        model=f"google_genai:{model_name}",
        temperature=0,
        max_retries=2,
    )
    return llm.with_structured_output(ResumeMatch)


@st.cache_data(show_spinner=False)
def extract_resume_text(file_bytes: bytes) -> str:
    with pymupdf.open(stream=file_bytes, filetype="pdf") as document:
        text = "\n".join(page.get_text() for page in document)
    return text.strip()[:MAX_RESUME_CHARS]


def analyze_resume(matcher, job_description: str, resume_name: str, resume_text: str):
    if resume_text.startswith("__PDF_ERROR__:"):
        return {
            "resume": resume_name,
            "match_score": None,
            "matching_skills": [],
            "missing_skills": [],
            "evidence": [],
            "summary": "The PDF could not be opened.",
            "error": resume_text.removeprefix("__PDF_ERROR__:") or "Invalid PDF",
        }
    if not resume_text:
        return {
            "resume": resume_name,
            "match_score": None,
            "matching_skills": [],
            "missing_skills": [],
            "evidence": [],
            "summary": "No readable text was found. OCR may be needed.",
            "error": "No readable text",
        }

    try:
        result = matcher.invoke([
            (
                "system",
                """Compare the resume only against the job description.
Assess job-related skills and qualifications using explicit resume evidence.
Do not infer skills. Ignore names, age, gender, race, photos, disability, and
all other protected traits. The score measures skills alignment, not job
performance. Keep evidence concise and factual.""",
            ),
            (
                "human",
                f"JOB DESCRIPTION:\n{job_description}\n\nRESUME:\n{resume_text}",
            ),
        ])
        return {"resume": resume_name, **result.model_dump(), "error": ""}
    except Exception as error:
        error_text = str(error)
        if "404" in error_text or "NOT_FOUND" in error_text:
            error_text = (
                "The configured Gemini model is unavailable. Set "
                "RESUME_MATCHER_MODEL to a supported model, then retry."
            )
        return {
            "resume": resume_name,
            "match_score": None,
            "matching_skills": [],
            "missing_skills": [],
            "evidence": [],
            "summary": "This resume could not be analyzed.",
            "error": error_text[:200],
        }


st.title("📄 AI Resume-to-Job Matcher")
st.caption("Evidence-based resume screening for faster, more consistent HR review.")

use_demo_data = st.toggle(
    "Use default demo inputs",
    value=True,
    help="Loads a sample job and resumes so HR can test the app immediately.",
)

if use_demo_data:
    job = st.text_area("Job description", value=DEMO_JOB, height=180)
    uploaded_files = []
    resume_inputs = list(DEMO_RESUMES.items())
    st.info("Demo mode is active. Turn it off to upload real PDF resumes.")
else:
    job = st.text_area(
        "Paste the job description",
        placeholder="Required skills, responsibilities, and qualifications...",
        height=180,
    )
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    resume_inputs = []

if st.button("Match resumes", type="primary", use_container_width=True):
    if not job.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not use_demo_data and not uploaded_files:
        st.error("Please upload at least one PDF resume.")
        st.stop()

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Add GOOGLE_API_KEY to the deployment secrets or `.env` file.")
        st.stop()

    if use_demo_data:
        resume_inputs = [(name, text) for name, text in resume_inputs]
    else:
        resume_inputs = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > MAX_FILE_SIZE:
                st.warning(f"Skipped {uploaded_file.name}: file is larger than 10 MB.")
                continue
            try:
                resume_inputs.append(
                    (uploaded_file.name, extract_resume_text(file_bytes))
                )
            except Exception as error:
                resume_inputs.append(
                    (uploaded_file.name, f"__PDF_ERROR__:{str(error)[:200]}")
                )

    if not resume_inputs:
        st.error("No usable resumes were found.")
        st.stop()

    with st.spinner("Preparing the resume matcher…"):
        matcher = get_matcher()

    progress = st.progress(0, text="Analyzing resumes…")
    results = []
    job_text = job.strip()[:MAX_JOB_CHARS]

    with ThreadPoolExecutor(max_workers=min(4, len(resume_inputs))) as executor:
        pending = {
            executor.submit(analyze_resume, matcher, job_text, name, text): name
            for name, text in resume_inputs
        }
        for completed, future in enumerate(as_completed(pending), start=1):
            results.append(future.result())
            progress.progress(completed / len(pending), text=f"Analyzed {completed} of {len(pending)} resumes")

    st.session_state["matches"] = pd.DataFrame(results)
    st.session_state["matches"] = st.session_state["matches"].sort_values(
        "match_score", ascending=False, na_position="last"
    ).reset_index(drop=True)
    st.success("Resume analysis complete.")


if "matches" in st.session_state:
    dataframe = st.session_state["matches"]
    valid = dataframe.dropna(subset=["match_score"]).copy()
    valid["match_score"] = valid["match_score"].astype(int)

    st.subheader("HR comparison dashboard")
    first, second, third, fourth = st.columns(4)
    first.metric("Resumes reviewed", len(dataframe))
    second.metric("Successfully analyzed", len(valid))
    third.metric("Average alignment", f"{valid['match_score'].mean():.0f}%" if not valid.empty else "—")
    fourth.metric("Top alignment", f"{valid['match_score'].max()}%" if not valid.empty else "—")

    if not valid.empty:
        st.dataframe(
            valid[["resume", "match_score", "summary"]],
            use_container_width=True,
            hide_index=True,
        )
        selected_name = st.selectbox("Inspect a resume", valid["resume"].tolist())
        selected = valid[valid["resume"] == selected_name].iloc[0]
        st.subheader(selected_name)
        st.write(selected["summary"])
        st.write("**Matching skills:**", ", ".join(selected["matching_skills"]) or "None identified")
        st.write("**Potential skill gaps:**", ", ".join(selected["missing_skills"]) or "None identified")
        st.write("**Resume evidence:**")
        for evidence in selected["evidence"]:
            st.write(f"- {evidence}")
    else:
        st.warning("No resumes were successfully analyzed.")

    failed = dataframe[dataframe["error"].astype(bool)]
    if not failed.empty:
        with st.expander(f"Review {len(failed)} skipped or failed resume(s)"):
            st.dataframe(failed[["resume", "error"]], hide_index=True, use_container_width=True)

    st.download_button(
        "Download comparison report",
        dataframe.to_csv(index=False),
        "resume_match_report.csv",
        "text/csv",
        use_container_width=True,
    )
