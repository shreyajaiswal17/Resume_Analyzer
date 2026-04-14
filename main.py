import streamlit as st                            
from pdfminer.high_level import extract_text      
from sentence_transformers import SentenceTransformer      
from sklearn.metrics.pairwise import cosine_similarity     
from groq import Groq                             
import re                                        
from dotenv import load_dotenv                    
import os
import html


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .main .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, p, label {
        color: #0f172a !important;
    }
    .report-card {
        text-align: left;
        background: #0b1220;
        color: #f8fafc;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
        margin: 8px 0;
        line-height: 1.7;
        font-size: 1rem;
    }
    .report-card * {
        color: #f8fafc !important;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #1d4ed8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#  Session States to store values 
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume=""

if "job_desc" not in st.session_state:
    st.session_state.job_desc=""


st.title("AI Resume Analyzer")



def extract_pdf_text(uploaded_file):
    try:
        extracted_text = extract_text(uploaded_file)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Could not extract text from the PDF file."


def calculate_similarity_bert(text1, text2):
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')      # Use BERT or SBERT This loads a pretrained NLP model

    # Encode the texts directly to embeddings
    embeddings1 = ats_model.encode([text1])
    embeddings2 = ats_model.encode([text2])

    # Text → Vector → Compare
    # Calculate cosine similarity without adding an extra list layer
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity


def get_report(resume,job_desc):
    client = Groq(api_key=api_key)

    prompt=f"""
        You are an AI Resume Analyzer.

        Goal:
        - Compare the candidate resume against the provided job description.
        - Evaluate how well the candidate matches each major requirement.

        Scoring Rules (very important):
        - For each evaluation point, start the line with a score in this exact format: X/5
        - Example valid formats: 3/5, 4.5/5, 0/5
        - Do not use any other score format (no percentages, no /10 scale).
        - Include one emoji after the score:
            - ✅ if aligned
            - ❌ if not aligned
            - ⚠️ if evidence is unclear

        What to evaluate:
        - Required technical skills
        - Relevant experience and responsibilities
        - Domain/industry fit
        - Tools, certifications, and projects
        - Communication/leadership hints if relevant to JD

        Writing requirements:
        - Give concise but specific reasoning for each point.
        - Cite resume evidence when available.
        - If a requirement is missing, clearly state what is missing.

        Required section at the end:
        - Add a final heading exactly as:
            Suggestions to improve your resume:
        - Under this heading, provide practical, prioritized improvements.

        Inputs:
        Candidate Resume: {resume}
        ---
        Job Description: {job_desc}
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def extract_scores(text):
    # Regular expression pattern to find scores in the format x/5, where x can be an integer or a float
    pattern = r'(\d+(?:\.\d+)?)/5'
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    scores = [float(match) for match in matches]
    return scores





if not st.session_state.form_submitted:
    with st.form("my_form"):

        resume_file = st.file_uploader(label="Upload your Resume/CV in PDF format", type="pdf")

        st.session_state.job_desc = st.text_area("Enter the Job Description of the role you are applying for:",placeholder="Job Description...")

     
        submitted = st.form_submit_button("Analyze")
        if submitted:
            if st.session_state.job_desc and resume_file:
                st.info("Extracting Information")

                st.session_state.resume = extract_pdf_text(resume_file)     

                st.session_state.form_submitted = True
                st.rerun()               

            else:
                st.warning("Please Upload both Resume and Job Description to analyze")


if st.session_state.form_submitted:
    score_place = st.info("Generating Scores...")

    ats_score = calculate_similarity_bert(st.session_state.resume,st.session_state.job_desc)

    col1,col2 = st.columns(2,border=True)
    with col1:
        st.write("Few ATS systems use this score to shortlist candidates:")
        st.metric("Similarity Score", f"{ats_score * 100:.1f}%")
        st.caption(f"Raw cosine similarity: {ats_score:.3f}")

    # Call the function to get the Analysis Report from LLM 
    report = get_report(st.session_state.resume,st.session_state.job_desc)

    # Calculate the Average Score 
    report_scores = extract_scores(report)                 
    avg_score = sum(report_scores) / (5*len(report_scores)) 


    with col2:
        st.write("Total Average score according to our AI report:")
        st.metric("AI Average Score", f"{avg_score * 100:.1f}%")
        st.caption(f"Equivalent to {avg_score * 5:.2f}/5")
    score_place.success("Scores generated successfully!")


    st.subheader("AI Generated Analysis Report:")

    formatted_report = html.escape(report).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class='report-card'>
            {formatted_report}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.download_button(
        label="Download Report",
        data=report,
        file_name="report.txt",
        icon=":material/download:",
        )
    

