# AI Resume Analyzer

AI Resume Analyzer is a Streamlit web app that compares a candidate resume (PDF) with a job description and produces:

- A semantic similarity score (ATS-style) using Sentence Transformers
- A detailed AI evaluation report using Groq LLM
- Per-point scoring extracted from the report
- A downloadable text report

## Features

- Upload resume in PDF format
- Paste job description text
- Extract resume text automatically
- Compute similarity with `all-mpnet-base-v2`
- Generate point-by-point feedback with scores (out of 5)
- Show average score summary
- Download report as `report.txt`

## Tech Stack

- Python 3.12
- Streamlit
- pdfminer.six
- sentence-transformers
- scikit-learn
- Groq API
- python-dotenv
- torchvision

## Project Structure

```text
Resume_Analyzer/
|-- main.py
|-- requirements.txt
|-- README.md
|-- .env                  # Create this locally (not committed)
|-- myenv/                # Local virtual environment (optional name)
```

## Prerequisites

- Python 3.10+ (3.12 recommended)
- A Groq API key

Create a Groq key from your account dashboard, then use it in `.env` as shown below.

## Installation

1. Clone or download this project.
2. Open terminal in the project folder.
3. Create and activate a virtual environment.
4. Install dependencies.

### Windows (PowerShell)

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run the App

```bash
streamlit run main.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How It Works

1. User uploads resume PDF and enters job description.
2. App extracts resume text with `pdfminer.six`.
3. App computes semantic similarity using sentence embeddings and cosine similarity.
4. App sends resume + job description to Groq model (`llama-3.3-70b-versatile`) for analysis.
5. App parses scores like `x/5` from model output and calculates an average.
6. App displays full report and lets user download it.



## Future Improvements

- Better ATS-style scoring normalization (0 to 100)
- Skill gap highlighting with keyword mapping
- Export to PDF/Markdown report formats
- Multi-resume batch comparison
