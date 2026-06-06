# HireMind AI

HireMind AI is a resume screening and candidate ranking application built using Python, Streamlit, Machine Learning, and MySQL.

I built this project to understand how companies filter resumes and shortlist candidates during the hiring process. While learning Data Analytics and Machine Learning, I wanted to create something that combines NLP, data processing, visualization, and database management into a single real-world application.

The application allows recruiters to upload multiple resumes in PDF format, compare them against a job description, calculate an ATS-style match score, identify candidate skills, and rank applicants based on their relevance to the role.

---

## Features

* Upload multiple resumes in PDF format
* Extract and clean resume text
* Compare resumes with a job description
* Calculate ATS-style similarity scores
* Detect technical skills automatically
* Identify missing skills
* Rank candidates based on ATS score
* Visualize candidate performance using charts
* Store candidate information in MySQL
* View historical candidate records


### Frontend

* Streamlit

### Backend

* Python

### Libraries Used

* Pandas
* Scikit-learn
* NLTK
* PyPDF2
* Plotly
* MySQL Connector

### Database

* MySQL

## How It Works

1. The recruiter uploads one or more resumes.
2. A job description is entered into the system.
3. Resume text is extracted from PDF files.
4. The text is cleaned using NLP techniques.
5. Skills are detected from the resume.
6. TF-IDF Vectorization is applied to both the resume and job description.
7. Cosine Similarity is used to calculate the ATS score.
8. Candidates are ranked according to their score.
9. Results are stored in a MySQL database for future analysis.

## Project Structure

```text
HireMind-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── resumes/
├── assets/
└── database/
```

## What I Learned

While building this project, I learned:

* Working with PDF files in Python
* Text preprocessing using NLP
* TF-IDF Vectorization
* Cosine Similarity
* MySQL database integration
* Streamlit application development
* Data visualization using Plotly
* Debugging real-world Python projects
* Handling database and deployment issues

---

## Challenges Faced

Some of the challenges I encountered while building this project were:

* Extracting text from different resume formats
* Handling missing values and inconsistent resume structures
* Managing MySQL database errors
* Fixing package compatibility issues
* Organizing a large Streamlit application
* Debugging ATS score calculation logic

Solving these issues helped me improve my problem-solving and debugging skills significantly.

## Future Improvements

Planned improvements for future versions:

* Experience detection
* Education analysis
* Resume recommendations
* Candidate shortlisting system
* Email notifications
* AI-generated interview questions
* Resume feedback suggestions
* Cloud deployment with online database
* Admin dashboard

## Installation

Clone the repository:

```bash
git clone https://github.com/rudreshpatil05/HireMind-AI.git
```

Move into the project directory:

```bash
cd HireMind-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Author

Rudresh Patil

Aspiring Data Analyst and Machine Learning Enthusiast interested in building practical projects that solve real-world problems using data and AI.
