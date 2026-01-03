AI Resume Ranker

This project implements an AI-powered resume ranking system using Natural Language Processing (NLP) techniques.

Objective
To automatically rank multiple resumes based on their relevance to a given job description.

Technologies Used
- Python
- Flask
- SpaCy
- Scikit-learn
- TF-IDF Vectorization

How It Works
1. Upload multiple PDF resumes
2. Enter a job description
3. Resumes are preprocessed using NLP techniques
4. TF-IDF and cosine similarity are used for scoring
5. Resumes are ranked based on relevance
6. A downloadable CSV report is generated

Output
- Ranked list of candidates
- Relevance score for each resume
- Automated shortlisting

## Limitations
- Keyword-based matching (no semantic understanding)
- Resume formatting may affect text extraction
