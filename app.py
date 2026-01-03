import re
import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from flask import Flask, request, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_WEIGHTS = {
    "python": 3,
    "machine learning": 4,
    "data analysis": 3,
    "sql": 2,
    "flask": 2,
    "nlp": 3
}

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def skill_boost(resume_text):
    score = 0
    for skill, weight in SKILL_WEIGHTS.items():
        if skill in resume_text:
            score += weight
    return score

def experience_boost(text):
    years = re.findall(r'(\d+)\+?\s+years?', text)
    if years:
        years = max(map(int, years))
        if years >= 5:
            return 10
        elif years >= 3:
            return 5
    return 0


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def rank_resumes(job_desc, resumes):
    all_docs = [job_desc] + resumes
    tfidf_matrix, _ = vectorize_texts(all_docs)

    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(job_vector, resume_vectors).flatten()

    final_scores = []
    for i, score in enumerate(scores):
        boosted_score = score + skill_boost(resumes[i]) + experience_boost(resumes[i])
        final_scores.append(boosted_score)

        max_score = max(final_scores)
        final_scores = [round(40 + (s / max_score) * 60, 2) for s in final_scores]

    return final_scores


def create_report(names, scores):
    df = pd.DataFrame({"Candidate": names, "Score": scores})
    df = df.sort_values(by="Score", ascending=False)
    #df = df[df["Score"] >= 50]
    df.to_csv("ranked_report.csv", index=False)
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc_raw = request.form["job_description"]
        job_desc = preprocess(job_desc_raw)

        resumes = request.files.getlist("resumes")
        resume_texts = []
        names = []

        for file in resumes:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            raw_text = extract_text_from_pdf(file_path)
            cleaned_text = preprocess(raw_text)
            resume_texts.append(cleaned_text)
            names.append(file.filename)

        scores = rank_resumes(job_desc, resume_texts)
        df = create_report(names, scores)
        table_html = df.to_html(index=False)
        return render_template("results.html", table=table_html)

    return render_template("index.html")

@app.route("/download")
def download():
    return send_file("ranked_report.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
