from flask import Flask, render_template, request, jsonify
import requests
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Hugging Face API details
HF_TOKEN = "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"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Load a small sentence transformer for similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_summary(text, prompt_type="normal"):
    if prompt_type == "contrastive":
        text = "Rewrite the following text with a different perspective but still factual: " + text
    
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL}",
        headers=headers,
        json={"inputs": text}
    )
    summary = response.json()[0]['summary_text']
    return summary

def hallucination_score(summary1, summary2):
    emb1 = model.encode(summary1, convert_to_tensor=True)
    emb2 = model.encode(summary2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return round((1 - similarity) * 100, 2)  # higher = more hallucination risk

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    normal_summary = get_summary(text, "normal")
    contrastive_summary = get_summary(text, "contrastive")
    risk = hallucination_score(normal_summary, contrastive_summary)
    
    return jsonify({
        "original": text,
        "summary": normal_summary,
        "contrastive_summary": contrastive_summary,
        "hallucination_risk": f"{risk}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
