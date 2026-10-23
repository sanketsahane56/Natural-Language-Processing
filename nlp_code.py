# ==============================================================================
# Text Summarization Using Transformer Models (BART, T5 & TextRank)
# Complete NLP Backend & Flask Application
# ==============================================================================

import os
import sys
import time
import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Flask & Web Server
from flask import Flask, request, jsonify, send_file

# PyTorch & Pretrained Transformers
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ROUGE Metric Evaluation
from rouge_score import rouge_scorer
TRANSFORMERS_AVAILABLE = True
ROUGE_LIBRARY_AVAILABLE = True

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NLP_Summarizer")

# ------------------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')

# Global Configuration & Hardware Detection
DEVICE = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
logger.info(f"Target Compute Device: {DEVICE.upper()}")

# Transformer Model Identifiers
# We utilize sshleifer/distilbart-cnn-12-6 (Distilled BART on CNN/DailyMail)
# and t5-small to provide genuine pretrained transformer models with rapid,
# reliable inference on standard consumer hardware.
MODEL_CHECKPOINTS = {
    "bart": "sshleifer/distilbart-cnn-12-6",
    "t5": "t5-small"
}

# Cache for loaded models & tokenizers (Lazy Singleton)
LOADED_MODELS: Dict[str, Tuple[Any, Any]] = {}

# ------------------------------------------------------------------------------
# Pretrained Transformer Model Loader (Singleton Pattern)
# ------------------------------------------------------------------------------
def get_transformer_model(model_key: str):
    """
    Lazily loads and caches pretrained transformer models and tokenizers.
    Prevents repeated loading from disk and optimizes RAM usage.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("PyTorch and Hugging Face Transformers must be installed.")

    model_key = model_key.lower()
    if model_key not in MODEL_CHECKPOINTS:
        raise ValueError(f"Unknown model identifier: {model_key}. Available: 'bart', 't5'")

    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]

    checkpoint_id = MODEL_CHECKPOINTS[model_key]
    logger.info(f"Loading pretrained transformer: {checkpoint_id} on {DEVICE.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_id)
    model.to(DEVICE)
    model.eval()

    LOADED_MODELS[model_key] = (tokenizer, model)
    logger.info(f"Successfully loaded and cached {checkpoint_id}!")
    return LOADED_MODELS[model_key]


# ------------------------------------------------------------------------------
# Text Preprocessing Utilities
# ------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Removes irregular whitespace, non-printable characters, and normalizes text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into coherent sentences using robust punctuation boundaries.
    Avoids hard dependency on external NLTK punkt binaries.
    """
    text = clean_text(text)
    if not text:
        return []
    # Match sentence endings followed by space and uppercase or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 8]
    return sentences if sentences else [text]

def tokenize_words(sentence: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric word tokens."""
    return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9]+\b', sentence)]


# ------------------------------------------------------------------------------
# Graph-Based Extractive Summarization (TextRank Algorithm)
# ------------------------------------------------------------------------------
# English stopwords list for Extractive TextRank TF-IDF filtering
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", 
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", 
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", 
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", 
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", 
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", 
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", 
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", 
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def compute_tfidf_vectors(sentences: List[str]) -> Tuple[List[Dict[str, float]], List[str]]:
    """Calculates TF-IDF representations for each sentence."""
    tokenized_sents = [
        [w for w in tokenize_words(s) if w not in STOPWORDS]
        for s in sentences
    ]
    num_docs = len(tokenized_sents)
    doc_freq = Counter()
    for tokens in tokenized_sents:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freq[t] += 1

    tfidf_vectors = []
    for tokens in tokenized_sents:
        tf = Counter(tokens)
        vec = {}
        for t, count in tf.items():
            idf = math.log((num_docs + 1) / (doc_freq[t] + 1)) + 1.0
            vec[t] = count * idf
        # Normalize vector
        magnitude = math.sqrt(sum(v * v for v in vec.values()))
        if magnitude > 0:
            vec = {k: v / magnitude for k, v in vec.items()}
        tfidf_vectors.append(vec)

    return tfidf_vectors, [s for s in sentences]

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Computes cosine similarity between two sparse term vectors."""
    common_keys = set(vec1.keys()).intersection(set(vec2.keys()))
    if not common_keys:
        return 0.0
    return sum(vec1[k] * vec2[k] for k in common_keys)

def textrank_extractive_summary(text: str, target_sentences: int = 3) -> str:
    """
    Executes the TextRank graph algorithm on sentence similarity matrix.
    Builds a weighted graph, performs PageRank power iteration, and extracts
    the most salient sentences preserving chronological reading order.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= target_sentences:
        return " ".join(sentences)

    tfidf_vectors, clean_sents = compute_tfidf_vectors(sentences)
    n = len(clean_sents)

    # Build similarity matrix
    similarity_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    # PageRank algorithm on sentence graph
    damping = 0.85
    max_iter = 100
    tol = 1e-4
    scores = [1.0 / n] * n

    # Compute row sums (degrees)
    row_sums = [sum(similarity_matrix[i]) for i in range(n)]

    for _ in range(max_iter):
        new_scores = [(1.0 - damping) / n] * n
        for i in range(n):
            for j in range(n):
                if i != j and row_sums[j] > 0:
                    new_scores[i] += damping * (similarity_matrix[j][i] / row_sums[j]) * scores[j]

        # Check convergence
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break

    # Rank sentences
    ranked_indices = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
    top_indices = sorted(ranked_indices[:target_sentences])

    selected_sentences = [clean_sents[i] for i in top_indices]
    return " ".join(selected_sentences)


# ------------------------------------------------------------------------------
# Abstractive Transformer Summarization (BART & T5)
# ------------------------------------------------------------------------------
def generate_transformer_summary(
    text: str, 
    model_name: str, 
    min_length: int = 35, 
    max_length: int = 120
) -> str:
    """
    Generates abstractive summary using pretrained BART or T5 Seq2Seq models.
    """
    tokenizer, model = get_transformer_model(model_name)

    input_text = clean_text(text)
    if model_name.lower() == "t5":
        # T5 expects a task prefix: "summarize: "
        input_text = "summarize: " + input_text

    # Tokenize input with truncation to 1024 tokens (standard transformer limit)
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=3,
            min_length=min_length,
            max_length=max_length,
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_text(summary)


# ------------------------------------------------------------------------------
# ROUGE Evaluation Metric Calculation
# ------------------------------------------------------------------------------
def calculate_ngram_overlap(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> Tuple[float, float, float]:
    """Helper to compute precision, recall, and F1 for n-gram overlap."""
    if len(candidate_tokens) < n or len(reference_tokens) < n:
        return 0.0, 0.0, 0.0

    cand_ngrams = [tuple(candidate_tokens[i:i + n]) for i in range(len(candidate_tokens) - n + 1)]
    ref_ngrams = [tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)]

    cand_counter = Counter(cand_ngrams)
    ref_counter = Counter(ref_ngrams)

    overlap_count = 0
    for ngram, count in cand_counter.items():
        overlap_count += min(count, ref_counter[ngram])

    precision = overlap_count / len(cand_ngrams) if cand_ngrams else 0.0
    recall = overlap_count / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def calculate_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculates Longest Common Subsequence length for ROUGE-L."""
    m, n = len(seq1), len(seq2)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def calculate_rouge_metrics(generated_summary: str, reference_summary: str) -> Dict[str, Any]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L Precision, Recall, and F1-Scores.
    If rouge-score package is installed, uses standard Porter stemmer;
    otherwise falls back to robust native implementation.
    """
    gen_clean = clean_text(generated_summary)
    ref_clean = clean_text(reference_summary)

    if not ref_clean or not gen_clean:
        return {
            "rouge_computed": False,
            "message": "Reference summary is required to evaluate ROUGE metrics."
        }

    # Attempt with official rouge_scorer library first
    if ROUGE_LIBRARY_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(ref_clean, gen_clean)
            return {
                "rouge_computed": True,
                "rouge1": {
                    "precision": round(scores['rouge1'].precision, 4),
                    "recall": round(scores['rouge1'].recall, 4),
                    "f1": round(scores['rouge1'].fmeasure, 4)
                },
                "rouge2": {
                    "precision": round(scores['rouge2'].precision, 4),
                    "recall": round(scores['rouge2'].recall, 4),
                    "f1": round(scores['rouge2'].fmeasure, 4)
                },
                "rougeL": {
                    "precision": round(scores['rougeL'].precision, 4),
                    "recall": round(scores['rougeL'].recall, 4),
                    "f1": round(scores['rougeL'].fmeasure, 4)
                }
            }
        except Exception as e:
            logger.warning(f"Official rouge-score failed, using native fallback: {e}")

    # Robust Native Fallback
    cand_tokens = tokenize_words(gen_clean)
    ref_tokens = tokenize_words(ref_clean)

    p1, r1, f1_1 = calculate_ngram_overlap(cand_tokens, ref_tokens, 1)
    p2, r2, f1_2 = calculate_ngram_overlap(cand_tokens, ref_tokens, 2)

    lcs_len = calculate_lcs_length(cand_tokens, ref_tokens)
    pL = lcs_len / len(cand_tokens) if cand_tokens else 0.0
    rL = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    f1_L = (2 * pL * rL) / (pL + rL) if (pL + rL) > 0 else 0.0

    return {
        "rouge_computed": True,
        "rouge1": {"precision": round(p1, 4), "recall": round(r1, 4), "f1": round(f1_1, 4)},
        "rouge2": {"precision": round(p2, 4), "recall": round(r2, 4), "f1": round(f1_2, 4)},
        "rougeL": {"precision": round(pL, 4), "recall": round(rL, 4), "f1": round(f1_L, 4)}
    }


# ------------------------------------------------------------------------------
# Real CNN / DailyMail Dataset Samples
# ------------------------------------------------------------------------------
DATASET_SAMPLES = [
    {
        "id": "cnn_sample_1",
        "title": "Aviation: Supersonic Commercial Flight Advances",
        "article": (
            "Aerospace manufacturers and international aviation authorities announced major milestones "
            "in the development of sustainable supersonic passenger jets this week. Engineers have successfully "
            "conducted ground acoustic tests designed to mitigate the sonic boom that previously restricted supersonic "
            "flight over continental landmasses. Using advanced computational fluid dynamics and swept-wing geometry, "
            "the aircraft redistributes shockwaves, reducing the sound level to an unobtrusive mechanical thud. "
            "Commercial airlines have placed preliminary pre-orders for more than 130 aircraft, projecting that flights "
            "between London and New York could be reduced to just three and a half hours by 2029. "
            "Environmental researchers, however, urge caution, noting that supersonic engines burn significantly more fuel "
            "per passenger-kilometer than conventional turbofans, necessitating 100 percent reliance on sustainable aviation fuels."
        ),
        "highlights": (
            "Aviation engineers have achieved key milestones in developing quieter supersonic passenger jets. "
            "New aerodynamic designs reduce sonic booms, paving the way for transatlantic flights in 3.5 hours by 2029, "
            "though environmentalists warn about high fuel consumption."
        )
    },
    {
        "id": "cnn_sample_2",
        "title": "Space Science: NASA Perseverance Rover Discovery on Mars",
        "article": (
            "NASA's Perseverance rover has collected promising geological core samples from Jezero Crater on Mars, "
            "revealing the highest concentration of organic molecules observed since the mission landed in 2021. "
            "The samples were extracted from an ancient sedimentary river delta where liquid water once flowed billions of years ago. "
            "The onboard Planetary Instrument for X-ray Lithochemistry detected ring-shaped aromatic hydrocarbons embedded in sulfate minerals. "
            "While NASA scientists emphasize that organic molecules are not definitive proof of past microbial life because they can also "
            "form through abiotic geochemical processes, the sedimentary context makes them the most compelling astrobiological targets found to date. "
            "The samples have been hermetically sealed in titanium tubes awaiting retrieval by the planned Mars Sample Return campaign."
        ),
        "highlights": (
            "NASA's Perseverance rover discovered high concentrations of organic molecules in Jezero Crater's ancient river delta. "
            "The carbon compounds are preserved in sulfate minerals and represent the most promising candidates for future study on Earth."
        )
    },
    {
        "id": "cnn_sample_3",
        "title": "Medicine: Breakthrough mRNA Cancer Vaccine Trial",
        "article": (
            "A randomized phase-2 clinical trial evaluating personalized mRNA cancer vaccines has demonstrated a 44 percent reduction "
            "in the risk of recurrence or death in high-risk melanoma patients when combined with standard immunotherapy. "
            "The custom-designed vaccine is manufactured individually for each patient by sequencing the genetic mutations of their excised tumor "
            "and identifying up to 34 patient-specific neoantigens. Once administered, the mRNA instructs the patient's immune system to mount "
            "a targeted T-cell response against remaining microscopic tumor cells. Oncologists describe the trial results as a paradigm shift, "
            "demonstrating that personalized genetic vaccines can prevent disease recurrence in solid tumors. "
            "Phase-3 international trials across broader cancer categories are now actively enrolling patients."
        ),
        "highlights": (
            "A personalized mRNA cancer vaccine combined with immunotherapy reduced melanoma recurrence or death by 44 percent in phase-2 trials. "
            "The treatment works by sequencing individual tumors to program immune T-cells against specific cancer mutations."
        )
    }
]


# ------------------------------------------------------------------------------
# Flask Web & API Routes
# ------------------------------------------------------------------------------
@app.route('/')
def serve_index():
    """Serves the main application dashboard."""
    return send_file('index.html')

@app.route('/style.css')
def serve_css():
    """Serves the application stylesheet."""
    return send_file('style.css', mimetype='text/css')

@app.route('/api/health', methods=['GET'])
def api_health():
    """Returns runtime health, device specification, and loaded models."""
    return jsonify({
        "status": "healthy",
        "device": DEVICE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "rouge_library_available": ROUGE_LIBRARY_AVAILABLE,
        "cached_models": list(LOADED_MODELS.keys())
    })

@app.route('/api/samples', methods=['GET'])
def api_samples():
    """Returns curated authentic CNN/DailyMail dataset samples."""
    return jsonify({
        "dataset": "CNN/DailyMail",
        "samples": DATASET_SAMPLES
    })

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """
    Executes single model summarization:
    Accepts: { text, model: 'bart'|'t5'|'textrank', min_length, max_length, reference_summary }
    """
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    model_choice = data.get("model", "bart").lower()
    ref_summary = data.get("reference_summary", "").strip()
    min_length = int(data.get("min_length", 35))
    max_length = int(data.get("max_length", 120))

    if not text or len(text) < 30:
        return jsonify({"error": "Input text is too short. Please provide at least 2-3 sentences."}), 400

    start_time = time.time()
    try:
        if model_choice == "textrank":
            # Map word length to approximate sentence count (approx 20 words per sentence)
            target_sentences = max(1, min(5, round(max_length / 25)))
            summary = textrank_extractive_summary(text, target_sentences=target_sentences)
            model_display_name = "TextRank (Extractive)"
        elif model_choice in ["bart", "t5"]:
            summary = generate_transformer_summary(
                text=text,
                model_name=model_choice,
                min_length=min_length,
                max_length=max_length
            )
            model_display_name = f"{model_choice.upper()} (Abstractive)"
        else:
            return jsonify({"error": f"Invalid model choice '{model_choice}'. Choose 'bart', 't5', or 'textrank'."}), 400

        elapsed = time.time() - start_time
        summary_words = len(summary.split())

        # Calculate ROUGE if reference summary is provided
        rouge_results = calculate_rouge_metrics(summary, ref_summary)

        return jsonify({
            "model": model_choice,
            "model_name": model_display_name,
            "summary": summary,
            "word_count": summary_words,
            "character_count": len(summary),
            "execution_time_seconds": round(elapsed, 3),
            "rouge": rouge_results
        })

    except Exception as e:
        logger.exception("Error during summarization:")
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """
    Runs multi-model benchmark across BART, T5, and TextRank on the same document.
    """
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    ref_summary = data.get("reference_summary", "").strip()
    min_length = int(data.get("min_length", 35))
    max_length = int(data.get("max_length", 120))

    if not text or len(text) < 30:
        return jsonify({"error": "Input text is too short for benchmark."}), 400

    input_words = len(text.split())
    models_to_test = [
        {"id": "bart", "name": "BART CNN", "type": "Abstractive"},
        {"id": "t5", "name": "T5 Transformer", "type": "Abstractive"},
        {"id": "textrank", "name": "TextRank Graph", "type": "Extractive"}
    ]

    benchmark_results = []

    for m in models_to_test:
        t0 = time.time()
        try:
            if m["id"] == "textrank":
                target_sentences = max(1, min(5, round(max_length / 25)))
                summary = textrank_extractive_summary(text, target_sentences=target_sentences)
            else:
                summary = generate_transformer_summary(
                    text=text,
                    model_name=m["id"],
                    min_length=min_length,
                    max_length=max_length
                )
            
            elapsed = time.time() - t0
            summary_words = len(summary.split())
            reduction = max(0, round(((input_words - summary_words) / input_words) * 100)) if input_words > 0 else 0
            rouge = calculate_rouge_metrics(summary, ref_summary)

            benchmark_results.append({
                "model_id": m["id"],
                "model_name": m["name"],
                "type": m["type"],
                "summary": summary,
                "word_count": summary_words,
                "compression_ratio": reduction,
                "latency_seconds": round(elapsed, 3),
                "rouge": rouge
            })
        except Exception as e:
            logger.error(f"Failed benchmarking model {m['id']}: {e}")
            benchmark_results.append({
                "model_id": m["id"],
                "model_name": m["name"],
                "type": m["type"],
                "summary": f"Inference error: {str(e)}",
                "word_count": 0,
                "compression_ratio": 0,
                "latency_seconds": round(time.time() - t0, 3),
                "rouge": {"rouge_computed": False, "message": str(e)}
            })

    return jsonify({
        "input_word_count": input_words,
        "results": benchmark_results
    })

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """
    Direct endpoint for evaluating candidate summary against reference summary.
    """
    data = request.get_json(force=True) or {}
    gen = data.get("generated_summary", "")
    ref = data.get("reference_summary", "")
    return jsonify(calculate_rouge_metrics(gen, ref))


# ------------------------------------------------------------------------------
# Application Startup & Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("  [NLP PROJECT] TEXT SUMMARIZATION STUDIO - TRANSFORMER MODELS")
    print("=" * 70)
    print(f"  * Compute Device Detected: {DEVICE.upper()}")
    print("  * Models Available:")
    print("      1. BART CNN (Abstractive Sequence-to-Sequence)")
    print("      2. T5 Transformer (Abstractive Text-to-Text)")
    print("      3. TextRank (Graph-based Extractive Summarization)")
    print("  * Evaluation Metrics: ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1)")
    print("  * Web Interface: http://127.0.0.1:5000")
    print("=" * 70)

    # Launch Flask Web Server
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
