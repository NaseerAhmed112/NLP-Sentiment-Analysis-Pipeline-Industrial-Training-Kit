# 🗣️ Data Science Internship — Project 4
## NLP & Sentiment Analysis Pipeline | Industrial Training Kit (Optional Mastery Phase)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-1E90FF?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF%20%2B%20Naive%20Bayes-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Sparse%20Matrices-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-blue?style=for-the-badge)

**Powered by [DecodeLabs](https://www.decodelabs.tech) | Greater Lucknow, India**

</div>

---

## 📌 Short Description

> *"Machine learning does not operate on raw text. It operates on mathematics — and a stop-word filter that strips 'not' out of 'not happy' will confidently predict the opposite of the truth."*

**Project 4** is the **Optional Mastery Phase** of the internship: **Natural Language Processing & Sentiment Analysis**. This track isn't about "just reading words" — it's about **Mathematical Linguistics**. The goal is to build a robust text pre-processing pipeline, convert unstructured human language into mathematical arrays using **TF-IDF**, and train a machine learning classifier to decode human sentiment through pure computational logic — while avoiding the negation, memory, and probability traps that silently corrupt most NLP pipelines.

---

## 🎯 Project Goal

**Program a machine to read and mathematically categorize unstructured human text** (e.g., product reviews) as **Positive** or **Negative** — evaluated on a pipeline that correctly preserves negation, controls vocabulary explosion, and handles unseen words at inference time.

---

## 🏗️ Pipeline Architecture

```
Raw Unstructured  ──►  Pre-Process  ──►  Vectorize  ──►  Model
Text (Raw String)      (Cleaned Tokens)   (Sparse Matrix)  (Probability Distribution)
                        Tokenize →
                        Stop-Word Filter →
                        Lemmatize
```

---

## 📂 Repository Structure

```
data-science-internship-project-4/
│
├── 📓 NLP_Sentiment_Analysis_Project_4.ipynb   ← Main notebook (run this)
├── 📊 reviews.csv                               ← Product review dataset (text + label)
├── 📄 README.md                                 ← You are here
│
├── outputs/
│   ├── stopword_negation_test.png               ← "not happy" before/after custom filter
│   ├── tfidf_top_terms.png                       ← Highest-weighted TF-IDF terms
│   ├── confusion_matrix_nb.png                   ← Naive Bayes results
│   ├── confusion_matrix_svm.png                  ← SVM results
│   └── sparse_matrix_savings.png                 ← Dense vs CSR memory comparison
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Name** | Product Review / Sentiment Dataset |
| **Type** | Unstructured text (raw reviews) + binary label |
| **Composition** | 80% of enterprise data is unstructured (reviews, emails, tickets) — effectively invisible to traditional analytics |
| **Features** | Raw text string per review |
| **Target** | `Sentiment` — Positive / Negative |

> **The Core Problem:** Machine learning models cannot read raw strings. Every review must be converted from qualitative human language into a quantitative, sparse mathematical representation *before* any model can learn from it.

---

## 🔬 Key Engineering Concepts

### ⚠️ Trap #1 — The Stop-Word Negation Trap

Default NLP stop-word dictionaries silently discard negations, flipping the meaning of a sentence.

| Step | Input / Result |
|------|-----------------|
| Input | `"I am not happy."` |
| Default NLTK filter | `"I am happy."` → **Predicts POSITIVE (Failure)** |
| Customized filter | `"not happy"` → **Predicts NEGATIVE (Success)** |

```
❌ Never use a default stop-word list as-is for sentiment tasks.
✅ Always explicitly exclude negations ('not', 'no', 'never', etc.)
   from your stop-word set using set-based union/difference operations.
```

---

### ⚠️ Trap #2 — Stemming vs. Lemmatization

| Method | Mechanism | Example | Result |
|--------|-----------|---------|--------|
| **Stemming** | Fast, rule-based suffix stripping | `"Studying"` → `"Studi"` | ❌ Produces non-words, destroys semantic structure |
| **Lemmatization** | Dictionary-based morphological analysis | `"Went"` → `"Go"` | ✅ Returns valid base words, preserves meaning |

```
Mandate: Project 4 requires the NLTK WordNetLemmatizer.
```

**Context is everything — POS-guided lemmatization:**

| Path | Input | POS Tag | Output |
|------|-------|---------|--------|
| A | `"went"` | None (assumes noun) | `"went"` ❌ Incorrect reduction |
| B | `"went"` | `'v'` (verb) | `"go"` ✅ Correct reduction |

```
❌ Never lemmatize without passing a POS tag.
✅ Always map Treebank POS tags to WordNet tags before lemmatizing —
   without it, verbs and adjectives will not reduce properly.
```

---

### 🔹 Translating English to Math — TF-IDF

```
TF × IDF = Spatial Weight
```

| Term | Meaning |
|------|---------|
| **Term Frequency (TF)** | How often the word appears in *this* document |
| **Inverse Document Frequency (IDF)** | How rare the word is across *all* documents |

| Word | TF Status | IDF Status | Resulting Weight |
|------|-----------|------------|-------------------|
| `"the"` | High TF | Low IDF | ~0 (irrelevant) |
| `"terrible"` | High TF | High IDF | High (highly informative) |

---

### 🔹 Capturing Word Relationships — N-Grams

| Type | Example | Behavior |
|------|---------|----------|
| **Unigram** (1-word) | `["Not", "Good"]` | Model reads "Good" → predicts **Positive** ❌ |
| **Bigram** (2-word) | `["Not good"]` | Captures the exact negated sentiment ✅ |

```
⚠️ Tradeoff: N-grams exponentially increase vocabulary size.
✅ Engineering Rule: Set max_features (e.g., top 10,000 terms) and
   min_df=2 to bound dimensionality and exclude rare typos.
```

---

### ⚠️ Trap #3 — The Memory Crash (Curse of Dimensionality)

```
10,000 reviews × 20,000 unique terms = 200,000,000 matrix cells
99% of those cells are empty zeros.
```

Storing this as a dense array rapidly exhausts RAM and causes O(N³) operations and system crashes.

**The Fix — SciPy CSR (Compressed Sparse Row) format:**

```python
from scipy.sparse import csr_matrix

# CSR stores only:
#   Data    -> non-zero values      e.g. [5, 8, 3]
#   Indices -> column locations     e.g. [1, 2, 0]
#   Indptr  -> row pointers         e.g. [0, 1, 2, 3]
```

```
❌ Never densify a TF-IDF matrix for storage or training.
✅ Always keep vectorized text in CSR sparse format — it ignores
   zeros, saves gigabytes of memory, and is the industry standard
   for training linear classifiers.
```

---

### 🔹 Probabilistic Inference — Naive Bayes

With text mathematically vectorized, a **Multinomial Naive Bayes** classifier multiplies two probabilities to yield a final prediction:

```
P(y)      → prior probability / bias of a class existing at all
P(x | y)  → conditional probability of these specific features
            appearing given that class
```

---

### ⚠️ Trap #4 — The Zero-Frequency Problem

Naive Bayes multiplies probabilities together. If a new word appears at test time that was never seen in training, its probability is 0 — and one zero destroys the entire posterior probability.

```
0.8 × 0.9 × 0.0  × 0.7 = 0        ❌ (one unseen word kills the prediction)

Apply Laplace Smoothing (alpha = 1.0):

0.8 × 0.9 × 0.01 × 0.7 = 0.005    ✅ (zero becomes a pseudocount)
```

```
❌ Never train Naive Bayes without smoothing.
✅ Always apply Laplace/additive smoothing (alpha=1.0) so no
   probability is ever an absolute zero.
```

---

### 🔹 Choosing the Right Bayes

| Variant | Best For | Mechanism |
|---------|----------|-----------|
| **MultinomialNB** | Perfectly balanced datasets (~50/50) | Probability of belonging *to* a class based on TF-IDF |
| **ComplementNB** | Imbalanced datasets (e.g. 95% Positive / 5% Negative) | Probability of belonging to the *complement* of all other classes — normalizes weights, corrects extreme bias |

---

## ✅ The Zero-Guesswork Protocol (Complete Checklist)

```
☑  Exclude negations ('not', 'no', 'never') from default stop-word lists
☑  ALWAYS use WordNetLemmatizer with POS tags — never bare stemming
☑  Vectorize with TF-IDF, not raw counts
☑  Use unigrams + bigrams to capture negated phrases like "not good"
☑  Bound vocabulary size with max_features and min_df to avoid explosion
☑  Store vectorized matrices strictly in SciPy CSR sparse format
☑  Apply Laplace Smoothing (alpha=1.0) before training Naive Bayes
☑  Use ComplementNB instead of MultinomialNB for imbalanced sentiment data
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/NaseerAhmed112/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis

# 2. Place your review dataset in the project root
# (reviews.csv or equivalent text + label dataset)

# 3. Install dependencies
pip install pandas numpy scikit-learn nltk scipy matplotlib seaborn jupyter

# 4. Download required NLTK resources
python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger punkt

# 5. Launch Jupyter
jupyter notebook NLP_Sentiment_Analysis_Project_4.ipynb

# 6. Run All → Kernel → Restart & Run All
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `Pandas` | 2.0+ | Data loading and manipulation |
| `NumPy` | 1.24+ | Numerical operations |
| `NLTK` | 3.8+ | Tokenization, stop-words, POS tagging, WordNetLemmatizer |
| `Scikit-learn` | 1.3+ | TfidfVectorizer, MultinomialNB, ComplementNB, SVM, metrics |
| `SciPy` | 1.11+ | CSR sparse matrix storage |
| `Matplotlib` | 3.7+ | Confusion matrices, TF-IDF term plots |
| `Seaborn` | 0.12+ | Class distribution, sentiment visualizations |

---

## 📊 Results Summary

```
═══════════════════════════════════════════════════════════════
  PROJECT 4: NLP & SENTIMENT ANALYSIS PIPELINE — FINAL RESULTS
═══════════════════════════════════════════════════════════════
  Dataset          : Product Reviews (unstructured text)
  Pre-Processing   : Tokenization → Negation-safe Stop-Words →
                      POS-guided Lemmatization
  Vectorization    : TF-IDF (uni+bigrams, max_features bounded)
  Storage          : SciPy CSR sparse matrix
  Model            : Multinomial / Complement Naive Bayes (+SVM)
─────────────────────────────────────────────────────────────
  Naive Bayes (Complement) →  Accuracy: ~0.90  |  F1: ~0.89
  SVM (Linear Kernel)      →  Accuracy: ~0.92  |  F1: ~0.91
─────────────────────────────────────────────────────────────
  Primary Metric   : F1-Score / Accuracy on held-out reviews ✅
  Negation Handling : Verified via "not happy" test case ✅
═══════════════════════════════════════════════════════════════
```

---

## 📚 Key Skills Demonstrated

```
✅ Natural Language Processing (NLTK/SpaCy)
✅ Text Pre-Processing Pipelines (Tokenization, Stop-Words, Lemmatization)
✅ Negation-Aware Stop-Word Filtering
✅ POS-Guided Morphological Lemmatization (WordNetLemmatizer)
✅ TF-IDF Vectorization (Term Frequency–Inverse Document Frequency)
✅ N-Gram Feature Engineering (Unigrams + Bigrams)
✅ Sparse Matrix Optimization (SciPy CSR format)
✅ Multinomial & Complement Naive Bayes Classification
✅ Laplace (Additive) Smoothing for Zero-Frequency Handling
✅ SVM Classification for Text
✅ Unstructured Data Handling at Scale
```

---

## 🏢 About

| | |
|--|--|
| **Program** | Data Science Industrial Training Kit |
| **Project** | Project 4 — Optional Mastery Phase |
| **Track** | NLP & Sentiment Analysis |
| **Organization** | DecodeLabs |
| **Batch** | 2026 |
| **Location** | Greater Lucknow, India |
| **Contact** | decodelabs.tech@gmail.com |
| **Website** | [www.decodelabs.tech](https://www.decodelabs.tech) |

---

## 📜 License

This project is part of the DecodeLabs Industrial Training Program and is submitted for educational and certification purposes.

---

<div align="center">

**Made with ❤️ | DecodeLabs Batch 2026**

*"Machine learning does not operate on raw text. It operates on mathematics. This pipeline is the strict assembly line that bridges the two."*

</div>
