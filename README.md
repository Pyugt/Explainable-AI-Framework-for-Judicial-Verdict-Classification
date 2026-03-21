# Explainable AI Framework for Judicial Verdict Classification

An AI-powered framework for predicting and explaining Supreme Court of India verdicts using machine learning and Explainable AI (XAI) techniques.

---

## Overview

This project presents a transparent, legally justifiable system for classifying Indian Supreme Court judgments into three verdict categories — **Convicted**, **Acquitted**, and **Remanded** — using classical machine learning combined with multiple XAI layers. The goal is not just accurate prediction, but interpretable and auditable reasoning that aligns with Indian statutory law.

The framework was developed as part of a research paper submitted to IEEE.

---

## Features

- **Multi-model evaluation** across 6 classifiers (LR, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- **TF-IDF attention** to highlight information-dense sentences in judgments
- **SHAP-based feature attribution** to identify the most influential legal terms per prediction
- **Counterfactual analysis** to test prediction robustness via sentence ablation
- **IPC rule consistency checks** to validate predictions against Indian Penal Code statutes

---

## Dataset

- **Source:** Supreme Court of India judgments (scraped from the official Supreme Court Reports via the Judicial Information System)
- **Coverage:** 2,080 case documents spanning 2000–2025
- **Labels:** Convicted, Acquitted, Remanded (extracted via keyword pattern matching)
- **Final labeled records:** 988 (after filtering 1,064 unknown/ambiguous cases)

---

## Methodology

### Preprocessing
- Verdict labels extracted using keyword frequency matching (e.g., *"convicted"*, *"acquitted"*, *"remanded"*)
- IPC section references identified via regex (e.g., `Section 302`, `u/s 376`)
- Text cleaned: whitespace normalization, non-English character removal, sentence segmentation

### Feature Representation
- **TF-IDF vectors** with 5,000 max features, unigrams + bigrams, sublinear TF scaling

### Models Evaluated

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.62 | 0.52 | 0.63 |
| SVM | 0.71 | 0.61 | 0.69 |
| Random Forest | 0.77 | 0.68 | 0.74 |
| Gradient Boosting | 0.92 | 0.90 | 0.92 |
| **XGBoost** | **0.92** | **0.91** | **0.92** |
| LightGBM | 0.91 | 0.90 | 0.91 |

**XGBoost** was selected as the best-performing model and used for all downstream explainability analysis.

### Explainability Layers

1. **TF-IDF Attention** — Surfaces high-IDF sentences likely to contain legally discriminative content
2. **SHAP Attribution** — Computes per-feature contribution scores; categorized as Highly Positive → Highly Negative
3. **Counterfactual Analysis** — Removes sentences one at a time to find those that flip the predicted verdict
4. **Rule Consistency** — Verifies predictions against domain-specific IPC rules (e.g., IPC 302, IPC 376)

---

## Installation

```bash
git clone https://github.com/prnjxl/Explainable-AI-Framework-for-Judicial-Verdict-Classification.git
cd Explainable-AI-Framework-for-Judicial-Verdict-Classification
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the full pipeline
python src/main.py --input data/judgments/ --model xgboost --explain

# Run explainability on a single judgment
python src/explainability/shap_explainer.py --file data/sample_case.txt --class convicted
```

---

## Results

The XGBoost model achieves **92% accuracy** and a **weighted F1-score of 0.92** on the test set. The XAI analysis found that approximately **3.8% of cases** contained a single counterfactual sentence — meaning a verdict depended critically on one piece of evidence. SHAP analysis also revealed a correlation of **0.42** between IPC Section 376 references and gender-related terms, flagging a potential area of algorithmic bias.

---

## Ethical Considerations

This framework is designed as a **decision-support tool**, not a replacement for judicial reasoning. Human judges retain full authority. The system flags high-uncertainty predictions (confidence < 0.2) for mandatory human review, and SHAP attributions are used to surface potential bias rather than entrench it. The framework respects Articles 14 (equality) and 21 (due process) of the Indian Constitution.


