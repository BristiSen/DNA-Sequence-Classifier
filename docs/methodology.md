# 🧬 Methodology: DNA Sequence Classification System

## 1. Introduction

This project presents a machine learning-based approach for DNA sequence classification using bioinformatics-inspired feature engineering. The system is designed to classify DNA sequences across three distinct biological contexts:

* **G vs C Content Classification**
* **Promoter vs Non-Promoter Detection**
* **Species Classification using Real Genomic Data**

The methodology integrates biological insights (such as GC content and promoter motifs) with machine learning techniques to build an interpretable and flexible classification system.

---

## 2. Data Generation and Collection

### 2.1 Synthetic Dataset

For controlled experimentation, synthetic DNA sequences are generated using random sampling from the nucleotide set:

> {A, T, G, C}

Each sequence is generated with a configurable length (default: 60 bases).

#### Labeling Strategy

* **G vs C Classification**
  Sequences are labeled based on the relative dominance of Guanine (G) and Cytosine (C):

  * G-rich: if (G − C) / length > threshold
  * C-rich: if (G − C) / length < −threshold

* **Promoter Classification**
  Sequences are labeled as:

  * Promoter: if biological motifs like *TATA* are present
  * Non-Promoter: otherwise

---

### 2.2 Real Dataset (FASTA Files)

For species classification, real genomic data is loaded from FASTA files corresponding to:

* Human
* Chimpanzee
* Mouse
* Macaque

Sequences are segmented into fixed-length chunks (100 bases) to create a uniform dataset for training.

---

## 3. Feature Engineering

### 3.1 GC Content

GC content is calculated as:

GC = (Number of G + Number of C) / Total Length

This feature provides biological insight into:

* DNA stability
* Structural properties

---

### 3.2 K-mer Representation

DNA sequences are transformed into k-mer tokens (substrings of length *k*).

Example (k = 3):
Sequence: ATGCG
→ ATG, TGC, GCG

These k-mers are treated as "words" and vectorized using:

> **CountVectorizer (Bag-of-Words approach)**

This converts biological sequences into numerical feature vectors suitable for machine learning.

---

## 4. Model Selection

A **Random Forest Classifier** is used due to:

* Ability to handle high-dimensional data (k-mers)
* Robustness to noise
* Interpretability via feature importance

### Model Parameters

* Number of trees: 150
* Max depth: 10
* Class balancing: Enabled

---

## 5. Training Strategy

### 5.1 Train-Test Split

The dataset is split using stratified sampling to preserve class distribution:

* Test size dynamically adjusted (25%–40%)
* Random state fixed for reproducibility

---

### 5.2 Cross-Validation

Cross-validation is applied with adaptive folds:

* Number of folds = min(5, smallest class size)

This ensures:

* Stability in evaluation
* Avoidance of class imbalance issues

---

## 6. Evaluation Metrics

The model is evaluated using multiple metrics:

### 6.1 Accuracy

Overall correctness of predictions.

---

### 6.2 Classification Report

Includes:

* Precision
* Recall
* F1-score

Provides class-wise performance insights.

---

### 6.3 Confusion Matrix

Visual representation of:

* True vs predicted labels
* Misclassification patterns

---

### 6.4 ROC Curve (where applicable)

Evaluates model discrimination ability using:

* True Positive Rate (TPR)
* False Positive Rate (FPR)
* Area Under Curve (AUC)

---

## 7. Model Interpretability

### 7.1 Feature Importance

Top k-mer features influencing predictions are extracted using:

> Random Forest feature importance scores

This provides insight into:

* Sequence patterns driving classification
* Biological relevance of motifs

---

## 8. Additional Functionalities

### 8.1 Sequence Mutation

Random mutations simulate biological variation and test model robustness.

---

### 8.2 Batch Prediction

Multiple sequences can be evaluated simultaneously.

---

### 8.3 Biological Insight Layer

Based on GC content:

* High GC → Increased stability
* Low GC → Increased flexibility

---

### 8.4 Sequence Similarity (Heuristic)

Basic similarity comparison with dataset sequences to provide contextual understanding.

---

## 9. Limitations

* Synthetic data may not fully capture biological complexity
* Promoter detection relies on simple motif matching
* Limited species dataset (4 organisms)
* K-mer approach ignores long-range dependencies

---

## 10. Future Improvements

* Integration of deep learning (CNNs / LSTMs)
* Use of larger genomic datasets
* Advanced motif detection algorithms
* Multi-class ROC support improvements
* Deployment as a web-based bioinformatics tool

---

## 11. Conclusion

This project demonstrates how machine learning techniques can be effectively combined with biological insights to analyze DNA sequences. The system serves as both an educational tool and a foundational step toward more advanced bioinformatics applications.
