# 🧬 DNA Sequence Classification System

### 🚀 Hybrid Bioinformatics + Machine Learning System for DNA Intelligence

An interactive, research-oriented web application for DNA sequence classification, combining computational intelligence with biological insight.

🔗 **Live App:**
👉 https://dna-sequence-classifier-h2rdkshdw6ux49fj4dn5uv.streamlit.app/

---

## ✨ Overview

This project presents a **machine learning-based DNA sequence analysis platform** built using **Streamlit and Scikit-learn**, designed to explore how computational models can identify patterns in genomic data.

The system supports multiple classification tasks:

* 🧪 **G vs C Content Classification**
* 🧬 **Promoter vs Non-Promoter Detection**
* 🌍 **Species Classification using Real Genomic Data**

💡 This system goes beyond static classification by enabling **mutation impact analysis, sequence-level experimentation, and interpretable biological insights**, bridging the gap between computational models and real-world bioinformatics workflows.

---

## 🎯 Key Features

### 🔬 Machine Learning Pipeline

* K-mer based feature extraction (sequence → numerical representation)
* Hybrid feature engineering: **k-mer frequency + GC content integration**
* Multiple classification models:

  * 🌲 Random Forest (primary model)
  * 📈 Logistic Regression (baseline)
  * 📊 Naive Bayes (baseline)
* Stratified train-test splitting
* Class balancing for fair model training

---

### 📊 Model Evaluation & Validation

* Accuracy metrics for all models
* Model comparison table (research-oriented)
* Confusion Matrix visualization
* Classification Report (precision, recall, F1-score)
* Cross-validation with adaptive fold selection
* ROC Curve for multi-class evaluation

---

### 🧠 Interpretability & Explainability

* Feature importance (top k-mers driving predictions)
* Confidence scores for predictions
* Error analysis (misclassified samples)
* GC content contribution to predictions
* Biological interpretation of learned patterns

---

### 🧬 Biological Insights

* GC content analysis (DNA stability indicator)
* Promoter motif detection:

  * TATA box
  * CAAT box
  * TTGACA sequence
* Evolutionary interpretation in species classification
* Sequence-level pattern recognition using k-mer distributions

---

### ⚡ Interactive User Experience

* Upload DNA sequences (FASTA / TXT)
* Manual sequence input
* Batch prediction mode
* Adjustable parameters:

  * K-mer size
  * Dataset size
  * GC sensitivity threshold

---

### 🧪 Experimental Bioinformatics Tools

The system includes an experimental lab for sequence-level analysis:

* 🔍 Sequence similarity comparison
* 🧬 Motif detection and localization
* 🧪 Mutation simulation with classification impact analysis
* 🧵 Global sequence alignment (Needleman-Wunsch)
* ✂️ Restriction enzyme site identification
* 🧫 Primer design with GC% and melting temperature
* 📈 GC window analysis (local stability profiling)

---

## 🧪 Technologies Used

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas & NumPy**
* **Matplotlib & Seaborn**

---

## 📁 Dataset

### Synthetic Dataset

Generated dynamically using:

* Random nucleotide sequences (A, T, G, C)
* Controlled GC distribution
* Embedded biological motifs

---

### Real Dataset

FASTA-based genomic sequences:

* Human
* Chimpanzee
* Mouse
* Macaque

Sequences are segmented and processed for classification tasks.

---

## 🧠 Methodology

1. DNA sequences are segmented into **k-mers (substrings of length k)**
2. K-mers are treated like "words" and vectorized using **CountVectorizer**
3. A **hybrid feature vector (k-mer + GC content)** is constructed
4. Machine learning models learn patterns in sequence composition
5. Predictions are generated along with:

   * Confidence scores
   * Biological insights
6. Model performance is validated using multiple evaluation metrics

---

## 📊 Example Outputs

* DNA Base Composition Graph
* GC Content Distribution
* Confusion Matrix
* ROC Curve
* Feature Importance Visualization
* Model Comparison Table
* Mutation Impact Analysis (before vs after mutation)

---

## ⚠️ Limitations

* Synthetic data may not fully capture biological complexity
* Promoter detection is motif-based and not position-aware
* Limited real-world genomic dataset size
* Performance depends on parameter tuning and data quality

---

## 🚀 Future Improvements

* Deep learning models (CNN, LSTM for sequence learning)
* Integration with real biological databases (e.g., EPD, NCBI)
* Position-aware promoter detection
* Advanced feature engineering (embedding-based methods)
* API integration for external genomic data
* Deployment with user authentication and storage

---

## 👩‍💻 Developers

* **Bristi Sen**
* **Sambriddhi Debnath**
* **Rajjyashree Raychaudhuri**

---

## 💅 Connect

* 🔗 LinkedIn: https://www.linkedin.com/in/bristi-sen-709548311/
* 💻 GitHub: https://github.com/BristiSen

---

## ⭐ Why This Project Matters

This project demonstrates:

* Integration of **machine learning with biological sequence analysis**
* Development of an **interpretable and interactive bioinformatics system**
* Use of **hybrid feature engineering (statistical + biological features)**
* Simulation of **real-world laboratory workflows** (mutation, alignment, primer design)
* Emphasis on **explainability in biological predictions**

It represents a step toward making **computational genomics more accessible, interactive, and experimentally meaningful**.

---

## 📌 How to Run Locally

```bash
git clone https://github.com/BristiSen/DNA-Sequence-Classifier.git
cd dna-classification
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧬 Final Note

This project is a step toward bridging biology and artificial intelligence, highlighting how computational tools can assist in decoding complex genetic patterns.

⭐ If you found this interesting, consider starring the repository!
