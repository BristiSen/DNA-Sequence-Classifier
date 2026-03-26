# 🧬 DNA Sequence Classification System

### 🚀 Bioinformatics meets Machine Learning

An interactive, research-oriented web application for DNA sequence classification, combining computational intelligence with biological insight.

🔗 **Live App:**  
👉 https://dna-sequence-classifier-h2rdkshdw6ux49fj4dn5uv.streamlit.app/

---

## ✨ Overview

This project presents a **machine learning-based DNA sequence analysis platform** built using **Streamlit and Scikit-learn**, designed to explore how computational models can identify patterns in genomic data.

The system supports multiple classification tasks:

- 🧪 **G vs C Content Classification**
- 🧬 **Promoter vs Non-Promoter Detection**
- 🌍 **Species Classification using Real Genomic Data**

Beyond prediction, the application emphasizes **interpretability, biological reasoning, and comparative model evaluation**, making it both a practical tool and a learning framework.

---

## 🎯 Key Features

### 🔬 Machine Learning Pipeline

- K-mer based feature extraction (sequence → numerical representation)
- Multiple classification models:
  - 🌲 Random Forest (primary model)
  - 📈 Logistic Regression (baseline)
  - 📊 Naive Bayes (baseline)
- Stratified train-test splitting
- Class balancing for fair model training

---

### 📊 Model Evaluation & Validation

- Accuracy metrics for all models
- Model comparison table (research-oriented)
- Confusion Matrix visualization
- Classification Report (precision, recall, F1-score)
- Cross-validation with adaptive fold selection
- ROC Curve for multi-class evaluation

---

### 🧠 Interpretability & Explainability

- Feature importance (top k-mers driving predictions)
- Confidence scores for predictions
- Error analysis (misclassified samples)
- Biological interpretation of learned patterns

---

### 🧬 Biological Insights

- GC content analysis (DNA stability indicator)
- Promoter motif detection:
  - TATA box
  - CAAT box
  - TTGACA sequence
- Evolutionary interpretation in species classification:
  - Highlights similarity between closely related species (e.g., Human vs Chimpanzee)

---

### ⚡ Interactive User Experience

- Upload DNA sequences (FASTA / TXT)
- Manual sequence input
- Batch prediction mode
- Mutation simulation tool 🧪
- Adjustable parameters:
  - K-mer size
  - Dataset size
  - GC sensitivity threshold

---

## 🧪 Technologies Used

- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**

---

## 📁 Dataset

### Synthetic Dataset

Generated dynamically using:

- Random nucleotide sequences (A, T, G, C)
- Controlled GC distribution
- Embedded biological motifs

---

### Real Dataset

FASTA-based genomic sequences:

- Human
- Chimpanzee
- Mouse
- Macaque

Sequences are segmented and processed for classification tasks.

---

## 🧠 Methodology

1. DNA sequences are segmented into **k-mers (substrings of length k)**
2. K-mers are treated like "words" and vectorized using **CountVectorizer**
3. Machine learning models learn patterns in k-mer distributions
4. Predictions are generated along with:
   - Confidence scores
   - Biological insights
5. Model performance is validated using multiple evaluation metrics

---

## 📊 Example Outputs

- DNA Base Composition Graph
- GC Content Distribution
- Confusion Matrix
- ROC Curve
- Feature Importance Visualization
- Model Comparison Table

---

## ⚠️ Limitations

- Synthetic data may not fully capture biological complexity
- Promoter detection is motif-based and not position-aware
- Limited real-world genomic dataset size
- Performance depends on parameter tuning and data quality

---

## 🚀 Future Improvements

- Deep learning models (CNN, LSTM for sequence learning)
- Integration with real biological databases (e.g., EPD, NCBI)
- Position-aware promoter detection
- Advanced feature engineering (embedding-based methods)
- API integration for external genomic data
- Deployment with user authentication and storage

---

## 👩‍💻 Developers

- **Bristi Sen**
- **Sambriddhi Debnath**
- **Rajjyashree Raychaudhuri**

---

## 💅 Connect

- 🔗 LinkedIn: https://www.linkedin.com/in/bristi-sen-709548311/
- 💻 GitHub: https://github.com/BristiSen

---

## ⭐ Why This Project Matters

This project demonstrates:

- Application of machine learning in **bioinformatics**
- Development of an **end-to-end ML system**
- Emphasis on **interpretability and explainability**
- Integration of **biological knowledge with AI models**
- Comparative evaluation of multiple ML approaches

---

## 📌 How to Run Locally

```bash
git clone https://github.com/BristiSen/DNA-Sequence-Classifier.git
cd dna-classification
pip install -r requirements.txt
streamlit run app.py
