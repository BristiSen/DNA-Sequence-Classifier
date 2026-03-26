#Hello there# 

# 🧬 DNA Sequence Classification System

### 🚀 Bioinformatics meets Machine Learning

An interactive web application that classifies DNA sequences using machine learning techniques, combining biological insights with computational intelligence.

---

## ✨ Overview

This project is a **DNA sequence analysis platform** built using **Streamlit and Scikit-learn**, designed to:

* Classify DNA sequences based on:

  * 🧪 **G vs C content dominance**
  * 🧬 **Promoter vs Non-promoter regions**
  * 🌍 **Species classification using real genomic data**
* Provide **interpretability and biological insights**
* Enable **interactive experimentation** with sequence data

---

## 🎯 Key Features

### 🔬 Machine Learning Capabilities

* K-mer based feature extraction
* Random Forest Classification
* Stratified train-test splitting
* Class balancing for fair learning

### 📊 Model Evaluation

* Accuracy metrics
* Confusion Matrix (with labels)
* Classification Report (precision, recall, F1-score)
* Cross-validation with dynamic fold selection
* ROC Curve (where applicable)

### 🧠 Interpretability

* Top feature importance (k-mers driving predictions)
* Confidence score for predictions
* Error analysis (misclassified samples)

### 🧬 Biological Insights

* GC content analysis
* Promoter motif detection (TATA, CAAT, TTGACA)
* DNA stability interpretation

### ⚡ Interactive UI

* Upload FASTA or text sequences
* Batch prediction support
* Sequence mutation simulation
* Adjustable parameters:

  * K-mer size
  * Dataset size
  * GC sensitivity threshold

---

## 🧪 Technologies Used

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas & NumPy**
* **Matplotlib & Seaborn**

---

## 📁 Dataset

### Synthetic Data

Generated dynamically using:

* Random nucleotide sequences
* Controlled GC distribution
* Embedded promoter motifs

### Real Data

FASTA-based genomic sequences:

* Human
* Chimpanzee
* Mouse
* Macaque

---

## 🧠 How It Works

1. DNA sequences are broken into **k-mers** (substrings of length k)
2. K-mers are vectorized using **CountVectorizer**
3. A **Random Forest classifier** learns patterns in sequence distribution
4. Predictions are made with:

   * Class label
   * Probability/confidence score
5. Additional insights are generated using biological heuristics

---

## 📊 Example Outputs

* DNA Composition Graph
* GC Content Distribution
* Confusion Matrix
* ROC Curve
* Feature Importance Visualization

---

## ⚠️ Limitations

* Synthetic dataset may not fully represent biological complexity
* Small datasets can affect:

  * Cross-validation reliability
  * ROC curve generation
* Model performance depends on parameter tuning

---

## 🚀 Future Improvements

* Deep Learning models (CNN/RNN for sequence analysis)
* Larger real-world genomic datasets
* Advanced motif detection
* API integration for external datasets
* Deployment with user authentication

---

## 👩‍💻 Developers

* **Bristi Sen**
* **Sambriddhi Debnath**
* **Rajjyashree Raychaudhuri**

---

## 💅 Connect With Me

* 🔗 [LinkedIn](https://www.linkedin.com/in/bristi-sen-709548311/)
* 💻 [GitHub](https://github.com/BristiSen)

---

## ⭐ Why This Project Matters

This project demonstrates:

* Application of machine learning in **bioinformatics**
* Ability to build **end-to-end data science systems**
* Strong focus on **interpretability and user experience**
* Integration of **domain knowledge with AI**

---

## 📌 How to Run

```bash
git clone https://github.com/BristiSen/DNA-Sequence-Classifier.git
cd dna-classification
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧬 Final Note

This project is a step toward bridging **biology and artificial intelligence**, showing how computational tools can assist in understanding complex biological patterns.

---

⭐ If you found this interesting, feel free to star the repo!
