🧬 DNA Sequence Classification System

A bioinformatics web application that classifies DNA sequences using Machine Learning, combining synthetic data generation and real genomic FASTA data analysis.

---

## 🚀 Project Overview

This project analyzes DNA sequences and performs classification across **three distinct biological tasks**:

### 🧪 1. G vs C Classification
* 🟢 **G-Rich Sequences** → High Guanine dominance  
* 🔵 **C-Rich Sequences** → High Cytosine dominance  
* Uses a **threshold-based filtering approach** to avoid ambiguous sequences

---

### 🧬 2. Promoter vs Non-Promoter Detection
* 🧾 Detects regulatory regions using **TATA motif patterns**
* Highlights challenges of **class imbalance in biological datasets**

---

### 🌍 3. Species Classification (Real Data)
* Uses real genomic **FASTA sequences**
* Classifies sequences into:
  * Human
  * Chimpanzee
  * Mouse
  * Macaque
* Demonstrates real-world **bioinformatics + ML integration**

---

## 🧠 Key Features

* 🧬 Multi-mode DNA classification system
* 🔍 K-mer based feature extraction (k = 3)
* 📊 GC-content analysis for biological interpretation
* 📈 Interactive visualizations:
  * Class distribution
  * GC content histogram
  * Confusion matrix (normalized)
* ⚡ Real-time prediction via Streamlit interface
* 📁 FASTA file upload support
* 🧠 Handles:
  * Class imbalance
  * Overfitting control
  * Data leakage reduction

---

## 🏗️ Tech Stack

* **Language:** Python  
* **Libraries:**
  * Streamlit
  * Pandas
  * Scikit-learn
  * Matplotlib
  * Seaborn
* **ML Model:** Random Forest Classifier

---

## ⚙️ Methodology

### 1. Dataset Preparation

#### Synthetic Data:
* Random DNA sequences generated
* Threshold-based labeling for G/C classification to remove ambiguity

#### Real Data:
* FASTA files loaded and parsed
* DNA sequences split into **non-overlapping chunks** to reduce data leakage

---

### 2. Feature Engineering

* DNA converted into **k-mers (substrings of length 3)**
* Text-based representation using **CountVectorizer**

---

### 3. Model Training

* Data split using **stratified sampling**
* Random Forest model trained with:
  * Class balancing
  * Depth control to prevent overfitting

---

### 4. Evaluation

* Accuracy score calculated
* Confusion matrix (normalized) used for performance interpretation
* Model behavior analyzed for:
  * Bias
  * Generalization
  * Class imbalance effects

---

### 5. Biological Interpretation

* GC-content computed for all sequences
* Provides insights into:
  * Structural stability
  * Species variation
  * Functional regions (e.g., promoters)

---

## 📊 Results & Observations

* ✔ Realistic accuracy achieved (~80–90%) depending on mode  
* ✔ Model avoids overfitting through controlled training  
* ⚠ Promoter detection shows bias due to class imbalance  
* ⚠ Species classification may show high accuracy on small datasets (limited generalization)

> 📌 Note: Results are influenced by dataset size and biological complexity. Real-world genomic data requires larger datasets for robust performance.

---

## 🧪 Sample Input


ATGCGTACGTTAGC


### Output:

* Predicted Class: **Non-Promoter**
* GC Content: ~50%

---

## 🌐 Live Demo

👉 *(Add your Streamlit deployment link here)*

---

## 💡 Future Improvements

* 🔬 Expand real genomic datasets (NCBI integration)
* 🤖 Implement Deep Learning models (CNN/RNN for sequence learning)
* 📊 Add Precision-Recall and ROC curves
* ⚖ Improve handling of class imbalance
* 🧬 Multi-label and hierarchical classification
* 🎨 Enhanced UI/UX design

---

## 👩‍💻 Authors

* **Bristi Sen**
* **Sambriddhi Debnath**
* **Rajjyashree Raychaudhuri**

---

## 🏁 Conclusion

This project demonstrates the application of Machine Learning in bioinformatics by integrating sequence analysis, feature engineering, and predictive modeling.

It highlights key real-world challenges such as:
* Data imbalance  
* Overfitting  
* Limited biological datasets  

and provides a strong foundation for more advanced genomic analysis systems.

---
