# 🧬 DNA Sequence Classification System

A bioinformatics web application that analyzes and classifies DNA sequences using Machine Learning, while also providing real-time biological insights from user input.

---

## 🚀 Project Overview

This project enables classification and analysis of DNA sequences across multiple biological perspectives:

### 🧬 Modes of Classification

* 🟢 **G vs C Classification**

  * Classifies sequences based on Guanine (G) and Cytosine (C) dominance

* 🔴 **Promoter vs Non-Promoter Detection**

  * Uses biologically relevant motifs (TATA, CAAT, TTGACA)
  * Provides **promoter likelihood (%)** of the input sequence

* 🌍 **Species Classification (Real Data)**

  * Classifies sequences into:

    * Human
    * Chimpanzee
    * Mouse
    * Macaque
  * Displays **species similarity percentages**

---

## 🧠 Key Features

* 🧬 Multi-mode DNA classification system

* 🔍 K-mer based feature extraction (k = 3)

* 🤖 Machine Learning using Random Forest Classifier

* 📊 Real-time input-based analysis and visualization

* 📈 Interactive graphs:

  * DNA base composition (A, T, G, C)
  * G vs C comparison (color-coded)
  * Promoter likelihood distribution
  * Species similarity visualization
  * GC-content distribution
  * Confusion matrix

* ⚡ Instant prediction from user input

* 📁 Supports manual input and FASTA file upload

---

## 🏗️ Tech Stack

* **Language:** Python

* **Framework:** Streamlit

* **Libraries:**

  * Pandas
  * Scikit-learn
  * Matplotlib
  * Seaborn

* **Model:** Random Forest Classifier

---

## ⚙️ Methodology

### 1. Dataset Generation

* Synthetic DNA sequences are generated for training (except species mode)
* Balanced datasets used for better classification

---

### 2. Feature Engineering

* DNA sequences converted into **k-mers (substrings of length 3)**
* Vectorized using **CountVectorizer**

---

### 3. Model Training

* Train-test split with stratification
* Random Forest classifier trained on k-mer features

---

### 4. Prediction

* User input sequence is transformed into k-mers
* Model predicts classification based on trained patterns

---

### 5. Biological Analysis (Input-Based)

#### 🧬 GC Content

* Measures stability of DNA sequence
* Higher GC → more stable structure

#### 🧬 Base Composition

* Displays counts of A, T, G, C in the input sequence

#### 🧬 Promoter Detection (Improved Logic)

* Uses motif-based scoring:

  * TATA (weight 3)
  * CAAT (weight 2)
  * TTGACA (weight 3)
* Produces a **Promoter Likelihood (%)**

#### 🧬 Species Similarity

* Uses model probability outputs
* Displays similarity percentage across species

---

## 📊 Results

* High accuracy achieved on synthetic datasets
* Balanced training improves classification stability
* Real-time visualizations provide intuitive biological insights
* Model predictions align with nucleotide composition patterns

> ⚠️ Note: Synthetic datasets are used for training in some modes, so real-world accuracy may vary.

---

## 🧪 Sample Input

```
ATGCGTACGTTAGC
```

### Output:

* Prediction: **G-Rich Sequence**
* GC Content: ~60%
* Visual graphs showing composition and classification

---

## 🌐 Live Demo

👉 *(Add your Streamlit deployment link here)*

---

## 💡 Future Improvements

* 🔬 Integration with larger real-world genomic datasets (NCBI)
* 🤖 Deep learning models (CNN/RNN for sequence analysis)
* 🧬 Expanded species classification
* 📊 Advanced biological feature extraction
* 🎨 Enhanced UI/UX design

---

## 👩‍💻 Authors

* **Bristi Sen**
* **Sambriddhi Debnath**
* **Rajjyashree Raychaudhuri**

---

## 🏁 Conclusion

This project demonstrates how Machine Learning can be applied to DNA sequence analysis while combining computational techniques with biological interpretation. It serves as a strong foundation for more advanced genomic prediction systems.

---
