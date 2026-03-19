# 🧬 DNA Sequence Classification System

A bioinformatics web application that classifies DNA sequences based on nucleotide composition using Machine Learning.

---

## 🚀 Project Overview

This project focuses on analyzing DNA sequences and classifying them into:

* 🟢 **G-Rich Sequences** → Higher Guanine (G) content
* 🔵 **C-Rich Sequences** → Higher Cytosine (C) content

The system uses **k-mer based feature extraction** and a **Random Forest Classifier** to perform sequence classification, along with interactive visualizations and real-time prediction via a web interface.

---

## 🧠 Key Features

* 🧬 DNA sequence classification using Machine Learning
* 🔍 K-mer based feature extraction (k = 3)
* 📊 GC-content analysis for biological interpretation
* 📈 Interactive visualizations:

  * Class distribution
  * GC content histogram
  * Confusion matrix
* ⚡ Real-time prediction using Streamlit
* ⬇ Downloadable dataset
* 🌐 Deployable web application

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

1. **Dataset Generation**

   * Synthetic DNA sequences generated using probabilistic nucleotide distribution.

2. **Feature Engineering**

   * DNA sequences converted into **k-mers (substrings of length 3)**
   * Vectorized using **CountVectorizer**

3. **Model Training**

   * Data split into training and testing sets
   * Random Forest classifier trained on k-mer features

4. **Evaluation**

   * Accuracy score calculated
   * Confusion matrix used for performance visualization

5. **Biological Interpretation**

   * GC-content calculated for each sequence
   * Provides insights into structural and functional properties of DNA

---

## 📊 Results

* Achieved high classification accuracy on synthetic dataset
* Model effectively distinguishes between G-rich and C-rich sequences
* Visualizations confirm balanced dataset and strong predictive performance

> ⚠️ Note: The dataset is synthetically generated, so performance may vary on real-world biological data.

---

## 🧪 Sample Input

```
ATGCGTACGTTAGC
```

### Output:

* Predicted Class: **G-Rich Sequence**
* GC Content: ~60%

---

## 🌐 Live Demo

👉 *(Add your Streamlit deployment link here once deployed)*

---

## 💡 Future Improvements

* 🔬 Integration with real-world genomic datasets (NCBI, FASTA files)
* 🤖 Deep Learning models (CNN/RNN for sequence analysis)
* 🧬 Multi-class classification (beyond G/C richness)
* 🎨 Enhanced UI/UX design
* 📁 File upload for batch predictions

---

## 👩‍💻 Authors

* **Bristi Sen**
* **Sambriddhi Debnath**
* **Rajjyashree Raychaudhuri**

---

## 🏁 Conclusion

This project demonstrates how Machine Learning can be applied to biological sequence analysis, bridging the gap between computer science and biotechnology. It serves as a foundational model for more advanced genomic prediction systems.

---
