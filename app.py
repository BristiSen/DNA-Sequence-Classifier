import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score


# ================== 🔥 AESTHETIC ENHANCEMENTS ==================
st.markdown("## ✨ Welcome to your DNA Intelligence Lab")
st.caption("Analyze. Predict. Understand. — Like a bioinformatics pro.")

st.divider()
# ------------------ SIDEBAR ------------------

with st.sidebar:
    st.header("⚙️ Settings")
    k_value = st.slider("K-mer Size", 2, 6, 3)
    seq_length = st.slider("Synthetic Sequence Length", 40, 120, 60)
    dataset_size = st.slider("Synthetic Dataset Size", 200, 1000, 600)

    st.divider()
    threshold = st.slider("G-C Sensitivity Threshold", 0.05, 0.3, 0.15)

    st.markdown("---")
    st.caption("✨ Tune parameters like a real bioinformatics researcher")

# ------------------ PAGE TITLE ------------------

st.title("🧬 DNA Sequence Classification System")
st.markdown("### Bioinformatics Model for DNA Analysis")

st.divider()

# ================== 🔥 LAB MODE ==================

lab_mode = st.selectbox(
    "Select Lab Mode",
    ["🧬 Classification Lab", "🧪 Experimental Lab"]
)

if lab_mode == "🧬 Classification Lab":
    mode = st.selectbox(
        "Select Classification Mode",
        ["G vs C Classification", "Promoter vs Non-Promoter", "Species Classification (Real Data)"]
    )
else:
    mode = st.selectbox(
        "Select Experiment",
        ["Sequence Similarity", "Motif Finder", "Mutation Playground", "Sequence Alignment","Restriction Site Analysis","Primer Design", 
 "GC Window Analysis"]
    )
# ------------------ DESCRIPTION ------------------

if mode == "G vs C Classification":
    st.info("⚡ Detects dominance between Guanine (G) and Cytosine (C)")
elif mode == "Promoter vs Non-Promoter":
    st.info("🧬 Identifies biological promoter motifs like TATA box")
else:
    st.info("🌍 Uses real FASTA genomic data to match sequence patterns to known species genomes")

# ------------------ FUNCTIONS ------------------

def generate_sequence(length=60):
    return ''.join(random.choices(['A','T','G','C'], k=length))

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def mutate(seq, rate=0.1):
    bases = ['A','T','G','C']
    return ''.join([random.choice(bases) if random.random()<rate else c for c in seq])

# ================== 🔥 LAB FUNCTIONS ==================

def sequence_similarity(seq1, seq2):
    length = min(len(seq1), len(seq2))
    matches = sum(c1 == c2 for c1, c2 in zip(seq1[:length], seq2[:length]))
    return matches / length if length > 0 else 0

def find_motif_positions(seq, motif):
    positions = []
    for i in range(len(seq) - len(motif) + 1):
        if seq[i:i+len(motif)] == motif:
            positions.append(i)
    return positions

def simple_alignment(seq1, seq2):
    align1, align2 = "", ""
    score = 0

    for a, b in zip(seq1, seq2):
        align1 += a
        align2 += b
        if a == b:
            score += 1
        else:
            score -= 1

    return align1, align2, score

def restriction_sites(seq):
    enzymes = {
        "EcoRI": "GAATTC",
        "BamHI": "GGATCC",
        "HindIII": "AAGCTT"
    }

    results = {}

    for name, pattern in enzymes.items():
        positions = find_motif_positions(seq, pattern)
        results[name] = positions

    return results

def generate_primer(seq, length=20):
    return seq[:length], seq[-length:]

def gc_window(seq, window=20):
    values = []
    for i in range(len(seq) - window + 1):
        sub = seq[i:i+window]
        values.append(gc_content(sub))
    return values

# ------------------ FASTA LOADER ------------------

def load_fasta_file(filepath, label):
    sequences = []
    try:
        with open(filepath, "r") as f:
            seq = ""
            for line in f:
                if line.startswith(">"):
                    if seq:
                        sequences.append((seq, label))
                        seq = ""
                else:
                    seq += line.strip()
            if seq:
                sequences.append((seq, label))
    except:
        pass
    return sequences

def load_real_dataset():
    dataset_path = "dataset"
    data = []

    species_files = {
        "human.fasta": "Human",
        "chimp.fasta": "Chimpanzee",
        "mouse.fasta": "Mouse",
        "macaque.fasta": "Macaque"
    }

    for file, label in species_files.items():
        filepath = os.path.join(dataset_path, file)
        sequences = load_fasta_file(filepath, label)

        for seq, lbl in sequences:
            if len(seq) < 200:
                continue
            for i in range(0, len(seq) - 100, 100):
                data.append([seq[i:i+100], lbl])

    return pd.DataFrame(data, columns=["sequence", "label"])

# ------------------ DATA ------------------

if mode == "Species Classification (Real Data)":
    df = load_real_dataset()
else:
    data = []
    for _ in range(dataset_size):
        seq = generate_sequence(seq_length)

        if mode == "G vs C Classification":
            diff = (seq.count('G') - seq.count('C')) / len(seq)
            if diff > threshold:
                label = "G-Rich Sequence"
            elif diff < -threshold:
                label = "C-Rich Sequence"
            else:
                continue
        else:
            motifs = ["TATA", "CAAT", "TTGACA"]
            label = "Promoter" if any(m in seq for m in motifs) else "Non-Promoter"

        data.append([seq, label])

    df = pd.DataFrame(data, columns=["sequence", "label"])

# ------------------ BALANCE ------------------

if mode == "G vs C Classification" and len(df) > 0:
    df_g = df[df['label'] == "G-Rich Sequence"]
    df_c = df[df['label'] == "C-Rich Sequence"]

    min_size = min(len(df_g), len(df_c))

    if min_size > 0:
        df = pd.concat([
            df_g.sample(min_size, random_state=42),
            df_c.sample(min_size, random_state=42)
        ])

# ------------------ SAFETY ------------------

if len(df) < 10:
    st.error("Dataset too small to train model.")
    st.stop()

# ------------------ FEATURES ------------------

df["gc_content"] = df["sequence"].apply(gc_content)
df["kmers"] = df["sequence"].apply(lambda x: ' '.join(get_kmers(x, k_value)))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["kmers"])
y = df["label"]

num_classes = len(y.unique())

test_size = max(0.25, num_classes / len(df) + 0.05)
test_size = min(test_size, 0.4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    stratify=y,
    random_state=42
)

# ------------------ MODEL ------------------

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight='balanced',
    max_depth=10
)

model.fit(X_train, y_train)

# ================== 🔥 BASELINE MODELS ==================

baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

baseline_results = {}

for name, mdl in baseline_models.items():
    mdl.fit(X_train, y_train)
    pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, pred)
    baseline_results[name] = acc

# ------------------ EVAL ------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# 🔥 Add interpretation
if accuracy > 0.9:
    st.success("Model is performing extremely well 🎯")
elif accuracy > 0.75:
    st.info("Model performance is decent 👍")
else:
    st.warning("Model may need improvement ⚠️")

# ================== 🔥 MODEL COMPARISON ==================

st.subheader("📊 Model Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Random Forest"] + list(baseline_results.keys()),
    "Accuracy": [accuracy] + list(baseline_results.values())
})

st.dataframe(comparison_df.style.format({"Accuracy": "{:.2f}"}))

# ================== 🔥 CLASSIFICATION REPORT ==================
st.subheader("📊 Detailed Classification Metrics")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df.style.format("{:.2f}"))

# 🔥 NEW: CLASS DISTRIBUTION
st.subheader("📊 Training Data Distribution")
fig_dist, ax_dist = plt.subplots()
sns.countplot(x=df['label'], ax=ax_dist)
st.pyplot(fig_dist)

# ================== 🔥 CROSS VALIDATION ==================
st.subheader("🔁 Cross-Validation Score")

try:
    min_class_size = y.value_counts().min()
    cv_folds = min(5, min_class_size)

    if cv_folds < 2:
        st.warning("Not enough data for cross-validation.")
    else:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds)
        st.write(f"Mean CV Score: **{cv_scores.mean():.2f}**")
        st.write(f"Score Variability: ±{cv_scores.std():.2f}")

except Exception as e:
    st.warning("Cross-validation could not be computed for this dataset.")

# ================== 🧪 EXPERIMENTAL LAB ==================

if lab_mode == "🧪 Experimental Lab":

    st.subheader("🧪 Experimental DNA Tools")

    if mode == "Sequence Similarity":

        seq1 = st.text_input("Enter Sequence 1")
        seq2 = st.text_input("Enter Sequence 2")

        if st.button("Compare Sequences"):
            if seq1 and seq2:
                similarity = sequence_similarity(seq1.upper(), seq2.upper())
                st.metric("Similarity Score", f"{similarity*100:.2f}%")

                if similarity > 0.9:
                    st.success("Highly similar sequences 🧬")
                elif similarity > 0.6:
                    st.info("Moderate similarity")
                else:
                    st.warning("Low similarity")

    elif mode == "Motif Finder":

        seq = st.text_input("Enter DNA Sequence")
        motif = st.text_input("Enter Motif (e.g., TATA)")

        if st.button("Find Motif"):
            if seq and motif:
                positions = find_motif_positions(seq.upper(), motif.upper())

                if positions:
                    st.success(f"Motif found at positions: {positions}")
                else:
                    st.warning("Motif not found")

    elif mode == "Mutation Playground":

        seq = st.text_input("Enter DNA Sequence")

        if st.button("Mutate Sequence"):
            if seq:
                mutated = mutate(seq.upper())
                st.write("Original:", seq)
                st.write("Mutated:", mutated)

    elif mode == "Sequence Alignment":
        seq1_align = st.text_input("Enter Sequence 1")
        seq2_align = st.text_input("Enter Sequence 2")

        if st.button("Align Sequences"):
            if seq1 and seq2:
                a1, a2, score = simple_alignment(seq1.upper(), seq2.upper())

                st.text(a1)
                st.text(a2)
                st.metric("Alignment Score", score)

    elif mode == "Restriction Site Analysis":

        seq = st.text_input("Enter DNA Sequence")

        if st.button("Analyze Restriction Sites"):
            if seq:
                sites = restriction_sites(seq.upper())

                for enzyme, positions in sites.items():
                    st.write(f"🔬 {enzyme}: {positions if positions else 'No cut sites'}")

    elif mode == "Primer Design":

        seq = st.text_input("Enter DNA Sequence")

        if st.button("Generate Primers"):
            if seq:
                fwd, rev = generate_primer(seq.upper())

                st.write(f"Forward Primer: {fwd}")
                st.write(f"Reverse Primer: {rev}")

    elif mode == "GC Window Analysis":

        seq = st.text_input("Enter DNA Sequence")

        if st.button("Analyze GC Distribution"):
            if seq:
                values = gc_window(seq.upper())

                fig, ax = plt.subplots()
                ax.plot(values)
                ax.set_title("GC Content Across Sequence")
                ax.set_xlabel("Position")
                ax.set_ylabel("GC Ratio")

                st.pyplot(fig)
        window_size = st.slider("Window Size", 10, 50, 20)
        values = gc_window(seq.upper(), window=window_size)
    
    st.stop()
# ------------------ INPUT ------------------

st.subheader("🔬 Test Your DNA Sequence")

uploaded_file = st.file_uploader("Upload DNA File", type=["fasta", "txt"])
user_input = st.text_input("Enter DNA Sequence")

multi_input = st.text_area("Batch Input (optional, one per line)")

prediction = None
probs = None
confidence = None

if user_input:
    st.code(user_input[:100] + "...")
    st.write(f"📏 Sequence Length: **{len(user_input)} bases**")

if st.button("Predict"):

    if uploaded_file:
        lines = uploaded_file.read().decode("utf-8").split("\n")
        user_input = ''.join([l.strip() for l in lines if not l.startswith(">")])

    if user_input and all(c in "ATGC" for c in user_input):

        kmers = ' '.join(get_kmers(user_input, k_value))
        vector = vectorizer.transform([kmers])

        prediction = model.predict(vector)[0]
        prob_vals = model.predict_proba(vector)[0]
        confidence = max(prob_vals)

        if mode == "Species Classification (Real Data)":
            probs = prob_vals

        gc_val = gc_content(user_input)*100

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", prediction)
        col2.metric("GC Content", f"{gc_val:.2f}%")
        col3.metric("Confidence", f"{confidence*100:.2f}%")

        # 🔥 Confidence interpretation
        if confidence > 0.8:
            st.success("High confidence prediction ✅")
        elif confidence > 0.6:
            st.info("Moderate confidence 🤔")
        else:
            st.warning("Low confidence — interpret carefully ⚠️")


# ================== ANALYSIS ==================
if prediction:
    st.subheader("🧠 Deep Analysis")

    if mode == "G vs C Classification":
        st.write("""
This classification is based on **relative dominance of G vs C nucleotides**.

- Positive bias → G-rich  
- Negative bias → C-rich  
- Learned via k-mer frequency patterns
        """)

    elif mode == "Promoter vs Non-Promoter":
        st.write("""
Prediction uses **biological promoter motifs**:

- TATA box  
- CAAT box  
- TTGACA sequence  

Combined with machine learning pattern recognition.
        """)

    else:
        st.write("""
The model compares your sequence with **real genomic data** from:

- Human  
- Chimpanzee  
- Mouse  
- Macaque  

Using k-mer similarity patterns.
        """)
        st.write("""
💡 **Biological Insight:**

Closely related species (e.g., Human and Chimpanzee) often show similar k-mer patterns.
Misclassifications between such species may reflect evolutionary similarity rather than model error.
""")

# ------------------ EXPLAINABILITY ------------------

if prediction:
    st.subheader("🔍 Why this prediction?")

    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_

    top_features = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)[:5]

    for f, val in top_features:
        st.write(f"• {f} → {round(val,4)}")
        st.write("""
        💡 These k-mers may represent biologically significant sequence patterns.
        For example, GC-rich k-mers often indicate structurally stable regions of DNA.
        """)

# ================== 🔥 FEATURE IMPORTANCE GRAPH ==================
if prediction:
    st.subheader("📊 Top Feature Importance (Visual)")

    top_features_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(top_features_df["Feature"], top_features_df["Importance"])
    ax_imp.invert_yaxis()

    st.pyplot(fig_imp)

# ------------------ MUTATION ------------------

if user_input:
    if st.button("🧪 Mutate Sequence"):
        mutated = mutate(user_input)
        st.write("Mutated:", mutated)

# ================== BATCH ==================
if multi_input:
    st.subheader("📦 Batch Predictions")

    results = []
    for seq in multi_input.split("\n"):
        seq = seq.strip().upper()
        if seq and all(c in "ATGC" for c in seq):
            kmers = ' '.join(get_kmers(seq, k_value))
            vector = vectorizer.transform([kmers])
            pred = model.predict(vector)[0]
            results.append({"Sequence": seq[:20]+"...", "Prediction": pred})

    if results:
        st.dataframe(pd.DataFrame(results))

# ================== BIO INSIGHT ==================
if user_input:
    st.subheader("🧠 Biological Insight")

    gc_val = gc_content(user_input)*100

    if gc_val > 60:
        st.write("High GC → More stable DNA structure 🧬")
    elif gc_val < 40:
        st.write("Low GC → More flexible DNA")
    else:
        st.write("Moderate GC → Balanced structure")

# ================== 🔥 DATASET SUMMARY ==================
st.subheader("📦 Dataset Summary")

col1, col2 = st.columns(2)
col1.metric("Total Samples", len(df))
col2.metric("Unique Classes", len(df['label'].unique()))

# ------------------ ORIGINAL GRAPHS (ALL KEPT) ------------------

st.subheader("📊 DNA Composition (Your Input)")

if user_input:
    fig1, ax1 = plt.subplots()

    if mode == "G vs C Classification":
        ax1.bar(["G","C"],
                [user_input.count('G'), user_input.count('C')],
                color=["green","blue"])
    else:
        bases = ['A','T','G','C']
        ax1.bar(bases,
                [user_input.count(b) for b in bases],
                color=['orange','red','green','blue'])

    st.pyplot(fig1)

if user_input and mode == "Promoter vs Non-Promoter":
    motifs = {"TATA":3,"CAAT":2,"TTGACA":3}
    score = sum(weight for motif, weight in motifs.items() if motif in user_input)
    max_score = sum(motifs.values())

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(["Promoter","Non-Promoter"],
               [(score/max_score)*100, 100-(score/max_score)*100],
               color=["green","red"])
    st.pyplot(fig_bar)

if user_input and mode == "Species Classification (Real Data)" and probs is not None:
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(model.classes_, probs,
               color=["green","blue","purple","orange"])
    st.pyplot(fig_bar)

st.subheader("🧬 GC Content Distribution")
fig_gc, ax_gc = plt.subplots()
sns.histplot(df["gc_content"], bins=10, kde=True, ax=ax_gc)
st.pyplot(fig_gc)

st.subheader("📊 Confusion Matrix")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig2, ax2 = plt.subplots()

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ================== 🔥 ERROR ANALYSIS ==================
st.subheader("⚠️ Error Analysis")

errors = y_test != y_pred
error_count = errors.sum()

st.write(f"Misclassified Samples: **{error_count}**")

if error_count > 0:
    st.write("Model struggles with borderline or ambiguous sequences.")

# ================== 🔥 ROC CURVE ==================
st.subheader("📈 ROC Curve (Model Discrimination)")

try:
    y_bin = label_binarize(y_test, classes=model.classes_)
    y_score = model.predict_proba(X_test)

    fig_roc, ax_roc = plt.subplots()

    for i in range(len(model.classes_)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{model.classes_[i]} (AUC={roc_auc:.2f})")

    ax_roc.plot([0,1], [0,1], linestyle='--')
    ax_roc.legend()
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")

    st.pyplot(fig_roc)

except:
    st.info("ROC Curve not available for this configuration.")

# ------------------ DOWNLOAD ------------------

if prediction:
    df_download = pd.DataFrame({
        "Sequence": [user_input],
        "Prediction": [prediction],
        "GC_Content": [gc_val]
    })

    st.download_button("📥 Download Results",
                       df_download.to_csv(index=False),
                       "dna_result.csv")
    
# ================== 🔥 DOWNLOAD FULL REPORT ==================

full_results = comparison_df.copy()
full_results["Dataset Size"] = len(df)

st.download_button(
    "📥 Download Full Model Comparison",
    full_results.to_csv(index=False),
    "model_comparison.csv"
)

# ------------------ FOOTER ------------------
st.markdown("---")

st.markdown("👩‍💻 Developed by **Bristi Sen**, **Sambriddhi Debnath**, **Rajjyashree Raychaudhuri**")

st.markdown(
"""
💅 *Stalk responsibly:*  
🔗 [LinkedIn](https://www.linkedin.com/in/bristi-sen-709548311/)  
💻 [GitHub](https://github.com/BristiSen)  
"""
)
