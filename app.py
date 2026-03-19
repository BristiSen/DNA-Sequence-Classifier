import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ------------------ PAGE TITLE ------------------

st.title("🧬 DNA Sequence Classification System")
st.markdown("### Bioinformatics Model for DNA Analysis")

# ------------------ MODE SELECTION ------------------

mode = st.selectbox(
    "Select Classification Mode",
    ["G vs C Classification", "Promoter vs Non-Promoter", "Species Classification (Real Data)"]
)

# ------------------ DESCRIPTION ------------------

if mode == "G vs C Classification":
    st.write("Classifies DNA based on strong G vs C dominance.")

elif mode == "Promoter vs Non-Promoter":
    st.write("Detects promoter regions using sequence motifs.")

else:
    st.write("Uses real FASTA genomic data to classify species.")

# ------------------ DATA GENERATION ------------------

def generate_sequence(length=60):
    bases = ['A', 'T', 'G', 'C']
    return ''.join(random.choices(bases, k=length))

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

            chunk_size = 100

            for i in range(0, len(seq) - chunk_size, chunk_size):
                chunk = seq[i:i+chunk_size]
                data.append([chunk, lbl])

    return pd.DataFrame(data, columns=["sequence", "label"])

# ------------------ DATA CREATION ------------------

data = []

if mode == "Species Classification (Real Data)":
    df = load_real_dataset()

    if len(df) < 40:
        st.warning("⚠️ Very small dataset — results may overfit.")

else:
    for _ in range(600):
        seq = generate_sequence()

        if mode == "G vs C Classification":
            diff = (seq.count('G') - seq.count('C')) / len(seq)

            if diff > 0.15:
                label = "G-Rich Sequence"
            elif diff < -0.15:
                label = "C-Rich Sequence"
            else:
                continue

        else:
            label = "Promoter" if "TATA" in seq else "Non-Promoter"

        data.append([seq, label])

    df = pd.DataFrame(data, columns=["sequence", "label"])

# ------------------ BALANCE DATA ------------------

if mode == "G vs C Classification" and len(df) > 0:
    df_g = df[df['label'] == "G-Rich Sequence"]
    df_c = df[df['label'] == "C-Rich Sequence"]

    min_size = min(len(df_g), len(df_c))

    if min_size > 0:
        df = pd.concat([
            df_g.sample(min_size, random_state=42),
            df_c.sample(min_size, random_state=42)
        ])

# ------------------ SAFETY CHECK ------------------

if len(df) < 10:
    st.error("Dataset too small to train model.")
    st.stop()

# ------------------ GC CONTENT ------------------

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

df["gc_content"] = df["sequence"].apply(gc_content)

# ------------------ K-MERS ------------------

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

df["kmers"] = df["sequence"].apply(lambda x: ' '.join(get_kmers(x)))

# ------------------ VECTORIZATION ------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["kmers"])
y = df["label"]

# ------------------ SPLIT ------------------

test_size = 0.3 if len(df) < 100 else 0.25

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

# ------------------ EVALUATION ------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"Accuracy: **{round(accuracy * 100, 2)}%**")

if accuracy > 0.98:
    st.warning("⚠️ Extremely high accuracy — possible overfitting.")

# ------------------ USER INPUT ------------------

st.subheader("🔬 Test Your DNA Sequence")

uploaded_file = st.file_uploader("Upload DNA File (.fasta or .txt)", type=["fasta", "txt"])
user_input = st.text_input("Or enter DNA Sequence")

probs = None  # NEW

if st.button("Predict"):

    if uploaded_file:
        lines = uploaded_file.read().decode("utf-8").split("\n")
        user_input = ''.join([l.strip() for l in lines if not l.startswith(">")])

    if user_input:
        user_input = user_input.upper()

        if not all(c in "ATGC" for c in user_input):
            st.error("❌ Invalid DNA sequence.")
        else:
            kmers = ' '.join(get_kmers(user_input))
            vector = vectorizer.transform([kmers])
            prediction = model.predict(vector)

            # NEW: species probabilities
            if mode == "Species Classification (Real Data)":
                probs = model.predict_proba(vector)[0]

            st.success(f"🧬 Predicted: **{prediction[0]}**")
            st.info(f"GC Content: {round(gc_content(user_input)*100,2)}%")

# ------------------ INPUT COMPOSITION GRAPH ------------------

st.subheader("📊 DNA Composition (Your Input)")

if user_input:
    fig1, ax1 = plt.subplots()

    if mode == "G vs C Classification":
        g_count = user_input.upper().count('G')
        c_count = user_input.upper().count('C')

        ax1.bar(["G", "C"], [g_count, c_count],
                color=["green", "blue"])
        ax1.set_title("G vs C Composition")

    else:
        bases = ['A', 'T', 'G', 'C']
        values = [user_input.upper().count(b) for b in bases]
        colors = ['orange', 'red', 'green', 'blue']

        ax1.bar(bases, values, color=colors)
        ax1.set_title("A, T, G, C Composition")

    ax1.set_ylabel("Count")
    st.pyplot(fig1)

# ================= PROMOTER BAR GRAPH (REAL FIX) =================

if user_input and mode == "Promoter vs Non-Promoter":

    seq = user_input.upper()

    # 🔥 motif scoring (more realistic)
    motifs = {
        "TATA": 3,
        "CAAT": 2,
        "TTGACA": 3
    }

    score = 0
    max_score = sum(motifs.values())

    for motif, weight in motifs.items():
        if motif in seq:
            score += weight

    # normalize into percentage
    promoter_pct = (score / max_score) * 100 if max_score > 0 else 0
    non_promoter_pct = 100 - promoter_pct

    st.subheader("🧬 Promoter vs Non-Promoter (Input Based)")

    # TEXT OUTPUT
    st.write(f"🟢 Promoter Likelihood: **{round(promoter_pct,2)}%**")
    st.write(f"🔴 Non-Promoter Likelihood: **{round(non_promoter_pct,2)}%**")

    # BAR GRAPH
    fig_bar, ax_bar = plt.subplots()

    categories = ["Promoter", "Non-Promoter"]
    values = [promoter_pct, non_promoter_pct]
    colors = ["green", "red"]

    ax_bar.bar(categories, values, color=colors)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Percentage (%)")
    ax_bar.set_title("Promoter Likelihood of Input Sequence")

    for i, v in enumerate(values):
        ax_bar.text(i, v + 1, f"{v:.1f}%", ha='center')

    st.pyplot(fig_bar)

# ================= SPECIES GRAPH (ADDED ONLY) =================

if user_input and mode == "Species Classification (Real Data)" and probs is not None:

    st.subheader("🧬 Species Similarity")

    labels = model.classes_
    probs = probs / sum(probs)

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(labels, probs, color=["green", "blue", "purple", "orange"])
    ax_bar.set_ylim(0, 1)

    for i, v in enumerate(probs):
        ax_bar.text(i, v + 0.01, f"{v:.2f}", ha='center')

    st.pyplot(fig_bar)

# ------------------ GC CONTENT ------------------

st.subheader("🧬 GC Content Distribution")

fig_gc, ax_gc = plt.subplots()
sns.histplot(df["gc_content"], bins=10, kde=True, ax=ax_gc)
st.pyplot(fig_gc)

# ------------------ CONFUSION MATRIX ------------------

st.subheader("📊 Confusion Matrix")

labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

fig2, ax2 = plt.subplots()

sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues")

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ------------------ FOOTER ------------------

st.markdown("---")
st.markdown("👩‍💻 Developed using Machine Learning, Bioinformatics & Streamlit")
