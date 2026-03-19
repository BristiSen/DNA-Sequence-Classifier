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

# 🔥 FIX: Non-overlapping chunks (reduces data leakage)

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

            # 🔥 NON-overlapping chunks (IMPORTANT FIX)
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
    max_depth=10   # 🔥 prevents overfitting
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

            st.success(f"🧬 Predicted: **{prediction[0]}**")
            st.info(f"GC Content: {round(gc_content(user_input)*100,2)}%")

# ------------------ CLASS DISTRIBUTION ------------------

st.subheader("📈 Class Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(x=df['label'], ax=ax1)
plt.xticks(rotation=30)
st.pyplot(fig1)

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
