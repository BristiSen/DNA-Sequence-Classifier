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

# ------------------ SIDEBAR ------------------

with st.sidebar:
    st.header("⚙️ Settings")
    k_value = st.slider("K-mer Size", 2, 6, 3)
    seq_length = st.slider("Synthetic Sequence Length", 40, 120, 60)
    dataset_size = st.slider("Synthetic Dataset Size", 200, 1000, 600)

# ------------------ PAGE TITLE ------------------

st.title("🧬 DNA Sequence Classification System")
st.markdown("### Bioinformatics Model for DNA Analysis")

st.divider()

# ------------------ MODE ------------------

mode = st.selectbox(
    "Select Classification Mode",
    ["G vs C Classification", "Promoter vs Non-Promoter", "Species Classification (Real Data)"]
)

# ------------------ DESCRIPTION ------------------

if mode == "G vs C Classification":
    st.info("Classifies DNA based on strong G vs C dominance.")
elif mode == "Promoter vs Non-Promoter":
    st.info("Detects promoter regions using sequence motifs.")
else:
    st.info("Uses real FASTA genomic data to classify species.")

# ------------------ FUNCTIONS ------------------

def generate_sequence(length=60):
    return ''.join(random.choices(['A','T','G','C'], k=length))

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

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
                chunk = seq[i:i+100]
                data.append([chunk, lbl])

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

# ------------------ FIX ------------------

num_classes = len(y.unique())

if len(df) < num_classes * 2:
    st.error("Dataset too small for all classes.")
    st.stop()

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

# ------------------ EVAL ------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

st.divider()

# ------------------ INPUT ------------------

st.subheader("🔬 Test Your DNA Sequence")

uploaded_file = st.file_uploader("Upload DNA File", type=["fasta", "txt"])
user_input = st.text_input("Enter DNA Sequence")

prediction = None
probs = None
confidence = None

if user_input:
    st.code(user_input[:100] + "...")

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

# ------------------ ANALYSIS ------------------

if prediction:
    st.subheader("🧠 Analysis")

    if mode == "G vs C Classification":
        st.write("Classification based on G vs C dominance using k-mers.")
    elif mode == "Promoter vs Non-Promoter":
        st.write("Based on promoter motifs like TATA, CAAT, TTGACA.")
    else:
        st.write("Based on similarity to species genomic patterns.")

# ------------------ ALL ORIGINAL GRAPHS ------------------

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

# PROMOTER GRAPH

if user_input and mode == "Promoter vs Non-Promoter":

    motifs = {"TATA":3,"CAAT":2,"TTGACA":3}
    score = sum(weight for motif, weight in motifs.items() if motif in user_input)
    max_score = sum(motifs.values())

    promoter_pct = (score/max_score)*100 if max_score else 0
    non_promoter_pct = 100 - promoter_pct

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(["Promoter","Non-Promoter"],
               [promoter_pct, non_promoter_pct],
               color=["green","red"])
    st.pyplot(fig_bar)

# SPECIES GRAPH

if user_input and mode == "Species Classification (Real Data)" and probs is not None:

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(model.classes_, probs,
               color=["green","blue","purple","orange"])
    st.pyplot(fig_bar)

# GC DISTRIBUTION

st.subheader("🧬 GC Content Distribution")

fig_gc, ax_gc = plt.subplots()
sns.histplot(df["gc_content"], bins=10, kde=True, ax=ax_gc)
st.pyplot(fig_gc)

# CONFUSION MATRIX

st.subheader("📊 Confusion Matrix")

fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
st.pyplot(fig2)

st.divider()

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

# ------------------ FOOTER ------------------

st.markdown("👩‍💻 Developed by **Bristi Sen**, **Sambriddhi Debnath**, **Rajjyashree Raychaudhuri**")
