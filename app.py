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

    st.divider()
    threshold = st.slider("G-C Sensitivity Threshold", 0.05, 0.3, 0.15)

    st.markdown("---")
    st.caption("✨ Tune parameters like a real bioinformatics researcher")

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

# ------------------ EVAL ------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{accuracy*100:.2f}%")

# 🔥 NEW: CLASS DISTRIBUTION
st.subheader("📊 Training Data Distribution")
fig_dist, ax_dist = plt.subplots()
sns.countplot(x=df['label'], ax=ax_dist)
st.pyplot(fig_dist)

st.divider()

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

# ------------------ EXPLAINABILITY ------------------

if prediction:
    st.subheader("🔍 Why this prediction?")

    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_

    top_features = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)[:5]

    for f, val in top_features:
        st.write(f"• {f} → {round(val,4)}")

# ------------------ MUTATION ------------------

if user_input:
    if st.button("🧪 Mutate Sequence"):
        mutated = mutate(user_input)
        st.write("Mutated:", mutated)

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
st.markdown("---")

st.markdown("👩‍💻 Developed by **Bristi Sen**, **Sambriddhi Debnath**, **Rajjyashree Raychaudhuri**")

st.markdown(
"""
💅 *Stalk responsibly:*  
🔗 [LinkedIn](https://www.linkedin.com/in/bristi-sen-709548311/)  
💻 [GitHub](https://github.com/BristiSen)  
"""
)
