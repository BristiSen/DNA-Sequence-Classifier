import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

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
    ["G vs C Classification", "Promoter vs Non-Promoter"]
)

# ------------------ DESCRIPTION ------------------

if mode == "G vs C Classification":
    st.write("""
    This system classifies DNA sequences based on nucleotide composition.

    - 🟢 **G-Rich Sequence** → Higher Guanine content  
    - 🔵 **C-Rich Sequence** → Higher Cytosine content  
    """)

else:
    st.write("""
    This system classifies DNA sequences based on regulatory patterns.

    - 🟢 **Promoter** → Contains motifs like TATA box (gene regulation regions)  
    - 🔵 **Non-Promoter** → Does not contain promoter-like patterns  
    """)

# ------------------ DATASET GENERATION ------------------

def generate_sequence(length=20):
    bases = ['A', 'T', 'G', 'C']
    weights = [0.2, 0.2, 0.3, 0.3]
    return ''.join(random.choices(bases, weights=weights, k=length))

data = []

for _ in range(300):
    seq = generate_sequence()

    if mode == "G vs C Classification":
        if seq.count('G') > seq.count('C'):
            label = "G-Rich Sequence"
        else:
            label = "C-Rich Sequence"

    else:
        # Simulated promoter logic
        if "TATA" in seq:
            label = "Promoter"
        else:
            label = "Non-Promoter"

    data.append([seq, label])

df = pd.DataFrame(data, columns=["sequence", "label"])

# ------------------ GC CONTENT ------------------

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

df["gc_content"] = df["sequence"].apply(gc_content)

# ------------------ K-MER FUNCTION ------------------

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

df["kmers"] = df["sequence"].apply(lambda x: ' '.join(get_kmers(x)))

# ------------------ VECTORIZATION ------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["kmers"])
y = df["label"]

# ------------------ TRAIN-TEST SPLIT ------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ------------------ MODEL TRAINING ------------------

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ------------------ MODEL EVALUATION ------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"Accuracy: **{round(accuracy * 100, 2)}%**")

# ------------------ USER INPUT ------------------

st.subheader("🔬 Test Your DNA Sequence")

user_input = st.text_input("Enter DNA Sequence (A, T, G, C only)")

if st.button("Predict"):

    if user_input:

        user_input = user_input.upper()

        if not all(c in "ATGC" for c in user_input):
            st.error("❌ Invalid input! Use only A, T, G, C.")
        else:
            kmers = ' '.join(get_kmers(user_input))
            vector = vectorizer.transform([kmers])
            prediction = model.predict(vector)

            st.success(f"🧬 Predicted Class: **{prediction[0]}**")

            gc_val = gc_content(user_input)
            st.info(f"GC Content: {round(gc_val * 100, 2)}%")

            # ------------------ EXPLANATION ------------------

            if mode == "G vs C Classification":
                if prediction[0] == "G-Rich Sequence":
                    st.info("Higher Guanine content → linked to structural stability and regulatory regions.")
                else:
                    st.info("Higher Cytosine content → may indicate different genomic properties.")

            else:
                if prediction[0] == "Promoter":
                    st.info("Likely promoter region → may initiate gene transcription (TATA-like motifs detected).")
                else:
                    st.info("Likely non-promoter region → no strong regulatory motifs found.")

    else:
        st.warning("Please enter a DNA sequence.")

# ------------------ CLASS DISTRIBUTION ------------------

st.subheader("📈 Class Distribution")

label_counts = df['label'].value_counts()

fig1, ax1 = plt.subplots()

colors = []
for label in label_counts.index:
    if label in ["G-Rich Sequence", "Promoter"]:
        colors.append("#4CAF50")  # green
    else:
        colors.append("#2196F3")  # blue

label_counts.plot(kind='bar', ax=ax1, color=colors)

ax1.set_ylabel("Count")
ax1.set_title("Distribution of DNA Classes")

st.pyplot(fig1)

# ------------------ GC CONTENT GRAPH ------------------

st.subheader("🧬 GC Content Distribution")

fig_gc, ax_gc = plt.subplots()
sns.histplot(df["gc_content"], bins=10, kde=True, ax=ax_gc)
ax_gc.set_title("GC Content Distribution")

st.pyplot(fig_gc)

# ------------------ CONFUSION MATRIX ------------------

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

fig2, ax2 = plt.subplots()

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax2)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ------------------ DOWNLOAD BUTTON ------------------

st.subheader("⬇ Download Dataset")

st.download_button(
    label="Download DNA Dataset",
    data=df.to_csv(index=False),
    file_name="dna_dataset.csv",
    mime="text/csv"
)

# ------------------ FOOTER ------------------

st.markdown("---")
st.markdown("👩‍💻 Developed using Machine Learning, Bioinformatics & Streamlit")
