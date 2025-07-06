import streamlit as st
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ✅ Load entire dataset first, then stratified sample
@st.cache_data
def load_data():
    dataset = load_dataset("imdb", split="train")  # Load full 25,000 training samples
    df = pd.DataFrame({'text': dataset['text'], 'label': dataset['label']})

    # Sample 10,000 rows with balanced class distribution
    df_pos = df[df['label'] == 1].sample(n=5000, random_state=42)
    df_neg = df[df['label'] == 0].sample(n=5000, random_state=42)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

# ✅ Train ML model with guaranteed class balance
@st.cache_resource
def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, accuracy

# ✅ Streamlit app UI
st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬")
st.title("🎬 IMDb Sentiment Analysis (ML-Based)")

st.markdown("""
This app uses a **Logistic Regression model** trained on **10,000 IMDb reviews** (balanced between positive and negative)
to predict whether your review is 😊 Positive or ☹️ Negative.
""")

with st.spinner("Loading data and training model..."):
    df = load_data()
    model, vectorizer, accuracy = train_model(df)

st.success("✅ Model is trained and ready!")

# ✍️ User input
user_input = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid sentence.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0][prediction]

        sentiment = "😊 Positive" if prediction == 1 else "☹️ Negative"
        st.subheader("Prediction:")
        st.success(f"{sentiment} ({proba * 100:.2f}% confidence)")

# ✅ Accuracy display
st.markdown("---")
st.markdown(f"📊 **Model Accuracy on Test Set:** `{accuracy * 100:.2f}%`")
