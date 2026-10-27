
import streamlit as st
import joblib
import yake

from pathlib import Path
from src.preprocessing import clean_review


# -----------------------------
# Load model artifacts
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODEL_DIR / "sentiment_model.pkl")

kw_extractor = yake.KeywordExtractor(
    lan="en",
    n=3,
    dedupLim=0.9,
    top=10
)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="IMDb Review Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDb Movie Review Analyzer")

st.markdown(
    """
    Analyse a movie review using **Natural Language Processing (NLP)**.
    Predict its sentiment and discover keyphrases that highlight
    important topics in the review.
    """
)

st.divider()


# -----------------------------
# Review input
# -----------------------------
st.subheader("Enter a movie review")

review = st.text_area(
    "Review text",
    height=200,
    placeholder=(
        "Paste a movie review here to analyse its sentiment "
        "and extract keyphrases..."
    ),
    label_visibility="collapsed"
)

analyze = st.button(
    "Analyze Review",
    type="primary",
    use_container_width=True
)


# -----------------------------
# Analysis results
# -----------------------------
if analyze:
    if not review.strip():
        st.warning("Please enter a review before analysing.")
    else:
        cleaned_review = clean_review(review)
        features = vectorizer.transform([cleaned_review])

        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities) * 100)
        else:
            confidence = None

        keyphrases = kw_extractor.extract_keywords(review)

        st.divider()
        st.subheader("Analysis Results")

        result_col, confidence_col = st.columns(2)

        with result_col:
            st.markdown("**Predicted sentiment**")
            if prediction.lower() == "positive":
                st.success("😊 Positive")
            else:
                st.error("😞 Negative")

        with confidence_col:
            st.markdown("**Model confidence**")
            if confidence is not None:
                st.metric("Predicted class probability", f"{confidence:.2f}%")
            else:
                st.write("Not available")

        if confidence is not None:
            st.progress(confidence / 100)
            st.caption(
                "This is the model's estimated probability for its "
                "predicted class, not a guarantee of correctness."
            )

        st.divider()
        st.subheader("Keyphrases")
        st.caption(
            "Important phrases extracted from the review using YAKE."
        )

        if keyphrases:
            for phrase, score in keyphrases:
                st.markdown(f"- {phrase}")
        else:
            st.info("No keyphrases were found for this review.")

