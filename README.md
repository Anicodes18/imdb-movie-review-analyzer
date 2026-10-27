# IMDb Movie Review Sentiment & Theme Analyzer

An NLP project that classifies IMDb movie reviews as **positive or negative**, extracts keyphrases to highlight important themes, and presents insights through visualizations and an interactive Streamlit application.

## Features

- **Exploratory Data Analysis:** Examines sentiment distribution, review length, and dataset characteristics.
- **Text Preprocessing:** Removes HTML tags, decodes HTML entities, normalizes whitespace, and converts text to lowercase.
- **Sentiment Classification:** Uses TF-IDF features and Logistic Regression to predict review sentiment.
- **Keyphrase Extraction:** Uses YAKE to identify important words and phrases in reviews.
- **Visualizations:** Explores sentiment distribution, review-length patterns, model performance, and keyphrase frequencies.
- **Interactive Web App:** Accepts a review and displays its predicted sentiment, model confidence, and extracted keyphrases.

## Model Performance

The Logistic Regression classifier was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| Test accuracy | **90.79%** |
| Precision | Approximately 0.91 |
| Recall | Approximately 0.91 |
| F1-score | Approximately 0.91 |

The dataset was deduplicated to **49,582 reviews** and split into training and testing sets using an 80:20 ratio. TF-IDF was configured with unigrams and bigrams, followed by Logistic Regression for binary sentiment classification.

## Tech Stack

- **Language:** Python
- **NLP & Machine Learning:** scikit-learn, TF-IDF, Logistic Regression, YAKE
- **Data Analysis:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Model Persistence:** joblib
- **Development:** Jupyter Notebook, Git, GitHub

## Project Structure

```text
imdb-movie-review-analyzer/
├── dataset/
│   ├── raw/
│   └── processed/
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_sentiment_analysis.ipynb
│   ├── 04_keyphrase_extraction.ipynb
│   └── 05_visualization_and_insights.ipynb
├── src/
│   └── preprocessing.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Anicodes18/imdb-movie-review-analyzer.git
cd imdb-movie-review-analyzer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, enter a movie review, and select **Analyze Review** to view the results.

The saved model and TF-IDF vectorizer are included in the repository, so retraining is not required to run the app.

## Dataset

The project uses the IMDb 50K Movie Reviews dataset, containing movie reviews labelled as positive or negative. Duplicate reviews were removed during preprocessing.

Place the downloaded CSV in `dataset/raw/` if you want to reproduce the analysis from the notebooks. Dataset files are excluded from the Git repository.

## Future Improvements

- Compare additional sentiment classification models.
- Improve keyphrase relevance by filtering generic phrases.
- Add confidence-aware prediction explanations.
- Deploy the Streamlit application for public access.

---

**Author:** Aniket Rane