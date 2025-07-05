import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import mlflow

from utils import load_amazon_reviews_from_zip, preprocess_corpus, binary_sentiment
from mlflow_tracker import track_model

def setup_mlflow():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("NLP_Automotive")

def main():
    setup_mlflow()

    # Parámetros
    ZIP_PATH = "data/reviews_Automotive_5.json.zip"
    JSON_NAME = "reviews_Automotive_5.json.gz"
    SAMPLE_SIZE = 20000
    EXPERIMENT_NAME = "NLP_Automotive"

    # Carga y preprocesamiento
    df = load_amazon_reviews_from_zip(ZIP_PATH, JSON_NAME, sample_size=SAMPLE_SIZE)
    df = binary_sentiment(df, rating_col='overall', threshold=3)
    df = preprocess_corpus(df, text_column='reviewText', new_column='processed_text')

    # División y vectorización
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment_label'], test_size=0.25, stratify=df['sentiment_label'], random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000, max_df=0.95, min_df=5, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Logistic Regression
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
    start = time.time()
    lr_model.fit(X_train_vec, y_train)
    lr_time = time.time() - start

    lr_pipeline = Pipeline([("vectorizer", tfidf), ("classifier", lr_model)])
    track_model(
        experiment_name=EXPERIMENT_NAME,
        run_name="LogisticRegression",
        model_pipeline=lr_pipeline,
        results_dict={
            "y_true": y_test,
            "y_pred": lr_model.predict(X_test_vec),
            "y_proba": lr_model.predict_proba(X_test_vec)[:, 1],
            "train_time": lr_time
        },
        params={"model": "LogisticRegression"}
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', n_jobs=-1)
    start = time.time()
    rf_model.fit(X_train_vec, y_train)
    rf_time = time.time() - start

    rf_pipeline = Pipeline([("vectorizer", tfidf), ("classifier", rf_model)])
    track_model(
        experiment_name=EXPERIMENT_NAME,
        run_name="RandomForest",
        model_pipeline=rf_pipeline,
        results_dict={
            "y_true": y_test,
            "y_pred": rf_model.predict(X_test_vec),
            "y_proba": rf_model.predict_proba(X_test_vec)[:, 1],
            "train_time": rf_time
        },
        params={
            "model": "RandomForest",
            "n_estimators": rf_model.n_estimators,
            "max_depth": rf_model.max_depth
        }
    )

    print("Modelos entrenados y registrados en MLflow.")

if __name__ == "__main__":
    main()
