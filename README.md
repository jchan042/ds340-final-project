# Predicting Food Trends from Yelp Reviews Using Multimodal Temporal Modeling

This repository contains the notebooks used for the DS340 final project on predicting food trend signals from Yelp review data across Philadelphia, Nashville, and Tampa.

## Project overview

The project builds monthly food-trend signals from Yelp reviews, then compares several forecasting/modeling approaches:

1. ARIMA baseline
2. TF-IDF + Ridge Regression
3. DistilBERT + Linear Regression
4. LSTM using trend history only
5. Multimodal LSTM using both trend history and DistilBERT embeddings

The main target variable is a normalized monthly trend score for each city, month, and food keyword.

## Required data

The notebooks expect the Yelp Open Dataset JSON files to be available in Google Drive at:

```text
/content/drive/MyDrive/DS340 Final Project/Yelp JSON/yelp_dataset/
```

Required Yelp files:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_review.json
```

If your Drive folder is named differently, update the `BASE`, `REVIEWS`, and `BUSINESSES`/`BIZ` paths at the top of the notebooks.

## Recommended environment

These notebooks were written for Google Colab.

Installations are included in the notebooks, but the main packages used are:

```text
pandas
numpy
matplotlib
pyarrow
scikit-learn
scipy
statsmodels
torch
transformers
```

## How to run the notebooks

Run the notebooks independently in the order below. The preprocessing notebook must be run first because it creates the split parquet files used by the modeling notebooks.

### 1. `preprocessing.ipynb`

Purpose:

- Loads Yelp business and review JSON files
- Filters to restaurant businesses in Philadelphia, Nashville, and Tampa
- Extracts mentions of the food keywords
- Computes monthly normalized trend scores
- Applies rolling smoothing
- Creates train/validation/test temporal splits

Expected outputs:

```text
trend_signals.parquet
trend_signals_filled.parquet
split_train.parquet
split_val.parquet
split_test.parquet
```

These files are saved to:

```text
/content/drive/MyDrive/DS340 Final Project/
```

### 2. `distilBERT.ipynb`

Purpose:

- Loads the train/validation/test split files from Google Drive
- Re-reads Yelp reviews
- Generates DistilBERT embeddings for city-month-keyword review groups
- Trains/evaluates a DistilBERT + linear/ridge regression model

Expected outputs:

```text
embed_df.parquet
distilbert_predictions.parquet
```

`embed_df.parquet` is needed by the multimodal LSTM notebook.

### 3. `TFIDF.ipynb`

Purpose:

- Loads preprocessed train/validation/test split files from Google Drive
- Aggregates review text by city and month
- Builds TF-IDF features
- Trains/evaluates a TF-IDF + Ridge Regression model

Expected outputs:

```text
tfidf_test_predictions.parquet
tfidf_mae_per_keyword.csv
tfidf_predictions_<city>.png
```

### 4. `ARIMA.ipynb`

Purpose:

- Loads `split_train.parquet`, `split_val.parquet`, and `split_test.parquet`
- Fits ARIMA(1,1,1) independently for each city-keyword pair
- Performs walk-forward forecasting for 1-, 2-, and 3-month horizons
- Reports MAE and creates prediction-vs-actual plots

Important:

At the first upload prompt, upload:

```text
split_train.parquet
split_val.parquet
split_test.parquet
```

Expected output:

```text
arima_predictions.parquet
```

### 5. `LSTM.ipynb`

Purpose:

- Loads `split_train.parquet`, `split_val.parquet`, and `split_test.parquet`
- Builds 6-month trend sequences
- Trains a trend-only LSTM
- Evaluates 1-, 2-, and 3-month-ahead predictions
- Creates prediction-vs-actual plots for multiple cities

Important:

At the first upload prompt, upload:

```text
split_train.parquet
split_val.parquet
split_test.parquet
```

Expected output:

```text
lstm_predictions.parquet
```

### 6. `combined.ipynb`

Purpose:

- Loads split parquet files and DistilBERT embeddings
- Builds multimodal inputs using trend sequences plus text embeddings
- Trains the multimodal LSTM
- Evaluates 1-, 2-, and 3-month-ahead predictions
- Creates prediction-vs-actual plots for multiple cities

Important:

At the first upload prompt, upload:

```text
embed_df.parquet
split_train.parquet
split_val.parquet
split_test.parquet
```

Expected output:

```text
multimodal_lstm_predictions.parquet
```

## Notes

- The modeling notebooks can be run independently after `preprocessing.ipynb` has produced the split parquet files.
- `combined.ipynb` additionally requires `embed_df.parquet`, which is produced by `distilBERT.ipynb`.
- ARIMA, LSTM, and multimodal LSTM notebooks use Colab file upload prompts instead of directly reading from Drive, so the required parquet files should be uploaded when prompted.
- TF-IDF and DistilBERT read directly from the Google Drive project folder.
- Figures in the paper show representative keywords/cities for readability; the reported metrics are computed across all available city-keyword combinations.

## Suggested run order summary

```text
1. preprocessing.ipynb
2. distilBERT.ipynb
3. TFIDF.ipynb
4. ARIMA.ipynb
5. LSTM.ipynb
6. combined.ipynb
```

