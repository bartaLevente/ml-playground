# IMDb Sentiment Classification – Baseline

This project prepares the IMDb movie review dataset and trains a simple baseline model for sentiment classification.

## Files

### `data_preprocessing.py`
- Loads `data/imdb.csv` (needed to be present in the /data directory)
- Checks basic statistics and missing values
- Converts `sentiment` to binary labels:
  - `positive` → `1`
  - `negative` → `0`
- Splits data into train (80%) and test (20%)
- Saves:
  - `data/train.csv`
  - `data/test.csv`

### `baseline.py`
- Loads the train/test datasets
- Uses `DummyClassifier` from scikit-learn as a baseline
- Trains on the training set
- Reports accuracy on train and test sets

## Dataset Format

Input file: `data/imdb.csv`

Expected columns:
- `review` (text)
- `sentiment` (`positive` / `negative`)

After preprocessing:
- `label` (0 or 1) replaces `sentiment`

## Usage

1. Run preprocessing:
```bash
python data.py
