# Fitness Tracker: Barbell Exercise Detection

Classifying barbell exercises from wrist-worn accelerometer and gyroscope data, using a complete machine learning pipeline and a live web demo.

**Live demo:** https://fitness-tracker-barabell-exercise-detection.streamlit.app/

## Overview

A MetaMotion wrist sensor records motion while participants perform barbell exercises. This project turns the raw sensor signals into features and trains a model that predicts which exercise was performed. The Streamlit app lets you run the pipeline on a bundled sample recording or on your own uploaded sensor CSV files.

## Data

- **Sensor:** MetaMotion (wrist-worn)
- **Signals:** accelerometer (12.5 Hz) and gyroscope (25 Hz)
- **Participants:** 5
- **Exercises:** bench press, squat, row, overhead press, deadlift, and rest

## Pipeline

1. **Data processing:** read the raw CSV files, merge the accelerometer and gyroscope streams, and resample to 200 ms windows
2. **Outlier removal:** Chauvenet's criterion, applied per exercise (compared with IQR and LOF)
3. **Missing values:** time-series interpolation
4. **Low-pass filter:** Butterworth filter to remove sensor noise
5. **PCA:** reduces the correlated sensor columns (3 components, chosen with the elbow method)
6. **Sum-of-squares features:** overall acceleration and rotation magnitude (`acc_r`, `gyro_r`)
7. **Temporal features:** rolling mean and standard deviation, computed per set
8. **Frequency features:** Fourier transform (dominant frequency, weighted frequency, spectral entropy, amplitudes)
9. **Clustering:** K-means cluster number as an extra feature
10. **Overlap handling:** dropped incomplete windows and kept every second row to reduce overlap
11. **Modelling:** train/test split with stratification, forward feature selection, and a comparison of feature sets (basic, plus PCA, plus time, plus frequency and cluster)

## Results

| Feature set | Accuracy |
|---|---|
| Basic sensor features | _add your result_ |
| + squared and PCA features | _add your result_ |
| + temporal features | _add your result_ |
| + frequency and cluster features | _add your result_ |

Best model: _add model name and test accuracy._

## Streamlit app

The app has these sections:
- **Overview:** the problem and the pipeline
- **Try it:** pick a sample recording or upload accelerometer and gyroscope CSVs, view the signals, and get the predicted exercise with confidence
- **How it works:** before/after filtering, PCA variance, and the feature-set comparison
- **About:** dataset, tools and limitations

Run it locally:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Project structure

```
├── app/                 # Streamlit app, inference code, saved model, sample data
├── data/                # raw, interim and processed data
├── notebooks/           # exploration
├── src/
│   ├── data/            # data processing (make_dataset.py)
│   ├── features/        # outlier removal, filtering, PCA, temporal and frequency features
│   ├── models/          # classification algorithms and training
│   └── visualization/   # plotting
├── requirements.txt
└── README.md
```

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, matplotlib, seaborn, plotly, Streamlit

## Limitations

- Small dataset (5 participants, one sensor model), so the model may not generalise to other people or devices
- Overlapping windows can inflate accuracy if rows are split randomly instead of by set or participant
- Repetition counting is not implemented yet

## Possible next steps

- Repetition counting with peak detection
- Participant-based cross-validation
- More participants and exercises

## Acknowledgements

Project structure and methodology follow the data science template and tutorial by Dave Ebbelaar, based on *Machine Learning for the Quantified Self* by Mark Hoogendoorn and Burkhardt Funk (Springer, 2017).
