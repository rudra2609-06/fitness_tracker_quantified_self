"""Train and export the Smart Gym Coach model and preprocessing artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import SENSOR_COLUMNS, engineer_features, fit_pca_parameters


def selected_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Reproduce feature_set_4 from src/models/train_model.py in saved column order."""
    included = {
        *SENSOR_COLUMNS,
        "acc_r",
        "gyro_r",
        "pca_1",
        "pca_2",
        "pca_3",
        "cluster",
    }
    included.update(column for column in frame if "_temp_" in column)
    included.update(
        column
        for column in frame
        if "_freq_" in column or "_pse_" in column
    )
    return [
        column
        for column in frame.columns
        if column not in {"label", "participant", "category", "set", "epoch (ms)"}
        and column in included
    ]


def feature_group_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "Basic": SENSOR_COLUMNS.copy(),
        "+ PCA": ["acc_r", "gyro_r", "pca_1", "pca_2", "pca_3"],
        "+ Time": [column for column in frame if "_temp_" in column],
        "+ Frequency": [
            column
            for column in frame
            if "_freq_" in column or "_pse_" in column
        ],
        "+ Cluster": ["cluster"],
    }
    combined: list[str] = []
    result = {}
    for name, columns in groups.items():
        combined.extend(columns)
        result[name] = combined.copy()
    return result


def evaluate_feature_sets(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]]
) -> dict[str, float]:
    train_indexes, test_indexes = train_test_split(
        range(len(frame)),
        random_state=42,
        test_size=0.25,
        stratify=frame["label"],
    )
    scores = {}
    for name, columns in feature_sets.items():
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            criterion="entropy",
            random_state=42,
        )
        model.fit(frame.iloc[train_indexes][columns], frame.iloc[train_indexes]["label"])
        predictions = model.predict(frame.iloc[test_indexes][columns])
        scores[name] = float(
            accuracy_score(frame.iloc[test_indexes]["label"], predictions)
        )
    return scores


def main() -> None:
    interim = PROJECT_ROOT / "data" / "interim"
    raw_features_path = interim / "03_data_features.pkl"
    preprocessed_path = interim / "02_outlier_to_nan.pkl"
    if not raw_features_path.is_file() or not preprocessed_path.is_file():
        raise FileNotFoundError(
            "Expected data/interim/02_outlier_to_nan.pkl and "
            "data/interim/03_data_features.pkl in the project root."
        )

    reference = pd.read_pickle(raw_features_path)
    training_raw = pd.read_pickle(preprocessed_path)
    for column in SENSOR_COLUMNS:
        training_raw[column] = training_raw[column].interpolate(
            limit_direction="both"
        )
    pca_parameters = fit_pca_parameters(training_raw)
    engineered = engineer_features(training_raw, pca_parameters)

    if len(engineered) != len(reference):
        raise ValueError(
            "The app preprocessing produced "
            f"{len(engineered)} training windows; the saved feature artifact has "
            f"{len(reference)}. Refusing to export an unaligned model."
        )
    reference_labels = reference["label"].astype(object).astype(str).to_numpy()
    engineered_labels = engineered["label"].astype(object).astype(str).to_numpy()
    if not (reference_labels == engineered_labels).all():
        raise ValueError(
            "Preprocessed training labels do not align with "
            "03_data_features.pkl; refusing to train on mismatched windows."
        )

    feature_columns = selected_feature_columns(reference)
    if not feature_columns:
        raise ValueError("No training feature columns were selected.")

    kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
    engineered["cluster"] = kmeans.fit_predict(
        engineered[["acc_x", "acc_y", "acc_z"]]
    )
    missing = [column for column in feature_columns if column not in engineered]
    if missing:
        raise ValueError(
            "The app preprocessing is missing training features: "
            + ", ".join(missing)
        )
    labels = engineered["label"].astype(object).astype(str)
    engineered["label"] = labels
    if "cluster" not in feature_columns:
        raise ValueError("The selected training feature set must include cluster.")

    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        criterion="entropy",
        random_state=42,
    )
    model.fit(engineered[feature_columns], labels)

    feature_sets = feature_group_columns(engineered)
    metrics = evaluate_feature_sets(engineered, feature_sets)

    artifact_dir = APP_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    joblib.dump(feature_columns, artifact_dir / "feature_columns.joblib")
    joblib.dump(list(model.classes_), artifact_dir / "class_names.joblib")
    joblib.dump(
        {"pca": pca_parameters, "kmeans": kmeans},
        artifact_dir / "preprocessors.joblib",
    )
    with (artifact_dir / "feature_set_accuracy.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(metrics, output, indent=2)
    with (artifact_dir / "pca_explained_variance.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(
            pca_parameters["explained_variance_ratio"].tolist(), output, indent=2
        )

    print(f"Exported {len(feature_columns)} ordered features.")
    print(f"Classes: {list(model.classes_)}")
    print("Held-out feature-set accuracy:", metrics)


if __name__ == "__main__":
    main()
