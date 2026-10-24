"""Sensor preprocessing and prediction for the Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.features.FrequencyAbstraction import FourierTransformation

SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
PREDICTOR_COLUMNS = SENSOR_COLUMNS + ["acc_r", "gyro_r"]
RAW_COLUMN_ALIASES = {
    "acc_x": ("acc_x", "x-axis (g)", "x-axis"),
    "acc_y": ("acc_y", "y-axis (g)", "y-axis"),
    "acc_z": ("acc_z", "z-axis (g)", "z-axis"),
    "gyro_x": ("gyro_x", "x-axis (deg/s)", "x-axis"),
    "gyro_y": ("gyro_y", "y-axis (deg/s)", "y-axis"),
    "gyro_z": ("gyro_z", "z-axis (deg/s)", "z-axis"),
}


def _timestamp_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    timestamp_column = next(
        (column for column in ("epoch (ms)", "timestamp", "time") if column in frame),
        None,
    )
    if timestamp_column is not None:
        values = frame[timestamp_column]
    elif isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    else:
        raise ValueError(
            "Each CSV must contain an 'epoch (ms)' timestamp column (or a datetime index)."
        )

    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().all():
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(numeric_values, unit="ms", errors="coerce")
        )
    else:
        timestamps = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if timestamps.isna().all():
        raise ValueError(
            "The timestamp column could not be parsed. Use epoch milliseconds or "
            "a readable date/time value."
        )
    return timestamps


def _canonical_sensor_frame(
    frame: pd.DataFrame, sensor: str
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"The {sensor} CSV is empty.")

    axes = SENSOR_COLUMNS[:3] if sensor == "accelerometer" else SENSOR_COLUMNS[3:]
    mapped = {}
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}

    for axis in axes:
        aliases = RAW_COLUMN_ALIASES[axis]
        source_column = next(
            (
                normalized_columns[alias.lower()]
                for alias in aliases
                if alias.lower() in normalized_columns
            ),
            None,
        )
        if source_column is None:
            expected = ", ".join(aliases)
            raise ValueError(
                f"The {sensor} CSV is missing an axis column. Expected one of: {expected}."
            )
        mapped[axis] = pd.to_numeric(frame[source_column], errors="coerce")

    timestamps = _timestamp_index(frame)
    valid_timestamps = ~timestamps.isna()
    result = pd.DataFrame(
        {
            column: values.loc[valid_timestamps].to_numpy()
            for column, values in mapped.items()
        },
        index=timestamps[valid_timestamps],
    )
    result.index.name = "epoch (ms)"
    return result.dropna(how="all").sort_index()


def merge_and_resample(
    acc_df: pd.DataFrame, gyro_df: pd.DataFrame
) -> pd.DataFrame:
    """Validate raw sensor tables and merge them on the 200 ms training cadence."""
    accelerometer = _canonical_sensor_frame(acc_df, "accelerometer")
    gyroscope = _canonical_sensor_frame(gyro_df, "gyroscope")
    merged = accelerometer.join(gyroscope, how="outer").sort_index()
    if merged.empty:
        raise ValueError("The uploaded sensor files contain no usable timestamped rows.")

    merged = merged.resample("200ms").mean()
    merged = merged.dropna(subset=SENSOR_COLUMNS, how="all")
    merged[SENSOR_COLUMNS] = merged[SENSOR_COLUMNS].interpolate(limit_direction="both")
    merged = merged.dropna(subset=SENSOR_COLUMNS)
    if len(merged) < 20:
        raise ValueError(
            "This recording is too short to process. Upload at least 20 resampled rows "
            "(about four seconds of sensor data)."
        )
    merged["set"] = 1
    return merged


def apply_low_pass(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    low_pass = LowPassFilter()
    for column in SENSOR_COLUMNS:
        filtered = low_pass.low_pass_filter(filtered, column, 5, 1.3)
        filtered[column] = filtered[f"{column}_lowpass"]
        filtered.drop(columns=[f"{column}_lowpass"], inplace=True)
    return filtered


def fit_pca_parameters(training_frame: pd.DataFrame) -> dict:
    """Fit the source PCA implementation and retain the exact training scaling."""
    training = training_frame.copy()
    for column in SENSOR_COLUMNS:
        training[column] = training[column].interpolate(limit_direction="both")
    filtered = apply_low_pass(training)
    pca_transformer = PrincipalComponentAnalysis()
    pca_transformer.apply_pca(filtered, SENSOR_COLUMNS, 3)
    means = filtered[SENSOR_COLUMNS].mean()
    ranges = filtered[SENSOR_COLUMNS].max() - filtered[SENSOR_COLUMNS].min()
    if (ranges == 0).any():
        raise ValueError("Cannot fit PCA: a training sensor axis has no variation.")
    return {
        "pca": pca_transformer.pca,
        "means": means,
        "ranges": ranges,
        "explained_variance_ratio": pca_transformer.pca.explained_variance_ratio_,
    }


def _apply_saved_pca(frame: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    normalized = (frame[SENSOR_COLUMNS] - parameters["means"]) / parameters["ranges"]
    components = parameters["pca"].transform(normalized[SENSOR_COLUMNS])
    transformed = frame.copy()
    for index in range(3):
        transformed[f"pca_{index + 1}"] = components[:, index]
    return transformed


def engineer_features(
    merged: pd.DataFrame,
    pca_parameters: dict,
    kmeans=None,
) -> pd.DataFrame:
    """Apply the same feature transforms and per-set windows used by training."""
    frame = merged.copy().sort_index()
    missing = [column for column in SENSOR_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing required sensor columns: {', '.join(missing)}.")
    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").interpolate(
            limit_direction="both"
        )
    frame = frame.dropna(subset=SENSOR_COLUMNS)

    if "set" not in frame:
        frame["set"] = 1
    if "duration" not in frame:
        frame["duration"] = 0.0
        for set_id, subset in frame.groupby("set", sort=False):
            frame.loc[subset.index, "duration"] = (
                subset.index[-1] - subset.index[0]
            ).total_seconds()

    frame = apply_low_pass(frame)
    frame = _apply_saved_pca(frame, pca_parameters)
    frame["acc_r"] = np.sqrt(
        frame["acc_x"] ** 2 + frame["acc_y"] ** 2 + frame["acc_z"] ** 2
    )
    frame["gyro_r"] = np.sqrt(
        frame["gyro_x"] ** 2 + frame["gyro_y"] ** 2 + frame["gyro_z"] ** 2
    )

    rolling = frame.groupby("set")[PREDICTOR_COLUMNS].rolling(5)
    means = rolling.mean().reset_index(level=0, drop=True)
    deviations = rolling.std().reset_index(level=0, drop=True)
    means.columns = [f"{column}_temp_mean_ws_5" for column in PREDICTOR_COLUMNS]
    deviations.columns = [f"{column}_temp_std_ws_5" for column in PREDICTOR_COLUMNS]
    frame = pd.concat([frame, means, deviations], axis=1)

    frequency_frame = frame.reset_index()
    timestamp_name = merged.index.name or "index"
    if timestamp_name != "epoch (ms)":
        frequency_frame.rename(columns={timestamp_name: "epoch (ms)"}, inplace=True)
    frequency = FourierTransformation()
    per_set = []
    for set_id in frequency_frame["set"].drop_duplicates():
        subset = frequency_frame[frequency_frame["set"] == set_id].reset_index(
            drop=True
        )
        per_set.append(
            frequency.abstract_frequency(subset, PREDICTOR_COLUMNS, 14, 5)
        )
    if not per_set:
        raise ValueError("No complete exercise windows could be generated.")
    frame = pd.concat(per_set, ignore_index=True)
    if "epoch (ms)" in frame:
        frame["epoch (ms)"] = pd.to_datetime(frame["epoch (ms)"])
        frame.set_index("epoch (ms)", inplace=True)

    if kmeans is not None:
        frame["cluster"] = kmeans.predict(frame[["acc_x", "acc_y", "acc_z"]])
    frame = frame.dropna()
    frame = frame.iloc[::2].copy()
    if frame.empty:
        raise ValueError(
            "No prediction windows remain after preprocessing. Upload a longer recording."
        )
    return frame


def load_model_artifacts() -> dict:
    import joblib

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    return {
        "model": joblib.load(artifact_dir / "model.joblib"),
        "preprocessors": joblib.load(artifact_dir / "preprocessors.joblib"),
        "feature_columns": joblib.load(artifact_dir / "feature_columns.joblib"),
    }


def predict_from_raw(
    acc_df: pd.DataFrame, gyro_df: pd.DataFrame, artifacts: dict | None = None
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, str]:
    """Return per-window labels, class probabilities, processed rows, and majority vote."""
    artifacts = artifacts or load_model_artifacts()
    model = artifacts["model"]
    preprocessors = artifacts["preprocessors"]
    feature_columns = artifacts["feature_columns"]

    merged = merge_and_resample(acc_df, gyro_df)
    processed = engineer_features(
        merged, preprocessors["pca"], preprocessors["kmeans"]
    )
    absent = [column for column in feature_columns if column not in processed]
    if absent:
        raise ValueError(
            "The processed data is missing model features: " + ", ".join(absent)
        )
    model_input = processed[feature_columns]
    labels = pd.Series(model.predict(model_input), index=processed.index, name="prediction")
    probabilities = pd.DataFrame(
        model.predict_proba(model_input),
        columns=model.classes_,
        index=processed.index,
    )
    overall = str(labels.mode().iloc[0])
    processed = processed.copy()
    processed["prediction"] = labels
    return labels, probabilities, processed, overall
