"""Smart Gym Coach: exercise classification from wrist-sensor recordings."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference import (
    apply_low_pass,
    load_model_artifacts,
    merge_and_resample,
    predict_from_raw,
)

SAMPLE_DIR = APP_DIR / "sample_data"
ARTIFACT_DIR = APP_DIR / "artifacts"
CHART_WIDTH = 10
CHART_HEIGHT = 4

sns.set_theme(
    style="whitegrid",
    palette="crest",
    rc={"axes.spines.top": False, "axes.spines.right": False},
)


@st.cache_resource
def cached_artifacts() -> dict:
    return load_model_artifacts()


@st.cache_data
def cached_sample_files(acc_path: str, gyro_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(acc_path), pd.read_csv(gyro_path)


@st.cache_data
def cached_json(path: str) -> object:
    import json

    with Path(path).open(encoding="utf-8") as input_file:
        return json.load(input_file)


def read_uploaded_csv(uploaded_file, sensor_name: str) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    if uploaded_file.size == 0:
        st.error(f"The {sensor_name} CSV is empty. Choose a file with sensor rows.")
        return None
    try:
        return pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the {sensor_name} CSV: {error}")
        return None


def show_chart(figure: plt.Figure) -> None:
    figure.tight_layout(pad=1.5, rect=(0, 0, 1, 0.97))
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def plot_raw_sensors(merged: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(CHART_WIDTH, CHART_HEIGHT * 1.7), sharex=True)
    sensor_groups = (
        (("acc_x", "acc_y", "acc_z"), "Accelerometer", "Acceleration (g)"),
        (("gyro_x", "gyro_y", "gyro_z"), "Gyroscope", "Angular velocity (deg/s)"),
    )
    for axis, (columns, title, ylabel) in zip(axes, sensor_groups):
        signal = merged[list(columns)].copy()
        signal.index.name = "Time"
        signal = signal.reset_index().melt(
            id_vars="Time", var_name="Axis", value_name=ylabel
        )
        sns.lineplot(data=signal, x="Time", y=ylabel, hue="Axis", ax=axis)
        axis.set(title=title, xlabel="", ylabel=ylabel)
        axis.legend(title="Axis", ncols=3, loc="upper right")
    axes[-1].set_xlabel("Time")
    show_chart(fig)


def render_result(acc_df: pd.DataFrame, gyro_df: pd.DataFrame, artifacts: dict) -> None:
    try:
        merged = merge_and_resample(acc_df, gyro_df)
        labels, probabilities, processed, overall = predict_from_raw(
            acc_df, gyro_df, artifacts
        )
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        if isinstance(error, FileNotFoundError):
            st.info("From the repository root, run `python app/export_model.py` first.")
        return

    st.subheader("Raw sensor signals")
    plot_raw_sensors(merged)

    first, second = st.columns(2)
    first.metric("Overall exercise (majority vote)", overall.replace("_", " ").title())
    second.metric("Prediction windows", len(labels))

    st.subheader("Average class confidence")
    average_probabilities = probabilities.mean().sort_values(ascending=False)
    confidence = average_probabilities.rename("Confidence").rename_axis("Exercise").reset_index()
    fig, axis = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    sns.barplot(
        data=confidence,
        x="Confidence",
        y="Exercise",
        color="#2F7D6D",
        ax=axis,
    )
    axis.set(
        xlim=(0, 1.12),
        xlabel="Average probability",
        ylabel="",
        title="Average class confidence",
    )
    for container in axis.containers:
        axis.bar_label(
            container,
            labels=[f"{value:.1%}" for value in container.datavalues],
            padding=4,
        )
    show_chart(fig)

    st.subheader("Exercise by window")
    timeline = pd.DataFrame(
        {"Window": range(1, len(labels) + 1), "Exercise": labels.to_numpy()}
    )
    fig, axis = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    sns.scatterplot(
        data=timeline,
        x="Window",
        y="Exercise",
        hue="Exercise",
        s=90,
        hue_order=probabilities.columns,
        legend=False,
        ax=axis,
    )
    axis.set(title="Per-window prediction timeline", xlabel="Window", ylabel="")
    show_chart(fig)

    with st.expander("How the signal was filtered"):
        filtered = apply_low_pass(merged)
        fig, axes = plt.subplots(
            3, 1, figsize=(CHART_WIDTH, CHART_HEIGHT * 1.8), sharex=True
        )
        for index, (axis, column) in enumerate(zip(axes, ("acc_x", "acc_y", "acc_z"))):
            signal = pd.DataFrame(
                {
                    "Time": merged.index.append(filtered.index),
                    "Acceleration (g)": pd.concat(
                        [merged[column], filtered[column]], ignore_index=True
                    ),
                    "Filter": ["Before filter"] * len(merged)
                    + ["Low-pass"] * len(filtered),
                }
            )
            sns.lineplot(
                data=signal,
                x="Time",
                y="Acceleration (g)",
                hue="Filter",
                ax=axis,
                legend=index == 0,
            )
            axis.set(title=column, ylabel="Acceleration (g)", xlabel="")
            if index:
                legend = axis.get_legend()
                if legend:
                    legend.remove()
        axes[-1].set_xlabel("Time")
        fig.suptitle("Accelerometer before and after low-pass filtering")
        show_chart(fig)
        st.caption(f"Processed feature windows: {len(processed)}")


def render_overview() -> None:
    st.subheader("Overview")
    st.write(
        "Wrist-worn sensors capture motion patterns that are difficult to identify "
        "from raw numbers alone. Smart Gym Coach classifies a recorded set as one of "
        "the exercises represented in the training data. It turns synchronized "
        "accelerometer and gyroscope CSV files into per-window and overall predictions."
    )
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          SensorCSVs -> "200 ms resampling" -> "Filter + PCA" -> "Time + frequency features"
          -> "K-means cluster" -> "Random forest" -> Prediction
        }
        """
    )


def render_how_it_works() -> None:
    st.subheader("How it works")
    st.write(
        "The app merges the sensor streams onto a 200 ms grid, interpolates gaps, "
        "applies the training low-pass filter, and reuses the PCA and K-means objects "
        "fitted during export. Rolling and Fourier features are then passed to the "
        "random-forest classifier."
    )
    scores_path = ARTIFACT_DIR / "feature_set_accuracy.json"
    variance_path = ARTIFACT_DIR / "pca_explained_variance.json"
    if not scores_path.is_file() or not variance_path.is_file():
        st.info("Run `python app/export_model.py` to generate the model diagnostics.")
        return

    variance = cached_json(str(variance_path))
    variance_frame = pd.DataFrame(
        {
            "Component": [f"PC {index + 1}" for index in range(len(variance))],
            "Explained variance": variance,
        }
    )
    fig, axis = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    sns.barplot(
        data=variance_frame,
        x="Explained variance",
        y="Component",
        color="#2F7D6D",
        ax=axis,
    )
    axis.set(title="PCA explained variance", xlabel="Explained variance ratio", ylabel="")
    show_chart(fig)

    scores = cached_json(str(scores_path))
    accuracy = pd.DataFrame(
        {"Feature set": list(scores.keys()), "Accuracy": list(scores.values())}
    )
    fig, axis = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
    sns.barplot(
        data=accuracy,
        x="Accuracy",
        y="Feature set",
        color="#2F7D6D",
        ax=axis,
    )
    axis.set(
        xlim=(0, 1.12),
        title="Random-forest accuracy by feature set",
        ylabel="",
        xlabel="Accuracy",
    )
    for container in axis.containers:
        axis.bar_label(
            container,
            labels=[f"{value:.1%}" for value in container.datavalues],
            padding=4,
        )
    show_chart(fig)
    st.caption(
        "Held-out, stratified 25% split (random_state=42); the exported classifier "
        "uses the complete feature set."
    )


def main() -> None:
    st.set_page_config(
        page_title="Smart Gym Coach",
        page_icon="🏋️",
        layout="wide",
    )
    st.title("Smart Gym Coach")
    st.caption("Upload your set, find out which exercise you did.")

    with st.sidebar:
        st.header("Choose your data")
        mode = st.radio("Input mode", ("Demo mode", "Upload"))

    overview_tab, try_tab, how_tab, about_tab = st.tabs(
        ("Overview", "Try it", "How it works", "About")
    )
    with overview_tab:
        render_overview()

    with try_tab:
        st.info(
            "CSV format: include an `epoch (ms)` timestamp column and three sensor "
            "axis columns. Accelerometer columns may be `x-axis (g)`, `y-axis (g)`, "
            "`z-axis (g)`; gyroscope columns may be `x-axis (deg/s)`, "
            "`y-axis (deg/s)`, `z-axis (deg/s)`. The app also accepts `acc_x` through "
            "`acc_z` and `gyro_x` through `gyro_z`. Upload matching recordings with "
            "at least four seconds of data."
        )
        acc_df = gyro_df = None
        if mode == "Demo mode":
            sample_paths = sorted(SAMPLE_DIR.glob("*_acc.csv"))
            if not sample_paths:
                st.error("No bundled samples found in app/sample_data/.")
            else:
                selected_sample = st.selectbox(
                    "Bundled recording",
                    sample_paths,
                    format_func=lambda path: path.stem.replace("_acc", "")
                    .replace("_", " ")
                    .title(),
                )
                gyro_path = selected_sample.with_name(
                    selected_sample.name.replace("_acc.csv", "_gyro.csv")
                )
                if not gyro_path.is_file():
                    st.error(f"Missing paired gyroscope sample: {gyro_path.name}")
                else:
                    acc_df, gyro_df = cached_sample_files(
                        str(selected_sample), str(gyro_path)
                    )
        else:
            acc_upload = st.file_uploader(
                "Accelerometer CSV", type=["csv"], key="accelerometer"
            )
            gyro_upload = st.file_uploader(
                "Gyroscope CSV", type=["csv"], key="gyroscope"
            )
            acc_df = read_uploaded_csv(acc_upload, "accelerometer")
            gyro_df = read_uploaded_csv(gyro_upload, "gyroscope")

        if acc_df is not None and gyro_df is not None:
            try:
                artifacts = cached_artifacts()
            except FileNotFoundError as error:
                st.error(f"Model artifacts are not available: {error}")
                st.info("From the repository root, run `python app/export_model.py`.")
            else:
                render_result(acc_df, gyro_df, artifacts)

    with how_tab:
        render_how_it_works()

    with about_tab:
        st.subheader("About")
        st.markdown(
            """
            - **Dataset:** MetaMotion wrist accelerometer and gyroscope recordings
              labelled bench, squat, row, overhead press, deadlift, and rest.
            - **Tools:** Python, pandas, SciPy, scikit-learn, Matplotlib, Seaborn, and Streamlit.
            - **Limitations:** a small number of participants, one sensor device,
              and overlapping feature windows mean results may not generalize to
              every person or recording setup. This app does not count repetitions.
            - **Project:** [GitHub repository](https://github.com/rudra2609-06/fitness_tracker_quantified_self.git)
            """
        )


if __name__ == "__main__":
    main()
