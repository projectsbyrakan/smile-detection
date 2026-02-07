import streamlit as st
import mlflow
import pandas as pd

st.set_page_config(page_title="Model Report", layout="wide")
st.title("Model Report")

# Load runs from MLflow
st.title("Model Report")

runs = mlflow.search_runs(experiment_names=["smile-classifier"])

if runs.empty:
    st.warning("No training runs found. Train models first.")
    st.stop()

# Comparison table
st.header("Model Comparison")

metrics = []
for _, run in runs.iterrows():
    metrics.append({
        "Model": run.get("params.architecture", "Unknown"),
        "Test Accuracy": run.get("metrics.final_test_acc", None),
        "F1 Score": run.get("metrics.final_f1_score", None),
        "Test Loss": run.get("metrics.final_test_loss", None),
        "Best Val Accuracy": run.get("metrics.best_val_accuracy", None),
    })

comparison = pd.DataFrame(metrics)
st.dataframe(comparison, use_container_width=True, hide_index=True)

# Training history charts
st.header("Training History")

for _, run in runs.iterrows():
    run_name = run.get("params.architecture", "Unknown")
    run_id = run["run_id"]

    st.subheader(run_name)

    client = mlflow.tracking.MlflowClient()

    # Get metric history
    train_loss = client.get_metric_history(run_id, "train_loss")
    val_loss = client.get_metric_history(run_id, "val_loss")
    train_acc = client.get_metric_history(run_id, "train_acc")
    val_acc = client.get_metric_history(run_id, "val_acc")

    if train_loss:
        col1, col2 = st.columns(2)

        with col1:
            loss_df = pd.DataFrame({
                "Epoch": [m.step for m in train_loss],
                "Train Loss": [m.value for m in train_loss],
                "Val Loss": [m.value for m in val_loss],
            }).set_index("Epoch")
            st.line_chart(loss_df)
            st.caption("Loss")

        with col2:
            acc_df = pd.DataFrame({
                "Epoch": [m.step for m in train_acc],
                "Train Acc": [m.value for m in train_acc],
                "Val Acc": [m.value for m in val_acc],
            }).set_index("Epoch")
            st.line_chart(acc_df)
            st.caption("Accuracy")

# Hyperparameters
st.header("Training Parameters")

params = []
for _, run in runs.iterrows():
    p = {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")}
    params.append(p)

params_df = pd.DataFrame(params)
st.dataframe(params_df, use_container_width=True, hide_index=True)