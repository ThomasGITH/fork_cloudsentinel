import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---------- Change these paths ----------
COMBINED_CSV = "Monitoring_run1/Scenario1/scenario1_prometheus_output/combined_data_labeled.csv"
TEST_DATA_CSV = "Monitoring_run1/Scenario1/scenario1_prometheus_output/test_array.csv"
TEST_LABELS_CSV = "Monitoring_run1/Scenario1/scenario1_prometheus_output/anomaly_label_array.csv"
PREDICTIONS_CSV = "Monitoring_run1/Scenario1/ManualAnalysis/scenario_prediction_output.csv"
OUTPUT_CSV = "Monitoring_run1/Scenario1/ManualAnalysis/aligned_evaluation.csv"

LOOKBACK = 100
# ----------------------------------------

# The combined file has headers and a timestamp column.
# sep=None lets pandas infer whether the file uses commas or tabs.
combined = pd.read_csv(COMBINED_CSV, sep=None, engine="python")

# The CloudSentinel input/test data normally has no header.
test_data = pd.read_csv(TEST_DATA_CSV, header=None)

# Labels normally contain one binary value per test-data row.
labels = pd.read_csv(TEST_LABELS_CSV, header=None).iloc[:, 0]

# CloudSentinel prediction output created by your code change.
predictions = pd.read_csv(PREDICTIONS_CSV)

# Validate the expected relationship.
n_train = len(combined) - len(test_data)

if n_train < 0:
    raise ValueError(
        "Combined CSV is shorter than test-data CSV. "
        "Check that the correct files were selected."
    )

if len(labels) != len(test_data):
    raise ValueError(
        f"Test labels have {len(labels)} rows, but test data has "
        f"{len(test_data)} rows. They should normally match before alignment."
    )

if len(predictions) != len(test_data) - LOOKBACK:
    raise ValueError(
        f"Predictions have {len(predictions)} rows, but expected "
        f"{len(test_data) - LOOKBACK} rows from test data and lookback "
        f"{LOOKBACK}. Check that the uploaded test CSV matches this run."
    )

# Extract only the test portion from the combined CSV.
test_with_timestamps = combined.iloc[n_train:].reset_index(drop=True)

if len(test_with_timestamps) != len(test_data):
    raise ValueError(
        "The extracted test portion does not have the same number of rows "
        "as the test-data CSV. Check whether the combined CSV contains "
        "exactly training data followed by test data."
    )

# Remove the first 100 rows from timestamps and labels because those rows
# served as model context; the first prediction corresponds to test row 100.
aligned_timestamps = test_with_timestamps["timestamp"].iloc[LOOKBACK:].reset_index(drop=True)
aligned_labels = labels.iloc[LOOKBACK:].reset_index(drop=True)
aligned_predictions = predictions["A_Pred_Global"].reset_index(drop=True)
aligned_scores = predictions["A_Score_Global"].reset_index(drop=True)

# Combine everything in one inspection/evaluation file.
evaluation = pd.DataFrame({
    "timestamp": aligned_timestamps,
    "ground_truth_label": aligned_labels,
    "anomaly_score": aligned_scores,
    "predicted_anomaly": aligned_predictions,
})

evaluation.to_csv(OUTPUT_CSV, index=False)

# Calculate confusion-matrix values and metrics.
y_true = evaluation["ground_truth_label"]
y_pred = evaluation["predicted_anomaly"]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

print("Aligned rows:", len(evaluation))
print()
print("True positives (TP): ", tp)
print("False positives (FP):", fp)
print("True negatives (TN): ", tn)
print("False negatives (FN):", fn)
print()
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1 score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
print()
print("Saved aligned timestamp-level evaluation to:")
print(OUTPUT_CSV)
