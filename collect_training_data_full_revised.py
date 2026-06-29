import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

PROM_URL_DEFAULT = "http://localhost:9090"
METRICS = {
    "cpu": 'sum(rate(container_cpu_usage_seconds_total{namespace="%s"}[1m])) by (pod)',
    "mem": 'sum(container_memory_usage_bytes{namespace="%s"}) by (pod)'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Prometheus data and build CGNN train/test CSVs."
    )
    parser.add_argument("--prom-url", default=PROM_URL_DEFAULT,
                        help="Prometheus base URL")
    parser.add_argument("--namespace", default="default",
                        help="Prometheus namespace to query")
    parser.add_argument("--start", default=None,
                        help="ISO start timestamp for collection (inclusive). Defaults to 30 minutes ago.")
    parser.add_argument("--end", default=None,
                        help="ISO end timestamp for collection (inclusive). Defaults to now.")
    parser.add_argument("--step", default="15s",
                        help="Prometheus query step interval")
    parser.add_argument("--train-end", default=None,
                        help="ISO end timestamp for training data. If unset, split_ratio is used.")
    parser.add_argument("--test-start", default=None,
                        help="ISO start timestamp for test data. If unset, split_ratio is used.")
    parser.add_argument("--split-ratio", type=float, default=0.7,
                        help="Fraction of rows to use for training when no explicit train/test split is provided.")
    parser.add_argument("--anomaly-interval", action="append", default=[],
                        help="Anomaly interval in the format start_iso,end_iso. Can be repeated.")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for generated CSV files")
    return parser.parse_args()


def parse_timestamp(value):
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_anomaly_intervals(interval_values):
    intervals = []
    for value in interval_values:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid anomaly interval format: {value}")
        start = parse_timestamp(parts[0].strip())
        end = parse_timestamp(parts[1].strip())
        if start is None or end is None:
            raise ValueError(f"Invalid anomaly interval timestamps: {value}")
        if start > end:
            raise ValueError(f"Anomaly interval start must be <= end: {value}")
        intervals.append((start, end))
    return intervals


def collect_metrics(prom_url, namespace, start, end, step):
    base_params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "step": step,
    }
    series = {}
    for short, query_template in METRICS.items():
        query = query_template % namespace
        params = base_params.copy()
        params["query"] = query

        resp = requests.get(f"{prom_url}/api/v1/query_range", params=params)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("result", [])

        for ts in data:
            pod = ts["metric"].get("pod") or ts["metric"].get("instance") or "unknown"
            points = ts.get("values", [])
            if not points:
                continue
            s = pd.Series(
                [float(v) for _, v in points],
                index=[datetime.fromtimestamp(float(t), tz=timezone.utc) for t, _ in points]
            )
            col = f"{pod}_{short}"
            series[col] = s
    return series


def build_label_array(test_df, anomaly_intervals):
    labels = np.zeros((len(test_df), 1), dtype=np.float32)
    for start, end in anomaly_intervals:
        mask = (test_df.index >= start) & (test_df.index <= end)
        labels[mask.to_numpy(), 0] = 1.0
    return labels


def main():
    args = parse_args()

    if args.start is None:
        args.start = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    if args.end is None:
        args.end = datetime.now(timezone.utc).isoformat()

    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if start >= end:
        raise SystemExit("start must be before end")

    anomaly_intervals = parse_anomaly_intervals(args.anomaly_interval)

    series = collect_metrics(args.prom_url, args.namespace, start, end, args.step)
    df = pd.DataFrame(series).sort_index()
    df = df.ffill().bfill().fillna(0)

    if df.empty:
        raise SystemExit("No data was collected. Check Prometheus queries and time range.")

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Collected data columns:", list(df.columns))
    print(df.head())

    raw_output_path = os.path.join(output_dir, "combined_data.csv")
    df.to_csv(raw_output_path, header=False, index=False)

    labeled_output_path = os.path.join(output_dir, "combined_data_labeled.csv")
    df.index.name = "timestamp"
    df.reset_index().to_csv(labeled_output_path, index=False)

    if args.train_end is not None or args.test_start is not None:
        train_end = parse_timestamp(args.train_end)
        test_start = parse_timestamp(args.test_start)

        if train_end is not None and test_start is not None and train_end >= test_start:
            raise SystemExit("train-end must be before test-start")

        if train_end is not None and test_start is not None:
            train_df = df[df.index <= train_end]
            test_df = df[df.index >= test_start]
        elif train_end is not None:
            train_df = df[df.index <= train_end]
            test_df = df[df.index > train_end]
        else:
            train_df = df[df.index < test_start]
            test_df = df[df.index >= test_start]
    else:
        split_index = int(len(df) * args.split_ratio)
        if split_index < 2 or split_index >= len(df):
            raise SystemExit("split-ratio results in too small train/test datasets")
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

    if train_df.empty or test_df.empty:
        raise SystemExit("Train or test dataset is empty after splitting")

    scaler = MinMaxScaler()
    X_train = train_df.to_numpy(dtype=np.float32)
    X_test = test_df.to_numpy(dtype=np.float32)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_path = os.path.join(output_dir, "train_array.csv")
    test_path = os.path.join(output_dir, "test_array.csv")
    label_path = os.path.join(output_dir, "anomaly_label_array.csv")

    pd.DataFrame(X_train_scaled).to_csv(train_path, header=False, index=False)
    pd.DataFrame(X_test_scaled).to_csv(test_path, header=False, index=False)

    labels = build_label_array(test_df, anomaly_intervals)
    pd.DataFrame(labels).to_csv(label_path, header=False, index=False)

    print("Wrote:")
    print(f"  {raw_output_path}")
    print(f"  {labeled_output_path}")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {label_path}")
    if not anomaly_intervals:
        print("Warning: anomaly_label_array.csv contains all zeros because no intervals were provided.")
    else:
        print("Anomaly intervals used:")
        for start, end in anomaly_intervals:
            print(f"  {start.isoformat()} to {end.isoformat()}")


if __name__ == "__main__":
    main()
