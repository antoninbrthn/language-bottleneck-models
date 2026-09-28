import ast
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

from src.utils.config import PROJECT_ROOT


def _clean_name(raw: str) -> str:
    name = raw.split("/")[-1]
    for suf in ("-Instruct", "-it"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def compute_auc_from_logits(logits_data, true_labels):
    """Compute AUC from logits and true labels."""
    assert len(logits_data) == len(
        true_labels
    ), f"Logits and true labels must have the same length. Got {len(logits_data)} logits and {len(true_labels)} labels."
    probs = []
    binary_true_labels = []

    bad_logits = 0
    for logits_dict, true_label in zip(logits_data, true_labels):
        logit_values = [logits_dict["Yes"], logits_dict["No"]]
        if None in logit_values:
            bad_logits += 1
            continue
        probs_yes = softmax(logit_values)[0]  # Probability of 'Yes'
        probs.append(probs_yes)
        binary_true_labels.append(1 if true_label == "Yes" else 0)
    if bad_logits > 0:
        print(f"[WARNING] Got bad logits for {bad_logits}/{len(logits_data)} entries")
    return roc_auc_score(binary_true_labels, probs)


def get_run_auc(run):
    """Try to compute AUC from exported results"""
    export_dir = run.config.get("export_dir")
    if not export_dir:
        print("No export_dir in run config")
        return None

    results_path = os.path.join(PROJECT_ROOT, export_dir, "evaluation_results.csv")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return None

    results_df = pd.read_csv(results_path)
    logits_data = []
    true_labels = []

    if (
        "output_decoder_outputs" in results_df.columns
        and "true_label" in results_df.columns
    ):
        # Parse the decoder outputs
        for idx, row in results_df.iterrows():
            output_str = row["output_decoder_outputs"]
            true_label = ast.literal_eval(row["true_label"])

            try:
                outputs = ast.literal_eval(output_str)
                logits = [
                    c["logits"]
                    for c in outputs
                    if isinstance(c, dict) and "logits" in c
                ]
                if (len(logits) > 0) and None not in logits:
                    logits_data.extend(logits)
                    # Convert true_label to binary (assuming 'Yes'=1, 'No'=0)
                    true_labels.extend(true_label)
            except:
                continue
    elif "logit" in results_df.columns:
        for idx, row in results_df.iterrows():
            output_str = row["logit"]
            true_label = ast.literal_eval(row["true_label"])

            if "nan" in output_str:
                output_str = output_str.replace("nan", "None")
            logits = ast.literal_eval(output_str)
            if (len(logits) > 0) and None not in logits:
                logits_data.extend(logits)
                # Convert true_label to binary (assuming 'Yes'=1, 'No'=0)
                true_labels.extend(true_label)
    elif "output_logits" in results_df.columns:
        for idx, row in results_df.iterrows():
            output_str = row["output_logits"]
            true_label = ast.literal_eval(row["true_label"])

            logits = ast.literal_eval(output_str)
            if (len(logits) > 0) and None not in logits:
                logits_data.extend(logits)
                # Convert true_label to binary (assuming 'Yes'=1, 'No'=0)
                true_labels.extend(true_label)
    elif "output_full" in results_df.columns:
        for idx, row in results_df.iterrows():
            output_str = row["output_full"]
            outputs = ast.literal_eval(output_str)
            true_label = ast.literal_eval(row["true_label"])

            logits = [o["logits"] for o in outputs]
            if (len(logits) > 0) and None not in logits:
                logits_data.extend(logits)
                true_labels.extend(true_label)
    else:
        raise Exception("No logits column in results")
    if logits_data and true_labels:
        assert len(logits_data) == len(
            true_labels
        ), f"Logits and true labels must have the same length. Got {len(logits_data)} logits and {len(true_labels)} labels."
        auc = compute_auc_from_logits(logits_data, true_labels)
    return auc
