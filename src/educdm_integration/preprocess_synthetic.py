"""
Reproduce preprocessing done by EduCDM for synthetic dataset
"""

#!/usr/bin/env python
# preprocess_synthetic.py

import argparse
import os
import pandas as pd
import numpy as np
import sys
import json
import random
import ast
from collections import defaultdict

from src.data.eedi_filtered import EediFilteredLoader
from src.data.xes3g5m_filtered import XES3G5MFilteredLoader
from src.utils.config import DATA_DIR, PROJECT_ROOT

# Add the path to the EduCDM library
# sys.path.append('third_party/EduCDM')


def custom_train_test_split(df, test_ratio=0.2, valid_ratio=0.1, random_seed=42):
    """
    Split each student's interactions into train/valid/test sets, ensuring each student has enough data.

    Args:
        df: DataFrame with student interactions
        test_ratio: Ratio of interactions per student for test set
        valid_ratio: Ratio of interactions per student for validation set
        random_seed: Random seed for reproducibility

    Returns:
        train_df, valid_df, test_df: DataFrames for training, validation, and testing
    """
    random.seed(random_seed)

    # Count interactions per student
    student_counts = df.groupby("student_id").size()

    # DISCARDED
    # Filter students with enough interactions
    # valid_students = student_counts[student_counts >= min_interactions_per_student]
    # print(f"Students with >= {min_interactions_per_student} interactions: {len(valid_students)} out of {len(student_counts)}")
    # Filter data to only include students with enough interactions
    # df_filtered = df[df['student_id'].isin(valid_students.index)]

    valid_students = student_counts
    df_filtered = df

    train_dfs = []
    valid_dfs = []
    test_dfs = []

    # Split each student's interactions
    for student_id in valid_students.index:
        student_data = df_filtered[df_filtered["student_id"] == student_id].copy()

        # Shuffle the student's interactions
        student_data = student_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        n_interactions = len(student_data)
        n_test = int(n_interactions * test_ratio)
        n_valid = int(n_interactions * valid_ratio)
        n_train = n_interactions - n_test - n_valid

        # Split the student's data
        train_data = student_data.iloc[:n_train]
        valid_data = student_data.iloc[n_train : n_train + n_valid]
        test_data = student_data.iloc[n_train + n_valid :]

        train_dfs.append(train_data)
        valid_dfs.append(valid_data)
        test_dfs.append(test_data)

    # Combine all splits
    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Report
    print(f"Total students: {len(valid_students)}")
    print(f"Total rows: {len(df_filtered)}, train rows: {len(train_df)}, valid rows: {len(valid_df)}, test rows: {len(test_df)}")
    print(
        f"Average interactions per student - train: {len(train_df)/len(valid_students):.1f}, valid: {len(valid_df)/len(valid_students):.1f}, test: {len(test_df)/len(valid_students):.1f}"
    )

    return train_df, valid_df, test_df


def create_irt_files(train_df, valid_df, test_df, output_dir, dataset_name):
    """
    Create files for IRT models (GDIRT, EMIRT).

    Args:
        train_df, valid_df, test_df: DataFrames for each split
        output_dir: Directory to save files
        dataset_name: Name for the dataset
    """
    print("Creating IRT files...")

    # Create output directory
    irt_dir = os.path.join(output_dir, dataset_name, "irt")
    os.makedirs(irt_dir, exist_ok=True)

    # Map student_id and question_id to consecutive integers starting from 1
    all_students = pd.concat([train_df, valid_df, test_df])["student_id"].unique()
    all_questions = pd.concat([train_df, valid_df, test_df])["question_id"].unique()

    student_id_map = {int(old_id): new_id for new_id, old_id in enumerate(all_students, 1)}
    question_id_map = {int(old_id): new_id for new_id, old_id in enumerate(all_questions, 1)}

    # Function to transform data
    def transform_df(df, split_name):
        transformed = df.copy()
        transformed["user_id"] = transformed["student_id"].astype(int).map(student_id_map)
        transformed["item_id"] = transformed["question_id"].astype(int).map(question_id_map)
        transformed["score"] = transformed["is_correct"].astype(int)

        # Save to CSV
        output_file = os.path.join(irt_dir, f"{split_name}.csv")
        transformed[["user_id", "item_id", "score"]].to_csv(output_file, index=False)
        print(f"Saved {split_name} data to {output_file}")

        return transformed

    # Transform and save all splits
    train_transformed = transform_df(train_df, "train")
    valid_transformed = transform_df(valid_df, "valid")
    test_transformed = transform_df(test_df, "test")

    # Save mapping info
    mapping_info = {
        "student_id_map": student_id_map,
        "question_id_map": question_id_map,
        "n_students": len(all_students),
        "n_questions": len(all_questions),
    }

    with open(os.path.join(irt_dir, "mapping_info.json"), "w") as f:
        json.dump(mapping_info, f, indent=2)

    return irt_dir, mapping_info


def create_dina_files(train_df, valid_df, test_df, output_dir, dataset_name):
    """
    Create files for DINA models (EMDINA, GDDINA).
    """
    print("Creating DINA files...")

    # Create output directory
    dina_dir = os.path.join(output_dir, dataset_name, "dina")
    os.makedirs(dina_dir, exist_ok=True)

    # Clean data first - remove any rows with NaN values
    print(f"Before cleaning - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    train_df = train_df.dropna().reset_index(drop=True)
    valid_df = valid_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    print(f"After cleaning - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Map student_id and question_id to consecutive integers starting from 0
    all_students = pd.concat([train_df, valid_df, test_df])["student_id"].unique()
    all_questions = pd.concat([train_df, valid_df, test_df])["question_id"].unique()

    # Remove any NaN values from the unique arrays
    all_students = all_students[~pd.isna(all_students)]
    all_questions = all_questions[~pd.isna(all_questions)]

    student_id_map = {int(old_id): new_id for new_id, old_id in enumerate(sorted(all_students))}
    question_id_map = {int(old_id): new_id for new_id, old_id in enumerate(sorted(all_questions))}

    # Create Q-matrix from constructs
    all_constructs = set()
    question_constructs = {}

    for _, row in pd.concat([train_df, valid_df, test_df]).iterrows():
        # if pd.isna(row['question_id']) or pd.isna(row['constructs_ids']):
        #     continue

        question_id = int(row["question_id"])
        constructs = ast.literal_eval(row["constructs_ids"]) if isinstance(row["constructs_ids"], str) else row["constructs_ids"]
        question_constructs[question_id] = constructs
        all_constructs.update(constructs)

    # Sort constructs for consistent ordering
    all_constructs = sorted(list(all_constructs))
    construct_id_map = {int(old_id): new_id for new_id, old_id in enumerate(all_constructs)}

    # Create Q-matrix
    q_matrix = np.zeros((len(all_questions), len(all_constructs)), dtype=int)
    for question_id, constructs in question_constructs.items():
        if question_id in [int(q) for q in all_questions]:
            new_question_id = question_id_map[question_id]
            for construct_id in constructs:
                if construct_id in construct_id_map:
                    new_construct_id = construct_id_map[construct_id]
                    q_matrix[new_question_id, new_construct_id] = 1

    # Save Q-matrix
    q_matrix_file = os.path.join(dina_dir, "q_m.csv")
    np.savetxt(q_matrix_file, q_matrix, delimiter=",", fmt="%d")
    print(f"Saved Q-matrix to {q_matrix_file}")

    # Function to create JSON data
    def create_json_data(df, split_name):
        if len(df) == 0:
            print(f"Warning: {split_name} split is empty!")
            return []

        json_data = []
        for _, row in df.iterrows():
            if pd.isna(row["student_id"]) or pd.isna(row["question_id"]) or pd.isna(row["is_correct"]):
                continue

            student_id = student_id_map[int(row["student_id"])]
            question_id = question_id_map[int(row["question_id"])]
            score = int(row["is_correct"])

            json_data.append({"user_id": student_id, "item_id": question_id, "score": score})

        # Save to JSON
        output_file = os.path.join(dina_dir, f"{split_name}_data.json")
        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved {split_name} data to {output_file} ({len(json_data)} rows)")

        return json_data

    # Create JSON files for train and test
    train_json = create_json_data(train_df, "train")
    test_json = create_json_data(test_df, "test")

    # Save mapping info
    mapping_info = {
        "student_id_map": student_id_map,
        "question_id_map": question_id_map,
        "construct_id_map": construct_id_map,
        "n_students": len(all_students),
        "n_questions": len(all_questions),
        "n_constructs": len(all_constructs),
    }

    with open(os.path.join(dina_dir, "mapping_info.json"), "w") as f:
        json.dump(mapping_info, f, indent=2)

    return dina_dir, mapping_info


def create_ncdm_files(train_df, valid_df, test_df, output_dir, dataset_name):
    """
    Create files for NCDM models.
    """
    print("Creating NCDM files...")

    # Create output directory
    ncdm_dir = os.path.join(output_dir, dataset_name, "ncdm")
    os.makedirs(ncdm_dir, exist_ok=True)

    # Clean data first - remove any rows with NaN values
    print(f"Before cleaning - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    train_df = train_df.dropna().reset_index(drop=True)
    valid_df = valid_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    print(f"After cleaning - Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Map student_id and question_id to consecutive integers starting from 1
    all_students = pd.concat([train_df, valid_df, test_df])["student_id"].unique()
    all_questions = pd.concat([train_df, valid_df, test_df])["question_id"].unique()

    # Remove any NaN values from the unique arrays
    all_students = all_students[~pd.isna(all_students)]
    all_questions = all_questions[~pd.isna(all_questions)]

    student_id_map = {int(old_id): new_id for new_id, old_id in enumerate(sorted(all_students), 1)}
    question_id_map = {int(old_id): new_id for new_id, old_id in enumerate(sorted(all_questions), 1)}

    # Create item.csv with knowledge codes
    all_constructs = set()
    item_knowledge = {}

    for _, row in pd.concat([train_df, valid_df, test_df]).iterrows():
        # if pd.isna(row['question_id']) or pd.isna(row['constructs_ids']):
        # continue
        question_id = int(row["question_id"])
        constructs = ast.literal_eval(row["constructs_ids"]) if isinstance(row["constructs_ids"], str) else row["constructs_ids"]
        item_knowledge[question_id] = constructs
        all_constructs.update(constructs)

    # Sort constructs for consistent ordering
    all_constructs = sorted(list(all_constructs))
    construct_id_map = {int(old_id): new_id for new_id, old_id in enumerate(all_constructs, 1)}

    # Create item.csv
    item_data = []
    for question_id in sorted(all_questions):
        question_id = int(question_id)
        if question_id in item_knowledge:
            new_question_id = question_id_map[int(question_id)]
            constructs = item_knowledge[int(question_id)]
            knowledge_codes = [construct_id_map[c] for c in constructs]
            item_data.append({"item_id": new_question_id, "knowledge_code": str(knowledge_codes)})

    item_df = pd.DataFrame(item_data)
    item_file = os.path.join(ncdm_dir, "item.csv")
    item_df.to_csv(item_file, index=False)
    print(f"Saved item.csv to {item_file}")

    # Function to transform data
    def transform_df(df, split_name):
        if len(df) == 0:
            print(f"Warning: {split_name} split is empty!")
            return pd.DataFrame(columns=["user_id", "item_id", "score"])

        transformed = df.copy()
        # Convert to int first to handle any float values
        transformed["user_id"] = transformed["student_id"].astype(int).map(student_id_map)
        transformed["item_id"] = transformed["question_id"].astype(int).map(question_id_map)
        transformed["score"] = transformed["is_correct"].astype(int)

        # Remove any rows where mapping failed (resulted in NaN)
        transformed = transformed.dropna()

        # Save to CSV
        output_file = os.path.join(ncdm_dir, f"{split_name}.csv")
        transformed[["user_id", "item_id", "score"]].to_csv(output_file, index=False)
        print(f"Saved {split_name} data to {output_file} ({len(transformed)} rows)")

        return transformed

    # Transform and save all splits
    train_transformed = transform_df(train_df, "train")
    valid_transformed = transform_df(valid_df, "valid")
    test_transformed = transform_df(test_df, "test")

    # Save mapping info
    mapping_info = {
        "student_id_map": student_id_map,
        "question_id_map": question_id_map,
        "construct_id_map": construct_id_map,
        "n_students": len(all_students),
        "n_questions": len(all_questions),
        "n_constructs": len(all_constructs),
    }

    with open(os.path.join(ncdm_dir, "mapping_info.json"), "w") as f:
        json.dump(mapping_info, f, indent=2)

    return ncdm_dir, mapping_info


def get_eedi_data(
    input_file=None,
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    min_answer_seconds=5,
    session_gap_minutes=15,
    min_session_questions=40,
):
    """
    Preprocess the Eedi dataset to make it compatible with PyKT.
    Args:
        input_file: Path to the Eedi answer.csv (not used, kept for API compatibility)
        output_dir: Directory to save processed files
        dataset_name: Name for the dataset
        max_n_trajectories: Maximum number of student trajectories to include in TRAIN/VALID (not counting test students)
        min_trajectory_length: Minimum number of questions per student
        max_trajectory_length: Maximum number of questions per student
        n_test_students: Number of students to reserve for test set
        min_answer_seconds: Minimum answer latency to retain (seconds)
        session_gap_minutes: Maximum allowed gap between two consecutive answers before a new session starts (minutes)
        min_session_questions: Minimum questions required in a (user, session) group
    """

    print(f"Preprocessing Eedi data with EediFilteredLoader...")

    # Create output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # Set up loader
    loader = EediFilteredLoader(
        data_path=(os.path.dirname(input_file) if input_file else os.path.join(DATA_DIR, "eedi_private/data/")),
        min_answer_seconds=min_answer_seconds,
        session_gap_minutes=session_gap_minutes,
        min_session_questions=min_session_questions,
        max_n_trajectories=max_n_trajectories,
        min_trajectory_length=min_trajectory_length,
        max_trajectory_length=max_trajectory_length,
    )
    # Load and preprocess data
    df = loader.load_data()
    print(f"Filtered Eedi data shape: {df.shape}")
    construct_df = pd.read_csv(os.path.join(loader.data_path, "question-construct.csv"))
    # merge construct_df[["QuestionId", "ConstructId"]] with df
    df = df.merge(
        construct_df[["QuestionId", "ConstructId"]],
        on="QuestionId",
        how="inner",
    )
    df["ConstructIds"] = df["ConstructId"].apply(lambda x: f"[{x}]")
    # rename: QuestionId -> question_id, UserId -> student_id, ConstructId -> constructs_id, ConsutrctIds -> constructs_ids
    df = df.rename(
        columns={
            "QuestionId": "question_id",
            "UserId": "student_id",
            "ConstructId": "constructs_id",
            "ConstructIds": "constructs_ids",
            "IsCorrect": "is_correct",
        }
    )
    return df


def get_xes3g5m_data(
    input_file=None,
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    session_gap_minutes=10,
    min_session_questions=34,
):
    """Get preprocessed XES3G5M data."""
    print(f"Preprocessing XES3G5M data with XES3G5MFilteredLoader...")

    # Set up loader
    loader = XES3G5MFilteredLoader(
        session_gap_minutes=session_gap_minutes,
        min_session_questions=min_session_questions,
        max_n_trajectories=max_n_trajectories,
        min_trajectory_length=min_trajectory_length,
        max_trajectory_length=max_trajectory_length,
    )

    # Load and preprocess data
    df = loader.load_data()
    print(f"Filtered XES3G5M data shape: {df.shape}")

    # Rename columns to match expected format
    df = df.rename(
        columns={
            "user_id": "student_id",
            "correct": "is_correct",
            "construct_id": "constructs_id",
        }
    )
    df["constructs_ids"] = df["constructs_id"].apply(lambda x: [int(s) for s in str(x).split("_")])

    return df


def preprocess_synthetic_dataset(
    df,
    output_dir,
    dataset_name="synthetic",
    test_ratio=0.2,
    valid_ratio=0.1,
    min_interactions_per_student=10,  # discarded
    random_seed=42,
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
):
    """
    Preprocess a synthetic dataset to make it compatible with EduCDM models.

    Args:
        input_file: Path to the synthetic dataset CSV
        output_dir: Directory to save processed files
        dataset_name: Name for the dataset
        test_ratio: Ratio of students to include in test set
        valid_ratio: Ratio of students to include in validation set
        min_interactions_per_student: Minimum interactions required per student
        random_seed: Random seed for reproducibility
        max_n_trajectories: Maximum number of students to include
        min_trajectory_length: Minimum number of questions per student
        max_trajectory_length: Maximum number of questions per student
    """

    # Filter by min_trajectory_length and max_trajectory_length
    if min_trajectory_length is not None or max_trajectory_length is not None:
        student_counts = df.groupby("student_id").size()
        valid_students = student_counts
        if min_trajectory_length is not None:
            valid_students = valid_students[valid_students >= min_trajectory_length]
        if max_trajectory_length is not None:
            pass  # Will truncate later
        df = df[df["student_id"].isin(valid_students.index)]
        print(f"Filtered by trajectory length, new shape: {df.shape}")

    # Truncate trajectories if max_trajectory_length is set
    if max_trajectory_length is not None:
        df = df.sort_values(["student_id", "question_id"])
        df = df.groupby("student_id").head(max_trajectory_length).reset_index(drop=True)
        print(f"Truncated trajectories to max length {max_trajectory_length}, new shape: {df.shape}")

    # Filter by max_n_trajectories
    if max_n_trajectories is not None:
        unique_students = df["student_id"].unique()
        if len(unique_students) > max_n_trajectories:
            selected_students = unique_students[:max_n_trajectories]
            df = df[df["student_id"].isin(selected_students)]
            print(f"Filtered to {max_n_trajectories} students, new shape: {df.shape}")

    # Split the data by students
    train_df, valid_df, test_df = custom_train_test_split(df, test_ratio, valid_ratio, random_seed)

    # Create files for each model type
    irt_dir, irt_mapping = create_irt_files(train_df, valid_df, test_df, output_dir, dataset_name)
    dina_dir, dina_mapping = create_dina_files(train_df, valid_df, test_df, output_dir, dataset_name)
    ncdm_dir, ncdm_mapping = create_ncdm_files(train_df, valid_df, test_df, output_dir, dataset_name)

    # Create a summary file
    summary = {
        "dataset_name": dataset_name,
        "original_data_shape": df.shape,
        "train_shape": train_df.shape,
        "valid_shape": valid_df.shape,
        "test_shape": test_df.shape,
        "irt_dir": irt_dir,
        "dina_dir": dina_dir,
        "ncdm_dir": ncdm_dir,
        "mapping_info": {
            "irt": irt_mapping,
            "dina": dina_mapping,
            "ncdm": ncdm_mapping,
        },
    }

    summary_file = os.path.join(output_dir, dataset_name, "preprocessing_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Preprocessing complete! Summary saved to {summary_file}")
    print(f"Files created for:")
    print(f"  - IRT models: {irt_dir}")
    print(f"  - DINA models: {dina_dir}")
    print(f"  - NCDM models: {ncdm_dir}")


def load_data_for_preprocessing(args):
    if "synthetic" in args.dataset_name:
        input_file = os.path.join(PROJECT_ROOT, f"data/{args.dataset_name}/data.csv")
        print(f"Preprocessing {input_file}...")
        # Read the synthetic dataset
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
    elif "eedi" in args.dataset_name:
        print(f"Preprocessing Eedi data...")
        df = get_eedi_data(
            max_n_trajectories=args.max_n_trajectories,
            min_trajectory_length=args.min_trajectory_length,
            max_trajectory_length=args.max_trajectory_length,
            min_answer_seconds=5,
            session_gap_minutes=3,
            min_session_questions=40,
        )
    elif "xes3g5m" in args.dataset_name:
        print(f"Preprocessing XES3G5M data...")
        df = get_xes3g5m_data(
            max_n_trajectories=args.max_n_trajectories,
            min_trajectory_length=args.min_trajectory_length,
            max_trajectory_length=args.max_trajectory_length,
            session_gap_minutes=args.session_gap_minutes,
            min_session_questions=args.min_session_questions,
        )
    return df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="synthetic-11")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Ratio of logs for test set")
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Ratio of logs for validation set",
    )
    parser.add_argument(
        "--min_interactions_per_student",
        type=int,
        default=10,
        help="Minimum interactions required per student",
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--max_n_trajectories",
        type=int,
        default=None,
        help="Maximum number of students to include",
    )
    parser.add_argument(
        "--min_trajectory_length",
        type=int,
        default=None,
        help="Minimum number of questions per student",
    )
    parser.add_argument(
        "--max_trajectory_length",
        type=int,
        default=None,
        help="Maximum number of questions per student",
    )
    # add session_gap_minutes and min_session_questions
    parser.add_argument(
        "--session_gap_minutes",
        type=int,
        default=10,
        help="Maximum allowed gap between two consecutive answers before a new session starts (minutes)",
    )
    parser.add_argument(
        "--min_session_questions",
        type=int,
        default=34,
        help="Minimum questions required in a (user, session) group",
    )
    args = parser.parse_args()

    df = load_data_for_preprocessing(args)
    output_dir = os.path.join(PROJECT_ROOT, "data")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    preprocess_synthetic_dataset(
        df,
        output_dir,
        args.dataset_name,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
        min_interactions_per_student=args.min_interactions_per_student,
        random_seed=args.random_seed,
        max_n_trajectories=args.max_n_trajectories,
        min_trajectory_length=args.min_trajectory_length,
        max_trajectory_length=args.max_trajectory_length,
    )

# example command:
# python preprocess_synthetic.py -d synthetic-11 --test_ratio 0.2 --valid_ratio 0.1 --min_interactions_per_student 10 --random_seed 42 --max_n_trajectories 1000 --min_trajectory_length 5 --max_trajectory_length 20
# python preprocess_synthetic.py -d synthetic-11 --test_ratio 0.2 --valid_ratio 0.1 --min_interactions_per_student 10 --random_seed 42 --max_n_trajectories 1000 --min_trajectory_length 5 --max_trajectory_length 20
