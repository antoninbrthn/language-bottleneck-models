"""
Reproduce preprocessing done by pyKT for synthetic dataset
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

# sys.path.append(PROJECT_ROOT)

from pykt.preprocess.utils import write_txt
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question

from src.synthetic_dataset.constructs import Construct
from src.data.eedi_filtered import EediFilteredLoader
from src.data.xes3g5m_filtered import XES3G5MFilteredLoader
from src.utils.config import DATA_DIR, PROJECT_DATA_DIR, PROJECT_ROOT

# Add the path to the pykt library
sys.path.append("third_party/pykt")


# A modified version of the train_test_split function from pykt
def custom_train_test_split(df, n_test_students=200):
    """
    Split the dataset into train and test sets based on a fixed number of test students.

    Args:
        df: DataFrame with student interactions
        n_test_students: Number of students to include in the test set

    Returns:
        train_df, test_df: DataFrames for training and testing
    """
    # Get unique student IDs
    student_ids = df["uid"].unique()

    # Shuffle the student IDs
    shuffled_ids = list(student_ids)
    random.shuffle(shuffled_ids)

    # Ensure we don't try to select more test students than available
    n_test_students = min(n_test_students, len(shuffled_ids))

    # Select students for test set
    test_student_ids = shuffled_ids[:n_test_students]
    train_student_ids = shuffled_ids[n_test_students:]

    # Split the data
    test_df = df[df["uid"].isin(test_student_ids)]
    train_df = df[df["uid"].isin(train_student_ids)]

    # Report
    print(
        f"Total students: {len(student_ids)}, train students: {len(train_student_ids)}, test students: {len(test_student_ids)}"
    )
    print(
        f"Total rows: {len(df)}, train rows: {len(train_df)}, test rows: {len(test_df)}"
    )

    return train_df, test_df


def create_data_txt_file(
    input_file,
    output_dir,
    dataset_name="synthetic",
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    n_test_students=200,
):
    """
    Preprocess a synthetic dataset to make it compatible with PyKT.

    Args:
            input_file: Path to the synthetic dataset CSV
            output_dir: Directory to save processed files
            dataset_name: Name for the dataset
            max_n_trajectories: Maximum number of student trajectories to include in TRAIN/VALID (not counting test students)
            min_trajectory_length: Minimum number of questions per student
            max_trajectory_length: Maximum number of questions per student
            n_test_students: Number of students to reserve for test set
    """
    print(f"Preprocessing {input_file}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the synthetic dataset
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")

    # Apply trajectory length filtering if specified
    if min_trajectory_length is not None or max_trajectory_length is not None:
        # Count questions per student
        student_counts = df.groupby("student_id").size()
        valid_students = student_counts

        if min_trajectory_length is not None:
            valid_students = valid_students[valid_students >= min_trajectory_length]

        if max_trajectory_length is not None:
            # For students with more questions than max_trajectory_length,
            # we'll truncate their sequences later
            pass

        df = df[df["student_id"].isin(valid_students.index)]
        print(
            f"Filtered to students with trajectory lengths meeting criteria, new shape: {df.shape}"
        )

    # Apply student trajectory filtering if specified
    # Note: We need to ensure we have enough students for test set + train/valid set
    if max_n_trajectories is not None:
        # Get unique student IDs
        unique_students = df["student_id"].unique()

        # Calculate total students needed (train/valid + test)
        total_students_needed = max_n_trajectories + n_test_students

        # Limit to total_students_needed
        if len(unique_students) > total_students_needed:
            selected_students = unique_students[:total_students_needed]
            df = df[df["student_id"].isin(selected_students)]
            print(
                f"Filtered to {total_students_needed} students total ({max_n_trajectories} train/valid + {n_test_students} test), new shape: {df.shape}"
            )

    # Assuming the synthetic dataset has the following columns:
    # student_id, question_id, constructs, operands, correct_answer, student_answer, is_correct
    if all(col in df.columns for col in ["student_id", "question_id", "is_correct"]):
        # Map your column names to PyKT expected column names
        df_processed = pd.DataFrame()
        df_processed["uid"] = df["student_id"]
        df_processed["question_id"] = df["question_id"]

        df_processed["construct_id"] = df["constructs"].apply(
            lambda x: list(Construct).index(Construct(eval(x)[0]))
        )

        # Map correctness to correct (ensure it's 0 or 1)
        df_processed["correct"] = df["is_correct"].astype(int)

        # Add timestamp if available, otherwise create sequential timestamps
        # Here we create sequential timestamps based on student_id
        df_processed["time_done"] = df.groupby("student_id").cumcount()

    else:
        print("Error: Required columns not found in dataset")
        print("Expected: student_id, question_id, is_correct")
        print(f"Found: {df.columns.tolist()}")
        return

    print(f"Processed data shape: {df_processed.shape}")

    data = []
    uids = df_processed.uid.unique()
    problems = df_processed.question_id.unique()
    print(
        f"df_processed: {df_processed.shape}, uids: {len(uids)}, problems: {len(problems)}"
    )

    ui_df = df_processed.groupby("uid", sort=False)

    for ui in ui_df:
        uid, curdf = ui[0], ui[1]
        curdf = curdf.sort_values(by=["time_done"])

        # Apply max_trajectory_length if specified
        if max_trajectory_length is not None and len(curdf) > max_trajectory_length:
            curdf = curdf.iloc[:max_trajectory_length]

        # questions = curdf["question_id"].astype(str).tolist()
        questions = curdf["question_id"].astype(str).tolist()
        concepts = curdf["construct_id"].astype(str).tolist()
        rs = curdf["correct"].astype(int).astype(str).tolist()
        # ts = curdf["time_done"].astype(str).tolist()
        ts = ["NA"] * len(rs)  # not included
        uts = ["NA"] * len(rs)  # not included
        seq_len = len(rs)
        uc = [str(uid), str(seq_len)]
        data.append([uc, questions, concepts, rs, ts, uts])
        if len(data) % 1000 == 0:
            print(len(data))
    # Save the preprocessed file as data.txt
    path = os.path.join(output_dir, dataset_name)
    os.makedirs(path, exist_ok=True)
    write_file = os.path.join(path, "data.txt")
    write_txt(write_file, data)
    print(f"Saved preprocessed file to {write_file}")

    # Return stats for config update
    stats = {
        "actual_n_students": len(uids),
        "actual_n_problems": len(problems),
        "max_n_trajectories": max_n_trajectories,
        "min_trajectory_length": min_trajectory_length,
        "max_trajectory_length": max_trajectory_length,
    }
    return stats


def create_data_txt_file_eedi(
    input_file,
    output_dir,
    dataset_name="eedi",
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    n_test_students=200,
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
    os.makedirs(output_dir, exist_ok=True)

    # Set up loader
    loader = EediFilteredLoader(
        data_path=(
            os.path.dirname(input_file)
            if input_file
            else os.path.join(DATA_DIR, "eedi_private/data/")
        ),
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
    trajectories, identifiers = loader.preprocess(df)
    print(f"Number of trajectories: {len(trajectories)}")

    # Build PyKT-compatible data
    data = []
    uids = []
    problems = set()
    for idx, (traj, ident) in enumerate(zip(trajectories, identifiers)):
        uid, session_id = ident
        questions = [str(entry["question_id"]) for entry in traj]
        concepts = [str(entry["construct_id"]) for entry in traj]
        rs = [str(int(entry["correct"])) for entry in traj]
        ts = ["NA"] * len(rs)
        uts = ["NA"] * len(rs)
        seq_len = len(rs)
        uc = [str(uid), str(seq_len)]
        data.append([uc, questions, concepts, rs, ts, uts])
        uids.append(uid)
        problems.update(questions)
        if (idx + 1) % 1000 == 0:
            print(idx + 1)

    # Save the preprocessed file as data.txt
    path = os.path.join(output_dir, dataset_name)
    os.makedirs(path, exist_ok=True)
    write_file = os.path.join(path, "data.txt")
    write_txt(write_file, data)
    print(f"Saved preprocessed file to {write_file}")

    # Return stats for config update
    stats = {
        "actual_n_students": len(set(uids)),
        "actual_n_problems": len(problems),
        "max_n_trajectories": max_n_trajectories,
        "min_trajectory_length": min_trajectory_length,
        "max_trajectory_length": max_trajectory_length,
    }
    return stats


def create_data_txt_file_xes3g5m(
    input_file,
    output_dir,
    dataset_name="xes3g5m",
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    n_test_students=200,
    session_gap_minutes=10,
    min_session_questions=34,
):
    """
    Preprocess the XES3G5M dataset to make it compatible with PyKT.
    Similar to create_data_txt_file_eedi but using XES3G5MFilteredLoader.
    """
    print(f"Preprocessing XES3G5M data with XES3G5MFilteredLoader...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

    print(df.columns)

    # Ensure all IDs are contiguous
    unique_students = df["user_id"].unique()
    unique_questions = df["question_id"].unique()
    unique_constructs = df["construct_id"].unique()

    student_map = {old: new for new, old in enumerate(sorted(unique_students))}
    question_map = {old: new for new, old in enumerate(sorted(unique_questions))}
    construct_map = {old: new for new, old in enumerate(sorted(unique_constructs))}

    df["user_id"] = df["user_id"].map(student_map)
    df["question_id"] = df["question_id"].map(question_map)
    df["construct_id"] = df["construct_id"].map(construct_map)

    trajectories, identifiers = loader.preprocess(df)
    print(f"Number of trajectories: {len(trajectories)}")

    # Build PyKT-compatible data
    data = []
    uids = []
    problems = set()
    for idx, (traj, ident) in enumerate(zip(trajectories, identifiers)):
        uid, session_id = ident
        questions = [str(entry["question_id"]) for entry in traj]
        concepts = [str(entry["construct_id"]) for entry in traj]
        rs = [str(int(entry["correct"])) for entry in traj]
        ts = ["NA"] * len(rs)
        uts = ["NA"] * len(rs)
        seq_len = len(rs)
        uc = [str(uid), str(seq_len)]
        data.append([uc, questions, concepts, rs, ts, uts])
        uids.append(uid)
        problems.update(questions)
        if (idx + 1) % 1000 == 0:
            print(idx + 1)

    # Save the preprocessed file as data.txt
    path = os.path.join(output_dir, dataset_name)
    os.makedirs(path, exist_ok=True)
    write_file = os.path.join(path, "data.txt")
    write_txt(write_file, data)
    print(f"Saved preprocessed file to {write_file}")

    # Return stats for config update
    stats = {
        "actual_n_students": len(set(uids)),
        "actual_n_problems": len(problems),
        "max_n_trajectories": max_n_trajectories,
        "min_trajectory_length": min_trajectory_length,
        "max_trajectory_length": max_trajectory_length,
    }
    return stats


def update_data_config(configf, dataset_name, stats):
    """Update the data_config.json file with dataset parameters"""
    try:
        with open(configf, "r") as f:
            data_config = json.load(f)

        # Create or update the dataset entry
        if dataset_name not in data_config:
            # Find a template (synthetic dataset) to copy
            template_key = next(
                (k for k in data_config.keys() if k.startswith("synthetic")), None
            )
            if template_key:
                data_config[dataset_name] = data_config[template_key].copy()
                data_config[dataset_name]["dpath"] = data_config[dataset_name][
                    "dpath"
                ].replace(template_key, dataset_name)

                # Update file paths in the config
                for key in data_config[dataset_name]:
                    if (
                        isinstance(data_config[dataset_name][key], str)
                        and template_key in data_config[dataset_name][key]
                    ):
                        data_config[dataset_name][key] = data_config[dataset_name][
                            key
                        ].replace(template_key, dataset_name)
            else:
                data_config[dataset_name] = {}

        # Add dataset parameters to the config
        data_config[dataset_name]["dataset_stats"] = stats

        # Write back to the config file
        with open(configf, "w") as f:
            json.dump(data_config, f, indent=4)

        print(f"Updated {configf} with dataset parameters")

    except Exception as e:
        print(f"Error updating data_config.json: {e}")


# Custom implementation of the split_datasets function
def custom_split_datasets(
    dname, writef, dataset_name, configf, min_seq_len, maxlen, kfold, n_test_students
):
    """
    Modified version of the split_datasets function from pykt to use a fixed number of test students.
    """
    # Import necessary functions from pykt
    from pykt.preprocess.split_datasets import (
        read_data,
        id_mapping,
        save_id2idx,
        generate_sequences,
    )
    from pykt.preprocess.split_datasets import (
        generate_window_sequences,
        calStatistics,
        write_config,
    )

    # Read the data
    total_df, effective_keys = read_data(writef)

    # Calculate max_concepts
    from pykt.preprocess.split_datasets import get_max_concepts

    if "concepts" in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    # Map IDs
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    # Save ID mapping
    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")

    # Get the keys to save
    from pykt.preprocess.split_datasets import ALL_KEYS

    df_save_keys = []
    for key in ALL_KEYS:
        if key in effective_keys:
            df_save_keys.append(key)

    # Train-test split with fixed number of test students
    train_df, test_df = custom_train_test_split(total_df, n_test_students)

    # KFold split for train/valid
    from pykt.preprocess.split_datasets import KFold_split

    splitdf = KFold_split(train_df, kfold)

    # Process and save train/valid data
    splitdf[df_save_keys].to_csv(os.path.join(dname, "train_valid.csv"), index=None)

    # Generate sequences
    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    split_seqs.to_csv(os.path.join(dname, "train_valid_sequences.csv"), index=None)

    # Process test data (adding fold -1)
    test_df["fold"] = [-1] * test_df.shape[0]
    from pykt.preprocess.split_datasets import get_inter_qidx

    test_df["cidxs"] = get_inter_qidx(test_df)

    # Generate test sequences
    test_seqs = generate_sequences(
        test_df, list(effective_keys) + ["cidxs"], min_seq_len, maxlen
    )
    test_window_seqs = generate_window_sequences(
        test_df, list(effective_keys) + ["cidxs"], maxlen
    )

    # Save test data
    test_df = test_df[df_save_keys + ["cidxs"]]
    test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
    test_window_seqs.to_csv(
        os.path.join(dname, "test_window_sequences.csv"), index=None
    )

    # Write config
    write_config(
        dataset_name,
        dkeyid2idx,
        effective_keys,
        configf,
        dname,
        kfold,
        min_seq_len,
        maxlen,
    )

    return dkeyid2idx, effective_keys


# Custom implementation of split_datasets_que function
def custom_split_datasets_que(
    dname, writef, dataset_name, configf, min_seq_len, maxlen, kfold, n_test_students
):
    """
    Modified version of the split_datasets_que function from pykt to use a fixed number of test students.
    """
    # Import necessary functions from pykt
    from pykt.preprocess.split_datasets_que import (
        read_data,
        id_mapping_que,
        save_id2idx,
        generate_sequences,
    )
    from pykt.preprocess.split_datasets_que import (
        generate_window_sequences,
        calStatistics,
        write_config,
    )

    # Read the data
    total_df, effective_keys = read_data(writef)

    # Calculate max_concepts
    from pykt.preprocess.split_datasets_que import get_max_concepts

    if "concepts" in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    # Map IDs
    total_df, dkeyid2idx = id_mapping_que(total_df)
    dkeyid2idx["max_concepts"] = max_concepts

    # Save ID mapping
    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    effective_keys.add("fold")

    # Get the keys to save
    from pykt.preprocess.split_datasets_que import ALL_KEYS

    df_save_keys = []
    for key in ALL_KEYS:
        if key in effective_keys:
            df_save_keys.append(key)

    # Train-test split with fixed number of test students
    train_df, test_df = custom_train_test_split(total_df, n_test_students)

    # KFold split for train/valid
    from pykt.preprocess.split_datasets_que import KFold_split

    splitdf = KFold_split(train_df, kfold)

    # Process and save train/valid data
    splitdf[df_save_keys].to_csv(
        os.path.join(dname, "train_valid_quelevel.csv"), index=None
    )

    # Generate sequences
    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    split_seqs.to_csv(
        os.path.join(dname, "train_valid_sequences_quelevel.csv"), index=None
    )

    # Process test data (adding fold -1)
    test_df["fold"] = [-1] * test_df.shape[0]

    # Generate test sequences
    test_seqs = generate_sequences(test_df, list(effective_keys), min_seq_len, maxlen)
    test_window_seqs = generate_window_sequences(test_df, list(effective_keys), maxlen)

    # Save test data
    test_df = test_df[df_save_keys]
    test_df.to_csv(os.path.join(dname, "test_quelevel.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences_quelevel.csv"), index=None)
    test_window_seqs.to_csv(
        os.path.join(dname, "test_window_sequences_quelevel.csv"), index=None
    )

    # Write config with question level data
    other_config = {
        "train_valid_original_file_quelevel": "train_valid_quelevel.csv",
        "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
        "test_file_quelevel": "test_sequences_quelevel.csv",
        "test_window_file_quelevel": "test_window_sequences_quelevel.csv",
        "test_original_file_quelevel": "test_quelevel.csv",
    }

    write_config(
        dataset_name,
        dkeyid2idx,
        effective_keys,
        configf,
        dname,
        kfold,
        min_seq_len,
        maxlen,
        other_config=other_config,
    )

    return dkeyid2idx, effective_keys


def preprocess_synthetic_dataset(
    input_file,
    output_dir,
    dataset_name="synthetic",
    min_seq_len=3,
    maxlen=200,
    kfold=5,
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    n_test_students=200,
):

    stats = create_data_txt_file(
        input_file,
        output_dir,
        dataset_name,
        max_n_trajectories,
        min_trajectory_length,
        max_trajectory_length,
        n_test_students,
    )

    # Creating the csv files to train one - adapted from `pykt/examples/data_preprocess.py``
    path = os.path.join(output_dir, dataset_name)
    dname, writef = path, os.path.join(path, "data.txt")

    # For concept level model with fixed test size
    # replaced `split_concept``
    custom_split_datasets(
        dname,
        writef,
        dataset_name,
        configf,
        min_seq_len,
        maxlen,
        kfold,
        n_test_students,
    )
    print("=" * 100)

    # For question level model with fixed test size
    # replaced `split_question`
    custom_split_datasets_que(
        dname,
        writef,
        dataset_name,
        configf,
        min_seq_len,
        maxlen,
        kfold,
        n_test_students,
    )

    # Update data_config.json with dataset parameters
    stats["n_test_students"] = n_test_students
    update_data_config(configf, dataset_name, stats)

    print("Done preprocessing {}".format(input_file))


def preprocess_eedi_dataset(
    input_file,
    output_dir,
    dataset_name="eedi",
    min_seq_len=3,
    maxlen=200,
    kfold=5,
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    n_test_students=200,
    min_answer_seconds=5,
    session_gap_minutes=15,
    min_session_questions=40,
):
    stats = create_data_txt_file_eedi(
        input_file,
        output_dir,
        dataset_name,
        max_n_trajectories,
        min_trajectory_length,
        max_trajectory_length,
        n_test_students,
        min_answer_seconds,
        session_gap_minutes,
        min_session_questions,
    )

    # Creating the csv files to train one - adapted from `pykt/examples/data_preprocess.py``
    path = os.path.join(output_dir, dataset_name)
    dname, writef = path, os.path.join(path, "data.txt")

    # For concept level model with fixed test size
    custom_split_datasets(
        dname,
        writef,
        dataset_name,
        configf,
        min_seq_len,
        maxlen,
        kfold,
        n_test_students,
    )
    print("=" * 100)

    # For question level model with fixed test size
    custom_split_datasets_que(
        dname,
        writef,
        dataset_name,
        configf,
        min_seq_len,
        maxlen,
        kfold,
        n_test_students,
    )

    # Update data_config.json with dataset parameters
    stats["n_test_students"] = n_test_students
    update_data_config(configf, dataset_name, stats)

    print("Done preprocessing {}".format(input_file))


def preprocess_xes3g5m_dataset(
    input_file,
    output_dir,
    dataset_name="xes3g5m",
    min_seq_len=3,
    maxlen=200,
    kfold=5,
    max_n_trajectories=None,
    min_trajectory_length=None,
    max_trajectory_length=None,
    n_test_students=200,
    session_gap_minutes=15,
    min_session_questions=34,
):
    stats = create_data_txt_file_xes3g5m(
        input_file,
        output_dir,
        dataset_name,
        max_n_trajectories,
        min_trajectory_length,
        max_trajectory_length,
        n_test_students,
        session_gap_minutes,
        min_session_questions,
    )

    # Creating the csv files to train one - adapted from `pykt/examples/data_preprocess.py``
    path = os.path.join(output_dir, dataset_name)
    dname, writef = path, os.path.join(path, "data.txt")

    # For concept level model with fixed test size
    custom_split_datasets(
        dname,
        writef,
        dataset_name,
        configf,
        min_seq_len,
        maxlen,
        kfold,
        n_test_students,
    )
    print("=" * 100)

    # For question level model with fixed test size
    custom_split_datasets_que(
        dname,
        writef,
        dataset_name,
        configf,
        min_seq_len,
        maxlen,
        kfold,
        n_test_students,
    )

    # Update data_config.json with dataset parameters
    stats["n_test_students"] = n_test_students
    update_data_config(configf, dataset_name, stats)

    print("Done preprocessing {}".format(input_file))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="synthetic-11")
    parser.add_argument("-m", "--min_seq_len", type=int, default=3)
    parser.add_argument("-l", "--maxlen", type=int, default=200)
    parser.add_argument("-k", "--kfold", type=int, default=5)
    parser.add_argument(
        "--max_n_trajectories",
        type=int,
        default=None,
        help="Maximum number of student trajectories to include in train/valid set",
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
    parser.add_argument(
        "--n_test_students",
        type=int,
        default=200,
        help="Number of students to include in test set",
    )
    parser.add_argument(
        "--min_answer_seconds", type=int, default=5, help="Only for EediFiltered"
    )
    parser.add_argument(
        "--session_gap_minutes", type=int, default=3, help="Only for EediFiltered"
    )
    parser.add_argument(
        "--min_session_questions", type=int, default=40, help="Only for EediFiltered"
    )
    args = parser.parse_args()

    output_dir = PROJECT_DATA_DIR
    configf = os.path.join(PROJECT_ROOT, "third_party/pykt/configs/data_config.json")

    # Pass the arguments to the function
    if "synthetic" in args.dataset_name:
        input_file = os.path.join(PROJECT_ROOT, f"data/{args.dataset_name}/data.csv")
        preprocess_synthetic_dataset(
            input_file,
            output_dir,
            args.dataset_name,
            args.min_seq_len,
            args.maxlen,
            args.kfold,
            args.max_n_trajectories,
            args.min_trajectory_length,
            args.max_trajectory_length,
            args.n_test_students,
        )
    elif "eedi" in args.dataset_name:
        input_file = None
        preprocess_eedi_dataset(
            input_file,
            output_dir,
            args.dataset_name,
            args.min_seq_len,
            args.maxlen,
            args.kfold,
            args.max_n_trajectories,
            args.min_trajectory_length,
            args.max_trajectory_length,
            args.n_test_students,
            min_answer_seconds=args.min_answer_seconds,
            session_gap_minutes=args.session_gap_minutes,
            min_session_questions=args.min_session_questions,
        )
    elif "xes3g5m" in args.dataset_name:
        input_file = None
        preprocess_xes3g5m_dataset(
            input_file,
            output_dir,
            args.dataset_name,
            args.min_seq_len,
            args.maxlen,
            args.kfold,
            args.max_n_trajectories,
            args.min_trajectory_length,
            args.max_trajectory_length,
            args.n_test_students,
            session_gap_minutes=args.session_gap_minutes,
            min_session_questions=args.min_session_questions,
        )
    else:
        raise ValueError("Dataset name not supported.")
