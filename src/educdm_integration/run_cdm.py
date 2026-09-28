import argparse
import subprocess
import sys
import wandb
import logging
import pandas as pd


def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, default="synthetic-11")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_students", type=int, default=None)
    parser.add_argument("--min_questions", type=int, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--n_test_questions", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--min_interactions_per_student", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    # add session_gap_minutes and min_session_questions
    parser.add_argument("--session_gap_minutes", type=int, default=10)
    parser.add_argument("--min_session_questions", type=int, default=34)
    args = parser.parse_args()

    if args.n_test_questions is not None:
        # check that min_question==max_questions
        assert args.min_questions == args.max_questions, "When specifying n_test_questions, min_questions must equal max_questions"
        args.test_ratio = args.n_test_questions / args.max_questions * 1.01  # small buffer to ensure int(n*test_ratio)=n_test_questions
        print(f"Adjusted test_ratio to {args.test_ratio} to get {args.n_test_questions} test questions per student")

    # Step 1: Preprocess
    preprocess_cmd = (
        f"python src/educdm_integration/preprocess_synthetic.py "
        f"-d {args.dataset_name} "
        f"--test_ratio {args.test_ratio} "
        f"--valid_ratio {args.valid_ratio} "
        f"--min_interactions_per_student {args.min_interactions_per_student} "
        f"--random_seed {args.random_seed} "
    )
    if args.max_students is not None:
        preprocess_cmd += f"--max_n_trajectories {args.max_students} "
    if args.min_questions is not None:
        preprocess_cmd += f"--min_trajectory_length {args.min_questions} "
    if args.max_questions is not None:
        preprocess_cmd += f"--max_trajectory_length {args.max_questions} "
    if args.session_gap_minutes is not None:
        preprocess_cmd += f"--session_gap_minutes {args.session_gap_minutes} "
    if args.min_session_questions is not None:
        preprocess_cmd += f"--min_session_questions {args.min_session_questions} "
    run_command(preprocess_cmd)

    model2script = {
        "irt": "run_irt.py",
        "dina": "run_dina.py",
        "ncdm": "run_ncdm.py",
        "mirt": "run_mirt.py",
        "kancd": "run_kancd.py",
    }
    script = model2script[args.model_name]

    # Step 2: Run all models
    run_command(
        f"python src/educdm_integration/{script} "
        f"--dataset_name {args.dataset_name} "
        f"--max_n_trajectories {args.max_students} "
        f"--min_trajectory_length {args.min_questions} "
        f"--max_trajectory_length {args.max_questions}"
    )


if __name__ == "__main__":
    main()
