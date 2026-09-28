#!/usr/bin/env python
# run_dkt.py

import argparse
import os
import subprocess
import sys
import json

def run_command(command):
    """Run a shell command and print its output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
    
    return process.returncode

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DKT on a synthetic dataset with controlled parameters')
    parser.add_argument('-d', '--dataset_name', type=str, default="synthetic-11", help='Name of the synthetic dataset')
    parser.add_argument('--max_students', type=int, default=None, help='Number of students to include in train/valid set (not counting test students)')
    parser.add_argument('--min_questions', type=int, default=None, help='Minimum questions per student')
    parser.add_argument('--max_questions', type=int, default=None, help='Maximum questions per student')
    parser.add_argument('--min_seq_len', type=int, default=3, help='Minimum sequence length for PyKT')
    parser.add_argument('--maxlen', type=int, default=200, help='Maximum sequence length for PyKT')
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument('--model', type=str, default='dkt', help='Which model to run')
    parser.add_argument('--n_test_students', type=int, default=200, help='Number of students to include in test set')
    parser.add_argument("--min_answer_seconds", type=int, default=5, help="Only for EediFiltered")
    parser.add_argument("--session_gap_minutes", type=int, default=3, help="Only for EediFiltered")
    parser.add_argument("--min_session_questions", type=int, default=40, help="Only for EediFiltered")

    args = parser.parse_args()
    
    # Step 1: Preprocess the dataset with our parameters
    preprocess_cmd = (
        f"python src/pykt_integration/preprocess_synthetic.py "
        f"-d {args.dataset_name} "
        f"-m {args.min_seq_len} "
        f"-l {args.maxlen} "
    )
    
    if args.max_students is not None:
        preprocess_cmd += f"--max_n_trajectories {args.max_students} "
    
    if args.min_questions is not None:
        preprocess_cmd += f"--min_trajectory_length {args.min_questions} "
    
    if args.max_questions is not None:
        preprocess_cmd += f"--max_trajectory_length {args.max_questions} "
    
    if args.min_answer_seconds is not None:
        preprocess_cmd += f"--min_answer_seconds {args.min_answer_seconds} "
    if args.session_gap_minutes is not None:
        preprocess_cmd += f"--session_gap_minutes {args.session_gap_minutes} "
    if args.min_session_questions is not None:
        preprocess_cmd += f"--min_session_questions {args.min_session_questions} "
    
    preprocess_cmd += f"--n_test_students {args.n_test_students} "
    
    exit_code = run_command(preprocess_cmd)
    if exit_code != 0:
        print("Preprocessing failed")
        return exit_code
    
    # # Step 2: Ensure the dataset is in the pyKT data_config.json
    # # We'll read the file, check if our dataset is there, and add it if not
    # config_file = "third_party/pykt/configs/data_config.json"
    
    # try:
    #     with open(config_file, 'r') as f:
    #         data_config = json.load(f)
        
    #     if args.dataset_name not in data_config:
    #         # We need to manually add the entry
    #         print(f"Adding {args.dataset_name} to data_config.json...")
            
    #         # Find an existing synthetic dataset and copy its structure
    #         template_key = next((k for k in data_config.keys() if k.startswith('synthetic')), None)
            
    #         if template_key is None:
    #             print("Error: No template synthetic dataset found in data_config.json")
    #             return 1
            
    #         # Copy the template and update paths
    #         new_config = data_config[template_key].copy()
    #         new_config['dpath'] = new_config['dpath'].replace(template_key, args.dataset_name)
            
    #         # Update file paths
    #         for key in new_config:
    #             if isinstance(new_config[key], str) and template_key in new_config[key]:
    #                 new_config[key] = new_config[key].replace(template_key, args.dataset_name)
            
    #         # Add to config
    #         data_config[args.dataset_name] = new_config
            
    #         # Write back to file
    #         with open(config_file, 'w') as f:
    #             json.dump(data_config, f, indent=4)
                
    # except Exception as e:
    #     print(f"Error updating data_config.json: {e}")
    #     return 1
    
    # Step 2: Run the model
    # Navigate to the pyKT examples directory
    os.chdir("third_party/pykt/examples")
    # make sure wandb is online
    os.environ["WANDB_MODE"] = "online"
    
    # Build the training command based on the specified model
    if args.model == 'dkt':
        train_cmd = f"python wandb_dkt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'qdkt':
        train_cmd = f"python wandb_qdkt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'rkt':
        train_cmd = f"python wandb_rkt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'akt':
        train_cmd = f"python wandb_akt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'dkvmn':
        train_cmd = f"python wandb_dkvmn_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'sakt':
        train_cmd = f"python wandb_sakt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'gkt':
        train_cmd = f"python wandb_gkt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'saint':
        train_cmd = f"python wandb_saint_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'lpkt':
        train_cmd = f"python wandb_lpkt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'simplekt':
        train_cmd = f"python wandb_simplekt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'qikt':
        train_cmd = f"python wandb_qikt_train.py --dataset_name={args.dataset_name}"
    elif args.model == 'deep_irt':
        train_cmd = f"python wandb_deep_irt_train.py --dataset_name={args.dataset_name}"
    else:
        print(f"In RUN DKT")
        print(f"Unsupported model: {args.model}")
        return 1
    # example command:
    # python wandb_gkt_train.py --dataset_name=synthetic-11 
    # Add wandb flag if specified
    if args.use_wandb != 1:
        train_cmd += f" --use_wandb={args.use_wandb}"
    print(train_cmd)
    # Run the training command
    exit_code = run_command(train_cmd)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 