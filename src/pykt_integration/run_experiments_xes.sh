#!/bin/bash

# Maximum number of concurrent jobs
MAX_PROCS=1

# Array of student counts for train/valid set only (test students are additional)
STUDENT_COUNTS=(10000) 

# Array of question counts
QUESTION_COUNTS=(34)

# Array of datasets
DATASETS=("xes3g5m")

# Array of models to use
MODELS=("dkt" "dkvmn" "sakt" "akt" "qikt" "gkt" "saint" "simplekt" "deep_irt")

# Number of test students (fixed)
N_TEST_STUDENTS=200

WANDB_TAGS="xes-1"

# Function to run experiment with given parameters
run_experiment() {
    local dataset=$1
    local model=$2
    local students=$3
    local questions=$4
    
    echo "Starting: dataset=$dataset model=$model students=$students questions=$questions"

    # set wandb tag
    export WANDB_TAGS="$WANDB_TAGS"
    echo "Using WANDB_TAGS: $WANDB_TAGS"

    python src/pykt_integration/run_dkt.py \
        -d "$dataset" \
        --max_students "$students" \
        --min_questions "$questions" \
        --max_questions "$questions" \
        --model "$model" \
        --n_test_students "$N_TEST_STUDENTS" \
        --session_gap_minutes 10 \
        --min_session_questions 34
    
    echo "Finished: dataset=$dataset model=$model students=$students questions=$questions"

    # Add a small delay between runs to avoid overwhelming the system
    sleep 2
}

# Launch experiments in background, capping at MAX_PROCS
for dataset in "${DATASETS[@]}"; do
    for students in "${STUDENT_COUNTS[@]}"; do
        for model in "${MODELS[@]}"; do
            for questions in "${QUESTION_COUNTS[@]}"; do
                #  Get rid of parallel processing - using the same underlying data files so they would get mixed up!
                # (
                run_experiment "$dataset" "$model" "$students" "$questions"
                # ) &

                # if we’ve hit MAX_PROCS running jobs, wait for at least one to finish
                # while [ "$(jobs -rp | wc -l)" -ge "$MAX_PROCS" ]; do
                    # sleep 1
                # done
            done
        done
    done
done

# wait for any remaining background jobs
wait

echo "All experiments completed!"
