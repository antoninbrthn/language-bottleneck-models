#!/bin/bash

# Maximum number of concurrent jobs
MAX_PROCS=1

# Array of student counts for train/valid set only (test students are additional)
STUDENT_COUNTS=(10 100 1000)
SEEDS=(1 2 3 4 5 6 7 8 9)

# Array of question counts
QUESTION_COUNTS=(9 14 24 54)  # 54 total questions, incl. 4 test questions
N_TEST_QUESTIONS=4

# Array of datasets
DATASETS=("synthetic-11")

# Array of models to use
MODELS=("irt" "mirt" "dina" "ncdm" "kancd")

# Number of test students (fixed)
WANDB_TAGS="synth-1"  


# Function to run experiment with given parameters
run_experiment() {
    local dataset=$1
    local model=$2
    local students=$3
    local questions=$4
    local seed=$5
    
    echo "Starting: dataset=$dataset model=$model students=$students questions=$questions seed=$seed"

    # set wandb tag
    export WANDB_TAGS="$WANDB_TAGS"
    echo "Using WANDB_TAGS: $WANDB_TAGS"

    python src/educdm_integration/run_cdm.py \
        -d "$dataset" \
        --model_name "$model" \
        --max_students "$students" \
        --min_questions "$questions" \
        --max_questions "$questions" \
        --random_seed "$seed" \
        --n_test_questions $N_TEST_QUESTIONS
    
    echo "Finished: dataset=$dataset model=$model students=$students questions=$questions seed=$seed"

    # Add a small delay between runs to avoid overwhelming the system
    sleep 2
}

# Launch experiments in background, capping at MAX_PROCS
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for students in "${STUDENT_COUNTS[@]}"; do
                for questions in "${QUESTION_COUNTS[@]}"; do
                    #  Get rid of parallel processing - using the same underlying data files so they would get mixed up!
                    # (
                    run_experiment "$dataset" "$model" "$students" "$questions" "$seed"
                    # ) &

                    # if we’ve hit MAX_PROCS running jobs, wait for at least one to finish
                    # while [ "$(jobs -rp | wc -l)" -ge "$MAX_PROCS" ]; do
                        # sleep 1
                    # done
                done
            done
        done
    done
done

# wait for any remaining background jobs
wait

echo "All experiments completed!"
