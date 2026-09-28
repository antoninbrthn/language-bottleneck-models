# Language Bottleneck Models for Qualitative Knowledge State Modeling

Official code for the paper: **[Language Bottleneck Models for Qualitative Knowledge State Modeling](https://arxiv.org/abs/2506.16982)**.


## Installation

### 1. Clone the repository

```bash
git clone <link-to-repository>
cd language-bottleneck-models
```

### 2. Install dependencies

```bash
pip install -e .
```

Or directly from requirements:

```bash
pip install -r requirements.txt
```

**For encoder training** (GRPO with Unsloth for fast LoRA training):

```bash
pip install unsloth
```

**For baseline integrations** (`src/pykt_integration/`, `src/educdm_integration/`):

```bash
# Required for EduCDM baselines
pip install EduCDM

# Required for pyKT baselines (scripts expect this exact path)
git clone https://github.com/pykt-team/pykt-toolkit.git third_party/pykt
pip install -e third_party/pykt
```

### 3. Configure environment variables

Create a `.env` file at the repository root:

```bash
# Required: Azure OpenAI API credentials (for API-based models)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key

# Optional: Override default paths
PROJECT_ROOT=/path/to/this/repo       # defaults to current working directory
DATA_DIR=/path/to/data                 # defaults to PROJECT_ROOT/data/
XES3G5M_ROOT=/path/to/XES3G5M         # defaults to DATA_DIR/XES3G5M/
```

Alternatively, export these as environment variables in your shell.

### 4. Set up Weights & Biases (optional but recommended)

```bash
wandb login
```

All experiments log to the `lbm` wandb project by default.

## Datasets

### Synthetic Dataset

A controlled arithmetic dataset with 4 constructs (addition, subtraction, multiplication, division) and 6 misconception types. Students have varying proficiency levels and misconceptions, making it a testbed for interpretable KT.

**Generate a new synthetic dataset:**

```bash
python -m src.synthetic_dataset.dataset_generator \
    --dataset_name synthetic-new \
    --n_students 1000 \
    --n_questions 5000 \
    --trajectory_length 75 \
    --output_bool True
```

Generated datasets are saved to `data/`.

The Synthetic dataset used in the paper is provided under `data/synthetic-11`.


### Eedi Dataset

While the exact version of the EEDI dataset used in this work is not available publicly, a very similar version including question texts is available via the ["Eedi - Mining Misconceptions in Mathematics" Kaggle Competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data).

`EEDIFiltered` is derived at load time by `EediFilteredLoader` (`src/data/eedi_filtered.py`): it first loads the full Eedi `answer.csv` via `EediLoader`, then applies in-class filtering (Checkin only, minimum answer time, session gap split, minimum session length).

### XES3G5M Dataset

This dataset is available in the [official XES3G5M repository](https://github.com/ai4ed/XES3G5M). Additionally, we use English translations of the questions and knowledge constructs from the [KCQRL repository](https://github.com/oezyurty/KCQRL).

Place the XES3G5M data under `XES3G5M_ROOT/` (see `src/utils/config.py` for the expected directory structure).

`XES3G5MFiltered` is derived at load time by `XES3G5MFilteredLoader` (`src/data/xes3g5m_filtered.py`): it first loads full XES3G5M sequences through `XES3G5MLoader`, then applies in-class session filtering (timestamp-gap segmentation + minimum session length).

## Running Experiments

All scripts use [Hydra](https://hydra.cc/) for configuration. Specify a config file name (without `.yaml`) and optionally override parameters from the command line.

### Zero-Shot Evaluation

Evaluate a model without any training:

```bash
# Evaluate on synthetic data with a direct LLM (no bottleneck)
python scripts/eval_zs.py config-synthetic

# Evaluate a Language Bottleneck Model on synthetic data
python scripts/eval_zs.py config-lbm-3

# Direct baseline (no bottleneck) on synthetic data
python scripts/eval_zs.py config-lbm-3-direct

# Steered misconception bottleneck
python scripts/eval_zs.py config-steer

# Evaluate on XES3G5M Filtered
python scripts/eval_zs.py config-xes-filt-lbm
python scripts/eval_zs.py config-xes-filt-direct
```

**Override config parameters from the command line:**

```bash
python scripts/eval_zs.py config-lbm-3 model.encoder.name=gpt-4o model.decoder.name=gpt-4o
```

### Ground-Truth Bottleneck Evaluation

Evaluate with a "perfect" bottleneck (oracle) to measure the upper bound on the Synthetic dataset:

```bash
python scripts/eval_ground_truth.py config-ground-truth
```

### Train the Encoder (GRPO)

Train the encoder LLM to produce better bottlenecks using GRPO with the decoder's accuracy as reward:

```bash
# Train on synthetic data (default: Gemma-3-12B encoder, Gemma-3-27B decoder)
python scripts/train_lbm_unsloth.py --config_name config-train-lbm
```

Trained models are saved to `results/trained_models/lbm_{wandb_run_id}/`.

### Train the Decoder (SFT)

Fine-tune the decoder on collected (bottleneck, question, label) triples:

```bash
python scripts/train_decoder.py --config_name config-train-decoder-xes
```

### Evaluate a Trained Model

Evaluate a trained encoder (or encoder+decoder) checkpoint:

```bash
python scripts/eval_us_model.py config-lbm-3
```

This script loads trained LoRA adapters from saved checkpoints and runs evaluation.

## Configuration System

Configs are YAML files under `configs/` loaded via Hydra. Key sections:

```yaml
dataset:
  name: synthetic-11            # Dataset identifier
  max_trajectory_length: 54     # Number of interactions in context
  n_y_prompts: 4                # Number of new questions to predict
  max_n_trajectories: 1000      # Number of student trajectories

model:
  type: lbm                     # Model type: lbm, hf, api
  max_bottleneck_tokens: 128    # Bottleneck length limit
  encoder:
    name: google/gemma-3-12b-it # Encoder model name 
    type: hf                    # hf (HuggingFace) or api (OpenAI); infered if omitted
    use_cot: False              # To use chain-of-thought before producing the bottleneck
  decoder:
    name: gpt-4o                # Decoder model name
    type: api
    use_cot: False              # To use chain-of-thought before predicting the answer

evaluation:
  batch_size: 20
  export_results: True          
```

## Baseline Integrations

### PyKT (Knowledge Tracing baselines)

We provide integration scripts for [PyKT](https://github.com/pykt-team/pykt-toolkit) to run traditional KT models (DKT, AKT, etc.) on the same datasets.

```bash
# Generate and preprocess synthetic data for PyKT
python src/pykt_integration/preprocess_synthetic.py -d synthetic-11

# Run the pyKT experiment bash script
bash src/pykt_integration/run_experiments.sh
```

### EduCDM (Cognitive Diagnosis baselines)

We provide integration scripts for [EduCDM](https://github.com/bigdata-ustc/EduCDM) to run cognitive diagnosis models (IRT, MIRT, NCDM, DINA, KaNCD).

```bash
# Preprocess data
python src/educdm_integration/preprocess_synthetic.py

# Run models (e.g., IRT)
python src/educdm_integration/run_irt.py

# Run the EduCDM experiment bash script
bash src/educdm_integration/run_experiments.sh
```


## Project Structure

```
├── configs/                    # Hydra configuration files
│   ├── config-synthetic.yaml       # Zero-shot eval on synthetic data
│   ├── config-lbm-3.yaml           # LBM evaluation config
│   ├── config-lbm-3-direct.yaml    # Direct (no-bottleneck) baseline
│   ├── config-steer.yaml           # Steered misconception bottleneck
│   ├── config-eedi-filt-lbm.yaml   # LBM eval on Eedi Filtered
│   ├── config-eedi-filt-direct.yaml
│   ├── config-xes-filt-lbm.yaml    # LBM eval on XES3G5M Filtered
│   ├── config-xes-filt-direct.yaml
│   ├── config-ground-truth.yaml    # Ground-truth bottleneck eval
│   ├── config-train-lbm.yaml       # GRPO encoder training
│   └── config-train-decoder-xes.yaml  # SFT decoder training
│
├── prompts/                    # Prompt templates (YAML)
│   ├── base_x.yaml                 # Encoder prompt template
│   ├── base_y.yaml                 # Decoder prompt template
│   ├── steer_misconception_x.yaml  # Steered encoder prompt
│   ├── iterative_encoder.yaml      # Iterative refinement prompt
│   └── bottleneck_annotation.yaml  # Bottleneck annotation prompt
│
├── scripts/                    # Top-level run scripts
│   ├── eval_zs.py                  # Zero-shot evaluation
│   ├── eval_us_model.py            # Evaluate trained (Unsloth) model
│   ├── eval_ground_truth.py        # Ground-truth bottleneck eval
│   ├── train_lbm_unsloth.py        # Train encoder with GRPO (Unsloth)
│   └── train_decoder.py            # Train decoder with SFT
│
├── src/
│   ├── data/                   # Dataset loading and preprocessing
│   │   ├── dataset_loader.py       # Base DatasetLoader class
│   │   ├── synthetic_loader.py     # Synthetic arithmetic dataset
│   │   ├── eedi.py                 # Eedi (full) dataset
│   │   ├── eedi_filtered.py        # Eedi Filtered subset
│   │   ├── xes3g5m.py              # XES3G5M (full) dataset
│   │   ├── xes3g5m_filtered.py     # XES3G5M Filtered subset
│   │   ├── utils.py                # Data loading utilities
│   │   └── utils_iterative.py      # Iterative data utilities
│   │
│   ├── models/                 # Model implementations
│   │   ├── api_models.py           # Azure OpenAI API wrapper
│   │   ├── hf_models.py            # HuggingFace model wrapper
│   │   ├── model_loader.py         # Unified model factory
│   │   ├── general_llm.py          # Base LLM interface
│   │   ├── finetune_lbm.py         # GRPO training utilities
│   │   ├── finetune_lora.py        # LoRA/SFT training utilities
│   │   └── lb_models/              # Language Bottleneck Models
│   │       ├── lb_base.py              # LBModel (standard)
│   │       ├── lb_steer.py             # LBModelSteer (misconception-guided)
│   │       ├── lb_gt.py                # LBModelGroundTruth
│   │       └── utils.py                # Shared LB utilities
│   │
│   ├── prompts/                # Prompt generation
│   │   ├── prompt_generator.py     # PromptGenerator class
│   │   └── prompt_utils.py         # Template loading helpers
│   │
│   ├── evaluation/             # Evaluation logic
│   │   └── evaluate_model.py       # ModelEvaluator class
│   │
│   ├── synthetic_dataset/      # Synthetic dataset generation
│   │   ├── dataset_generator.py    # Main generator script
│   │   ├── constructs.py           # Arithmetic constructs
│   │   ├── misconceptions.py       # Misconception definitions
│   │   ├── questions.py            # Question generation
│   │   └── students.py             # Student profiles
│   │
│   ├── utils/                  # Shared utilities
│   │   ├── config.py               # Path configuration (env vars)
│   │   ├── llm_eval_bn.py          # LLM-based bottleneck evaluation
│   │   ├── llm_pricing.py          # API pricing tracker
│   │   ├── results.py              # Result logging and export
│   │   ├── misc.py                 # Miscellaneous helpers
│   │   └── lb_models.py            # Model registry helpers
│   │
│   ├── pykt_integration/       # PyKT baseline integration
│   └── educdm_integration/     # EduCDM baseline integration
│
├── tests/                      # Unit tests
├── setup.py
├── requirements.txt
└── .gitignore
```


# Citation
```
@article{berthon2025language,
  title={Language Bottleneck Models for Qualitative Knowledge State Modeling},
  author={Berthon, Antonin and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2506.16982},
  year={2025}
}
```