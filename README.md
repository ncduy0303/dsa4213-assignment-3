# Fine-tuning RoBERTa for Emotion Detection

This project explores and compares two fine-tuning strategies for adapting a `roberta-base` model to the task of text-based emotion detection:

1. **Full Fine-Tuning:** Updating all 125 million parameters of the model.

2. **Prompt Tuning:** A Parameter-Efficient Fine-Tuning (PEFT) method where the base model is frozen and only a small set of "virtual tokens" are trained.

The experiments are conducted on the `dair-ai/emotion` dataset, and hyperparameter optimization is managed using Weights & Biases (W&B) Sweeps. The final analysis and report are created using Typst.

## Quick Links

- Weights & Biases Project: <https://wandb.ai/ncduy0303/dsa4213-assignment-3>
- Model Weights (Google Drive): <https://drive.google.com/drive/folders/1fu7JaSRZgzYIWQrdMngBoHxRez4NEHxz?usp=sharing>

## Folder Structure

The project is organized as follows:

```bash
.
├── .venv/                     # Virtual environment directory (ignored by Git, created with uv)
├── logs/                      # Slurm output and error logs
├── outputs/                   # Saved model checkpoints (ignored by Git, stored on Google Drive)
│   ├── full_finetuning/       # Checkpoints for full fine-tuning experiments
│   ├── lora/                  # Checkpoints for LoRA fine-tuning experiments
│   ├── prompt_tuning_hard/    # Checkpoints for prompt tuning with text initialization experiments
│   └── prompt_tuning_soft/    # Checkpoints for prompt tuning with random initialization experiments
├── sweep_configs/             # W&B sweep configuration files for different experiments
│   ├── sweep_full_finetuning.yaml
│   ├── sweep_lora.yaml
│   ├── sweep_prompt_tuning_hard.yaml
│   └── sweep_prompt_tuning_soft.yaml
├── typst/                     # Report source files and assets
│   ├── png/                   # Images used in the report
│   ├── Assignment3-report.pdf # Final report
│   ├── main.bib               # Bibliography file
│   └── main.typ               # Main Typst document
├── .gitignore
├── .python-version            # Python version for uv
├── Assignment3.pdf            # Assignment description
├── pyproject.toml             # Project dependencies for uv
├── README.md                  # This file
├── train.py                   # The main training script
├── train.sh                   # Slurm script to run W&B agents on the NUS SoC compute cluster
└── uv.lock                    # Dependency lock file
```

**Notes**:

- `lora` related experiments refer to the LoRA (Low-Rank Adaptation) variant of full fine-tuning. Did not include the results in the report due to report length constraints, but the code and checkpoints are available.
- `hard` and `soft` in the prompt tuning directories refer to the initialization method of the virtual tokens: hard (text initialization) and soft (random initialization). Technically, both are still considered "soft" prompts since they are being optimized during training.

## Setup and Installation

This project uses `uv` for package and environment management. Refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/) for installation instructions. When we use `uv run`, it automatically creates and activates a virtual environment in the `.venv/` directory.

### Clone the repository

```bash
git clone git@github.com:ncduy0303/dsa4213-assignment-3.git
cd dsa4213-assignment-3
```

### Reproduce Experiments

The experiments are orchestrated using W&B Sweeps and run on the NUS School of Computing (SoC) compute cluster via the Slurm scheduler. Refer to [W&B documentation](https://docs.wandb.ai/guides/sweeps) for details on W&B Sweeps and to [this guide](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/) for instructions on accessing and using the NUS SoC GPU cluster.

Our experiments were conducted on a NVIDIA H100 GPU with 96GB VRAM (the `xgpi` nodes).

#### Step 1: Set up W&B API Key

You need to provide your W&B API key to the Slurm script.

Get your key from <https://wandb.ai/authorize>.

Open `train.sh` and replace `"WANDB_API_KEY_HERE"` with your actual key.

#### Step 2: Initialize a W&B Sweep

Choose an experiment to run from the `sweep_configs/` directory. For example, to start the full fine-tuning sweep, run the following command locally:

```bash
uv run wandb sweep sweep_configs/sweep_full_finetuning.yaml
```

This will create the sweep on the W&B servers and give you a **SWEEP ID** in the terminal. It will look something like `ncduy0303/dsa4213-assignment-3/8ut679c4`.

#### Step 3: Run the W&B Agent on the Cluster

Edit `train.sh`:

- Replace `JOB_NAME_HERE` with a descriptive name for your job (e.g., full-finetuning). It will be used for the log file names in `logs/`.
- Replace `<SWEEP_ID_HERE>` with the actual sweep ID you got from the previous step.

Submit the job to Slurm:

```bash
sbatch train.sh
```

The script will now launch a W&B agent on a GPU node. The agent will connect to the W&B server, get a set of hyperparameters, and run train.py with those parameters. It will continue to do this until all hyperparameter combinations in the sweep have been tried.

You can monitor the progress, compare results, and analyze the runs in real-time from your W&B project dashboard.

### Alternative: Run A Single Experiment Without W&B Logging

To run the experiments locally, you can use the following command:

```bash
uv run train.py \
  --model_name_or_path roberta-base \
  --dataset_name dair-ai/emotion \
  --metric_name accuracy \
  --report-to none \
  <other_arguments>
```

To know the available arguments, run:

```bash
uv run train.py --help
```

Basically, the sweep configs `.yaml` files contain different combinations of these arguments for different experiments.
