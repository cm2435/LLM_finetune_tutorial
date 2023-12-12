# DPO Fine-tuning Project

This project is focused on fine-tuning Dynamic Policy Optimization (DPO) models for enhanced performance. Below is the structure of the project with details on the key directories and instructions for running the fine-tuning scripts.

## Project Structure

- `.github/`: Contains GitHub workflow and action configurations.
- `experimentation/`: 
  - `dataclass/`: Contains data classes for structured data management.
  - `launch_scripts/`: Scripts for launching training jobs, e.g., on a cluster or cloud service.
- `models/`: Storage for model files and checkpoints.
- `src/`: Source code for the DPO algorithms and utilities.
- `tests/`: Test scripts for validating the models and functions.
- `wandb/`: Weights & Biases tracking files for experiment tracking.
- `accelerate_config.yaml`: Configuration file for Hugging Face Accelerate.

## Experimentation

In the `experimentation` folder, you will find scripts and configurations critical for running and managing experiments:

- `dataclass/`: This directory contains Python data classes that define the structure of data used in experiments, ensuring type safety and easy-to-manage defaults.
- `launch_scripts/`: Here, you can find shell scripts that are used to initiate training runs. These may include batch scripts for SLURM or other job managers.

### Running an Experiment

To run a fine-tuning experiment, navigate to the `launch_scripts/` directory and execute the appropriate shell script. For example:

```bash
sh run_merge_adaptor.sh
