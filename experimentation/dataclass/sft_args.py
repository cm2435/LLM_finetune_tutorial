# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from peft import LoraConfig
from tqdm import tqdm
from transformers import TrainingArguments



@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "the model name"})

    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./results",
            max_steps=500,
            logging_steps=10,
            save_steps=10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            group_by_length=False,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.05,
            optim="paged_adamw_32bit",
            bf16=True,
            remove_unused_columns=False,
            run_name="sft_llama2",
            report_to="wandb",
        )
    )

    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    )
# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
#if script_args.training_args.gradient_checkpointing:
#    raise ValueError("gradient_checkpointing not supported")