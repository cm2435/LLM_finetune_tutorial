# Fine-Tune Llama2-7b on SE paired dataset
import os
import torch
import tyro
from accelerate import Accelerator
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import SFTTrainer
from trl.import_utils import is_xpu_available

from utils_sft import create_datasets
from dataclass.sft_args import ScriptArguments



if __name__ == "__main__":
    script_args = tyro.cli(ScriptArguments)

    if script_args.training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")



    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    peft_config = script_args.peft_config

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    training_args = script_args.training_args

    train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(script_args.training_args.output_dir)

    output_dir = os.path.join(script_args.training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del base_model
    if is_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)