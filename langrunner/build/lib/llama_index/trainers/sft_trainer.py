"""SFT Finetuning."""

import logging
import gc
import os
import functools

from typing import Any
from llama_index.core.llms.llm import LLM
from llama_index.finetuning.types import BaseLLMFinetuneEngine


logger = logging.getLogger(__name__)


class SFTFinetuneEngine(BaseLLMFinetuneEngine):
    """SFT trainer engine implementation."""

    def __init__(
        self,
        base_model_name: str,
        train_dataset: str,
        output_dir: str,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        optim: str = "paged_adamw_32bit",
        save_steps: int = 25,
        logging_steps: int = 25,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.001,
        fp16: bool = False,
        bf16: bool = False,
        max_grad_norm: float = 0.3,
        max_steps: int = -1,
        warmup_ratio: float = 0.03,
        group_by_length: bool = True,
        lr_scheduler_type: str = "constant",
        report_to: str = "tensorboard",
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_r: int = 8,
        lora_bias: str = "none",
        lora_tasktype: str = "CAUSAL_LM",
    ):
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

        self.peft_parameters = self.peft_params()
        self.train_parameters = self.training_params()

        self.finetuned_modelname = "finetuned_model"
        self.merged_modelname = "merged_model"
        self.merged_modelpath = os.path.join(self.output_dir, self.merged_modelname)

        self.finetuning_complete = False

    def quantization_config(self):
        """Defines quantization configuration"""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        # configure the model for efficient training
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        return quant_config

    def peft_params(self):
        """Fill lora config"""
        from peft import LoraConfig
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias=self.lora_bias,
            task_type=self.lora_tasktype,
        )

    def training_params(self):
        """Fill training params"""
        from transformers import TrainingArguments

        training_args = TrainingArguments(output_dir=self.output_dir)
        for key in self.__dict__.keys():
            if hasattr(training_args, key):
                try:
                    setattr(training_args, key, getattr(self, key))
                except Exception:
                    continue

        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to=self.report_to,
        )

        return training_args

    def load_dataset(self):
        # handle if dataset is pointint to a file
        from datasets import load_dataset

        return load_dataset(self.train_dataset, split="train")

    #@functools.lru_cache(maxsize=1)
    def load_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    #@functools.lru_cache(maxsize=1)
    def load_basemodel(self):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        self.quant_config = self.quantization_config()

        # load the base model with the quantization configuration
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.quant_config,
            device_map={"": 0},
        )

        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
        return base_model

    def sft_trainer(self):
        from trl import SFTTrainer

        print(f"<< train params -- {self.train_parameters} >>", flush=True)
        print(f"<< train params -- {self.train_parameters.report_to} >>", flush=True)
        # Set up the SFTTrainer with the model, training data, and parameters to learn from the new dataset
        return SFTTrainer(
            model=self.base_model,
            train_dataset=self.training_data,
            peft_config=self.peft_parameters,
            dataset_text_field="text",  # Dependent on your dataset
            tokenizer=self.tokenizer,
            args=self.train_parameters,
        )

    def finetune(self) -> None:
        import torch
        from llama_index.llms.huggingface import HuggingFaceLLM

        self.base_model = self.load_basemodel()
        self.training_data = self.load_dataset()
        self.tokenizer = self.load_tokenizer()
        self.trainer = self.sft_trainer()

        gc.collect()
        torch.cuda.empty_cache()

        self.trainer.train()

        # Save the fine-tuned model's weights and tokenizer files on the cluster
        self.trainer.model.save_pretrained(self.finetuned_modelname)
        self.trainer.tokenizer.save_pretrained(self.finetuned_modelname)

        self.merged_model = self.merge_finetuned_model()

        self.merged_model.save_pretrained(self.merged_modelname)
        self.trainer.tokenizer.save_pretrained(self.merged_modelname)

        gc.collect()
        torch.cuda.empty_cache()

        self.finetuning_complete = True

        logger.info(
            f"fine tuning complete, saved model @ {self.output_dir/self.finetuned_modelname}"
        )

    def get_current_job(self) -> Any:
        """Get current job."""
        pass

    @functools.lru_cache(maxsize=1)
    def merge_finetuned_model(self):
        import torch
        from peft import AutoPeftModelForCausalLM

        from pathlib import Path

        if not Path(f"~/{self.finetuned_modelname}").expanduser().exists():
            raise FileNotFoundError(
                "No fine tuned model found on the cluster. "
                "Call the `finetune` method to run the fine tuning."
            )

        finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
            self.finetuned_modelname,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
        )

        finetuned_model = finetuned_model.merge_and_unload()
        return finetuned_model

    def get_finetuned_model(self, **model_kwargs: Any) -> LLM:
        if self.finetuning_complete is False:
            raise ValueError(f"Finetuning has not completed yet, cannot get model")
        llm = HuggingFaceLLM(
            model_name=self.merged_modelpath,
            tokenizer_name=self.merged_modelpath,
            # https://github.com/run-llama/llama_index/blob/a70850500db20643f63d844691f49f9435dd1ad2/llama-index-integrations/llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py#L213
            **model_kwargs,
        )
        return llm
