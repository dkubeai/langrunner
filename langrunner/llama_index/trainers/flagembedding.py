"""Flag embedding Finetuning."""

import logging
import gc
import os
import functools

from typing import Any
from llama_index.core.llms.llm import LLM
from llama_index.finetuning.types import BaseLLMFinetuneEngine


logger = logging.getLogger(__name__)


class FlagEmbeddingFinetuneEngine(BaseLLMFinetuneEngine):
    """Flag embedding finetuning engine implementation."""

    def __init__(
        self,
        model_name_or_path: str,
        train_data: str,
        output_dir: str,
        query_max_len: int = 64,
        train_group_size: int = 2,
        negatives_cross_device: bool = True,
        dataloader_drop_last: bool = True,
        normlized: bool = True, 
        temperature: float = 0.02,
        passage_max_len: int = 256,
        query_instruction_for_retrieval: str = "",
        use_inbatch_neg: bool = False,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
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
    ):
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

        self.finetuning_complete = False        

    def finetune(self):
        import subprocess

        # convert the settings into command arguments
        command = "torchrun --nproc_per_node=1 -m FlagEmbedding.baai_general_embedding.finetune.run"
        for key, value in self.__dict__.items():
            if key in ['self', 'finetuning_complete'] or value == None:
                continue

            if value is False :
                continue
            elif value is True:
                command = f"{command} --{key}"
            elif isinstance(value, str) and not value:
                continue
            else:
                command = f"{command} --{key}={value}"

        result = subprocess.run(command.split(" "), capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"finetuning failed with error {result.stderr}")

    def get_finetuned_model(self, **model_kwargs: Any) -> Any:
        #MAK TODO
        return
