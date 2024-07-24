"""sentence transformers finetuning with langrunner."""

import sys
import os

from langrunner.llama_index.trainers import SFTFinetuneEngine

import langrunner
SFTFinetuneEngine = langrunner.runnable(SFTFinetuneEngine)
finetune_engine = SFTFinetuneEngine(
        base_model_name = "NousResearch/Llama-2-7b-chat-hf",
        train_dataset = "mlabonne/guanaco-llama2-1k",
        output_dir="./sft_output")

finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
print(embed_model)
