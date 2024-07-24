"""sentence transformers finetuning with langrunner."""

import sys
import os

from langrunner.llama_index.trainers.flagembedding import FlagEmbeddingFinetuneEngine

import langrunner
FlagEmbeddingFinetuneEngine = langrunner.runnable(FlagEmbeddingFinetuneEngine)
finetune_engine = FlagEmbeddingFinetuneEngine(
        model_name_or_path="BAAI/bge-large-zh-v1.5",
        train_data="data.jsonl",
        output_dir="ft_emb_model")

finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
print(embed_model)
