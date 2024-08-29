"""sentence transformers finetuning with langrunner."""

import sys
import os
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from os.path import abspath, dirname, join

from llama_index.finetuning import SentenceTransformersFinetuneEngine

train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

import langrunner
SentenceTransformersFinetuneEngine = langrunner.runnable(SentenceTransformersFinetuneEngine)

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
