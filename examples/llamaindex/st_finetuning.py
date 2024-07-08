"""sentence transformers finetuning with langrunner."""

import sys
import os
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from os.path import abspath, dirname, join

try:
    __import__("langrunner")
except ImportError:
    sys.path.append(dirname(abspath(join(dirname(__file__), "../.."))))

from langrunner import CloudRunner

train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

"""
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
"""

finetune_engine = CloudRunner.from_resource(
    SentenceTransformersFinetuneEngine,
    dataset=train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="ft_model",
    val_dataset=val_dataset,
)

finetune_engine.finetune()
