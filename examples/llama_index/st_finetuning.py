"""sentence transformers finetuning with langrunner."""

import sys
import os
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from os.path import abspath, dirname, join

from langrunner.llama_index.finetuning import SentenceTransformersFinetuneEngine
#from llama_index.finetuning import SentenceTransformersFinetuneEngine

try:
    __import__("langrunner")
except ImportError:
    sys.path.append(dirname(abspath(join(dirname(__file__), "../.."))))

train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
print(embed_model)
