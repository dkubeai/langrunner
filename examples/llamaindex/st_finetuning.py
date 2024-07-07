"""sentence transformers finetuning with langrunner."""

import sys
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.finetuning import SentenceTransformersFinetuneEngine

try:
    __import__("langrunner")
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__, "../../")))

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

finetune_engine = CloudRunner(
    SentenceTransformersFinetuneEngine,
    train_dataset=train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="ft_model",
    val_dataset=val_dataset,
)

finetune_engine.finetune()
