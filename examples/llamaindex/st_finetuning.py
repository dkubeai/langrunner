from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from langrunner.llama_index.finetuning import SentenceTransformersFinetuneEngine

train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)

from langrunner import CloudRunner

cr = CloudRunner()
cr.setup()
cr.run(finetune_engine)
