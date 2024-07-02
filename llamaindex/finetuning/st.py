import fire
from llama_index.finetuning import (
    SentenceTransformersFinetuneEngine as LI_SentenceTransformersFinetuneEngine,
)
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset


from typing import Any, Optional


class SentenceTransformersFinetuneEngine(LI_SentenceTransformersFinetuneEngine):
    """Runner friendly - Modified Sentence Transformer Finetuning Engine."""

    def __init__(
        self,
        dataset: EmbeddingQAFinetuneDataset,
        model_id: str = "BAAI/bge-small-en",
        model_output_path: str = "exp_finetune",
        batch_size: int = 10,
        val_dataset: Optional[EmbeddingQAFinetuneDataset] = None,
        loss: Optional[Any] = None,
        epochs: int = 2,
        show_progress_bar: bool = True,
        evaluation_steps: int = 50,
        use_all_docs: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        """Init params."""

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.model_id = model_id
        self.model_output_path = model_output_path
        self.model = model_id

        self.use_all_docs = use_all_docs

        self.batch_size = batch_size
        self.epochs = epochs
        self.show_progress_bar = show_progress_bar
        self.evaluation_steps = evaluation_steps
        self.warmup_steps = int(len(self.loader) * epochs * 0.1)


def remote_function(*args, **kwargs):
    train_dataset_fp = kwargs["dataset"]

    train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_fp)

    val_dataset = None
    if "val_dataset" in kwargs:
        val_dataset_fp = kwargs["val_dataset"]
        val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_fp)

    finetune_engine = SentenceTransformersFinetuneEngine(train_dataset, **kwargs)

    finetune_engine.finetune()


if __name__ == "__main__":
    fire.Fire(remote_function)
