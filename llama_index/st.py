"""remote execution of sentence transformers finetuning."""

import os

from typing import Any, Tuple

from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset

from langrunner.remote import check, prolog, run, epilog
from langrunner.utils import serialize, deserialize


@check.register(SentenceTransformersFinetuneEngine, None)
def _check(cls, fn) -> bool:
    return True


@check.register(SentenceTransformersFinetuneEngine, "finetune")
def _check(cls, fn) -> bool:
    return True


@prolog.register(SentenceTransformersFinetuneEngine, "finetune")
def _prolog(cls, fn, inputdir, outputdir, cls_params=None, fn_params=None):

    dataset = cls_params["dataset"]
    val_dataset = cls_params["val_dataset"]

    extra_params = {}

    dataset.save_json(os.path.join(inputdir, "train_dataset.json"))
    extra_params["train_dataset"] = "train_dataset.json"

    if val_dataset is not None:
        dataset.save_json(os.path.join(inputdir, "val_dataset.json"))
        extra_params["val_dataset"] = "val_dataset.json"

    cls_params["dataset"] = None  # reset the value
    cls_params["val_dataset"] = None

    # save the class params
    params_fp = os.path.join(inputdir, "class_params.pkl")
    serialize(cls_params, params_fp, mode="pickle")
    cls_params["dataset"] = dataset
    cls_params["val_dataset"] = val_dataset

    extraparams_fp = os.path.join(inputdir, "extra_params.json")
    serialize(extra_params, extraparams_fp, mode="json")

    model_dir = cls_params["model_output_path"]
    os.symlink(outputdir, model_dir)


@run.register(SentenceTransformersFinetuneEngine, "finetune")
def _run(cls, fn, run_name):
    """Execute the class and its method."""
    inputdir = "/mnt/input"

    cls_params = deserialize(os.path.join(inputdir, "class_params.pkl"), mode="pickle")
    extra_params = deserialize(os.path.join(inputdir, "extra_params.json"), mode="json")

    train_dataset = EmbeddingQAFinetuneDataset.from_json(extra_params["train_dataset"])
    cls_params["dataset"] = train_dataset

    if "val_dataset" in extra_params:
        val_dataset = EmbeddingQAFinetuneDataset.from_json(extra_params["val_dataset"])
        cls_params["val_dataset"] = val_dataset

    cls_params["model_output_path"] = "/mnt/output"

    finetune_engine = SentenceTransformersFinetuneEngine(**cls_params)
    finetune_engine.finetune()


@epilog.register(SentenceTransformersFinetuneEngine, "finetune")
def _epilog(cls, fn, outputdir) -> Tuple[Any, ...]:
    return None
