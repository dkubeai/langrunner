"""remote execution of sentence transformers finetuning."""

import os
import logging

from llama_index.finetuning import SentenceTransformersFinetuneEngine
from langrunner import get_current_context
from langrunner.remotes import RemoteRunnable
from langrunner import RunnerSettings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class SentenceTransformersFinetuneEngineRemote(RemoteRunnable, SentenceTransformersFinetuneEngine):
    """Remote exec compatible implementation of SentenceTransformersFinetuneEngine."""

    remote_attrs = ["model", "loss", "finetune"]
    remote_requirements = ['llama-index==0.10.53', 'llama-index-finetuning==0.1.10']
    remote_default_settings = RunnerSettings(memory="16+", accelerator='T4', accelerator_count=1)
    initialize_baseclass = False
    

    @staticmethod
    def _finetune_remotefunc():
        """The block which executes remotely."""
        from llama_index.finetuning import SentenceTransformersFinetuneEngine
        from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset

        context = get_current_context()

        inputdir = "/mnt/input"

        train_dataset_fp = os.path.join(inputdir, context.train_dataset_file)

        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_fp)
        val_dataset = None
        if context.val_dataset_file is not None:
            val_dataset_fp = os.path.join(inputdir, context.val_dataset_file)

            val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_fp)

        context.langclass_initparams["dataset"] = train_dataset
        context.langclass_initparams["val_dataset"] = val_dataset
        context.langclass_initparams["model_output_path"] = "/mnt/output"
        finetune_engine = SentenceTransformersFinetuneEngine(**context.langclass_initparams)
        finetune_engine.finetune()
           
    def finetune(self):
        """remote exec compatible impl of finetune method."""
        context = get_current_context()

        self.dataset.save_json(os.path.join(context.inputdir, "train_dataset.json"))
        context.train_dataset_file = "train_dataset.json"

        context.val_dataset_file = None
        if self.val_dataset is not None:
            self.val_dataset.save_json(os.path.join(context.inputdir, "val_dataset.json"))
            context.val_dataset_file = "val_dataset.json"

        model_dir = self.model_output_path
        if os.path.islink(model_dir):
            logging.info(f"{model_dir} link exists from previous run, overwriting it.")
            os.remove(model_dir)
        if os.path.exists(model_dir):
            raise FileExistsError(
                "{model_dir} already exists, make sure to provide a unique path in every run."
            )
        os.symlink(context.outputdir, model_dir)

        context.langclass_initparams.pop('dataset', None)
        context.langclass_initparams.pop('val_dataset', None)

        context.remote_tasktype = 'task'

        # yield a remote func
        yield self.__class__._finetune_remotefunc

        # post processing - nothing to process after the execution
        return
