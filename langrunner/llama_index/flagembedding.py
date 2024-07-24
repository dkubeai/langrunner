import os
import logging

from langrunner.remotes import RemoteRunnable
from langrunner.llama_index.trainers.flagembedding import FlagEmbeddingFinetuneEngine
from langrunner import get_current_context
from langrunner import RunnerSettings

class FlagEmbeddingFinetuneEngineRemote(RemoteRunnable, FlagEmbeddingFinetuneEngine):
    """Remote exec compatible implementation of FlagEmbeddingFinetuneEngineRemote."""
    remote_attrs = ["finetune"]
    remote_requirements = ['llama-index==0.10.53', 'llama-index-finetuning==0.1.10', 'llama-index-llms-huggingface==0.2.4', 'FlagEmbedding', 'tensorboard']
    remote_default_settings = RunnerSettings(memory="64+", accelerator='A10G', accelerator_count=1)
    initialize_baseclass = False

    @staticmethod
    def _finetune_remotefunc():
        context = get_current_context()

        train_dataset = context.langclass_initparams['train_data']
        if os.path.isfile(train_dataset):
            fname = os.path.basename(self.train_dataset)
            target_file = os.path.join("/mnt/input", fname)
            context.langclass_initparams['train_data'] = target_file

        context.langclass_initparams['output_dir'] = "/mnt/output"
        finetune_engine = FlagEmbeddingFinetuneEngine(**context.langclass_initparams)
        finetune_engine.finetune()


    def finetune(self):
        context = get_current_context()

        if os.path.isfile(self.train_data):
            # create sym link in context.inputdir
            fname = os.path.basename(self.train_data)
            target_file = os.path.join(context.inputdir, fname)
            os.symlink(target_file, self.train_data)

        model_dir = self.output_dir
        if os.path.islink(model_dir):
            logging.info(f"{model_dir} link exists from previous run, overwriting it.")
            os.remove(model_dir)
        if os.path.exists(model_dir):
            raise FileExistsError(
                "{model_dir} already exists, make sure to provide a unique path in every run."
            )
        os.symlink(context.outputdir, model_dir)            
        context.remote_tasktype = 'task'
        yield self.__class__._finetune_remotefunc
