import os
import logging

from langrunner.remotes import RemoteRunnable
from langrunner.llama_index.trainers.sft_trainer import SFTFinetuneEngine
from langrunner import get_current_context
from langrunner import RunnerSettings


class SFTFinetuneEngineRemote(RemoteRunnable, SFTFinetuneEngine):
    """Remote exec compatible implementation of SFTFinetuneEngineRemote."""
    remote_attrs = ["finetune"]
    remote_requirements = ['llama-index==0.10.53', 'llama-index-finetuning==0.1.10', 'llama-index-llms-huggingface==0.2.4', 'torch', 'tensorboard', 'scipy', 'peft==0.4.0', 'bitsandbytes==0.40.2', 'transformers==4.37.0', 'trl==0.4.7', 'accelerate', 'mistralai==0.4.2']
    remote_default_settings = RunnerSettings(memory="64+", accelerator='A10G', accelerator_count=1)
    initialize_baseclass = False

    @staticmethod
    def _finetune_remotefunc():
        context = get_current_context()

        train_dataset = context.langclass_initparams['train_dataset']
        '''
        if os.path.isfile(train_dataset):
            fname = os.path.basename(self.train_dataset)
            target_file = os.path.join("/mnt/input", fname)
            context.langclass_initparams['train_dataset'] = target_file
        '''
        context.langclass_initparams['output_dir'] = "/mnt/output"
        finetune_engine = SFTFinetuneEngine(**context.langclass_initparams)
        finetune_engine.finetune()

    def finetune(self):
        context = get_current_context()

        if os.path.isfile(self.train_dataset):
            # create sym link in context.inputdir
            fname = os.path.basename(self.train_dataset)
            target_file = os.path.join(context.inputdir, fname)
            os.symlink(f"{os.getcwd()}/{self.train_dataset}", target_file)
            # this is where the file is going to get mounted inside sky task
            context.langclass_initparams['train_dataset'] = os.path.join("/mnt/input", fname)

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
