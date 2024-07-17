"""Runners to run lang functions on remote clusters."""

from abc import ABC, abstractmethod

import os
import json
import logging
import uuid
import shutil

from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
import yaml

from .settings import get_current_settings
from .context import get_current_context
from .utils import run_shellcommand

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# this function must be cached so that we dont execute if there is no change.
def setup_langrunner():
    """setup/select a runner based on the compute resources requested."""
    settings = get_current_settings
    sk = SkyRunner()
    sk.setup(settings)
    return sk


class LangRunner(ABC):
    """Base class for different runners."""

    run_settings = get_current_settings
    runid = str(uuid.uuid4())
    rundir = os.path.join(os.path.expanduser("~"), f"./langrunner/{runid}")
    runname = get_random_name(separator="-", style="lowercase")

    @abstractmethod
    def setup(self, settings):
        """setup a runner."""
        pass

    @abstractmethod
    def remotecall(self, func, *args, **kwargs):
        """to execute a func remotely."""
        pass


class SkyRunner(LangRunner):
    """Class to run lang resources on public cloud infra"""

    providers = ["aws", "gcp", "azure", "kubernetes"]

    def setup(self, settings):
        self.run_settings = settings
        use_runners = settings.use_runners

        if use_runners == "auto":
            use_runners = self.providers

        with open(settings.credentials_file, 'r') as fp:
            creds = yaml.safe_load(fp)

        for runner in use_runners:
            if creds[runner]["enabled"] is True:
                params = dict(filter(lambda item: item[0] != 'enabled', creds[runner].items()))
                fn = getattr(self, f"_configure_{runner}")
                fn(**params)

        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(os.path.join(self.rundir, "input"))
        os.makedirs(os.path.join(self.rundir, "output"))

    def remotecall(self):
        import sky

        context = get_current_context()

        file_mounts = {"/mnt/input": context.inputdir}
        task_name = get_random_name(separator="-", style="lowercase")
        # task_name = f"{self.run_name} - {task_name}"
        workdir = os.path.expanduser(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        #workdir = os.path.expanduser("./")


        context_fp = os.path.join(context.inputdir, "remote_context.json")
        with open(context_fp, "w") as fp:
            json.dump(context.json(), fp)

        script = '''
import sys
import os
from os.path import abspath, dirname, join
from llama_index.finetuning import SentenceTransformersFinetuneEngine

try:
    import langrunner
except ImportError:
    sys.path.append(\'./\')

from langrunner import remote
remote.remote_main()
'''

        rc = f'python -c "{script}"'

        remote_requirements = context.REMOTE_REQUIREMENTS_FILE
        current_dir = os.path.dirname(os.path.abspath(__file__))
        setup_command = None
        if os.path.exists(os.path.join(current_dir, remote_requirements)):
            setup_command = f"pip install -r langrunner/{remote_requirements}"

        sky_task = sky.Task(
            name=task_name,
            setup = setup_command,
            run=rc,
            workdir=workdir,
        )

        sky_task.set_file_mounts(file_mounts)

        accelerators = None

        if self.run_settings.use_accelerator == 'auto':
            # [MAK - TODO] - From sky get if accelerators are enabled.
            # if yes then use it. For now setting it to True
            acclerators = {'T4':1}
        elif self.run_settings.use_accelerator is True:
            accl = self.run_settings.accelerator
            accl_count = self.run_settings.accelerator_count
            accelerators = {accl : accl_count}

        sky_task.set_resources(
            sky.Resources(
                # cloud=[
                # getattr(sky.clouds, provider)() for provider in self.cloud_providers
                # ],
                accelerators=accelerators,
                use_spot=self.run_settings.use_spot,
            )
        )

        # sky_clustername = f"lgr-{self.run_name}"
        sky_clustername = f"lgr-{task_name}"
        sky_clustername= 'lgr-fuchsia-mole'

        clusters = sky.status()
        if sky_clustername not in [cluster["name"] for cluster in clusters]:
            sky.launch(
                sky_task,
                cluster_name=sky_clustername,
                dryrun=False,
                stream_logs=True,
            )
        else:
            sky.exec(sky_task, cluster_name=sky_clustername, dryrun=False)

        #[MAK - TODO] get the job status here and print the status.
        jobs = sky.queue(cluster_name=sky_clustername)
        job_status = 'invalid'
        for job in jobs:
            if job['job_name'] == task_name:
                job_status = job.status

        logging.info(f"Task {task_name} completed with status {job_status}")
        remote_outputpath = "/mnt/output"
        rsync_command = (
            f"rsync -Pavz {sky_clustername}:{remote_outputpath} {context.outputdir}"
        )

        run_shellcommand(
            rsync_command,
            install_url="(e.g., 'sudo apt install rsync' on Debian-based systems)",
        )
        # bring cluster down. In case of exceptions leave as is, user can manually copy.
        logging.info("bringing down the sky cluster %s", sky_clustername)
        sky.down(sky_clustername)

    def _configure_aws(self, access_key_id, secret_access_key, region="us-west-2"):
        """Configure aws credentials in local system."""

        cmds = [
            f"aws configure set aws_access_key_id {access_key_id}",
            f"aws configure set aws_secret_access_key {secret_access_key}",
            f"aws configure set region {region}",
        ]

        for cmd in cmds:
            run_shellcommand(cmd, install_url="https://aws.amazon.com/cli/")

    def _configure_gcp(self, sa_json_filepath):
        """Configure GCP credentials in local system."""
        # Create the ~/.config/gcloud directory if it doesn't exist
        gcloud_config_dir = os.path.expanduser("~/.config/gcloud")
        os.makedirs(gcloud_config_dir, exist_ok=True)

        # Copy the JSON key file to application_default_credentials.json
        gcloud_path = os.path.join(
            gcloud_config_dir, "application_default_credentials.json"
        )
        shutil.copyfile(sa_json_filepath, gcloud_path)

        cmd = f"gcloud auth activate-service-account --key-file {sa_json_filepath}"
        run_shellcommand(cmd, install_url="https://cloud.google.com/sdk/")

        with open(gcloud_path, "r", encoding="utf-8") as jfp:
            project_id = json.load(jfp)["project_id"]

        cmd = f"gcloud config set project {project_id}"
        run_shellcommand(cmd, install_url="https://cloud.google.com/sdk/")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_path

    def _configure_azure(self, client_id, client_secret, tenant):
        """Configure azure creds in local system."""
        _, _, _ = client_id, client_secret, tenant

    def _configure_kubernetes(self, kubeconfig_filepath):
        """Configure kubeconfig in local system."""
        _ = kubeconfig_filepath
