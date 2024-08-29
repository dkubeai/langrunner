"""Runners to run lang functions on remote clusters."""

from abc import ABC, abstractmethod

import os
import time
import json
import logging
import uuid
import shutil
import tempfile
from pathlib import Path
from contextlib import suppress

from .settings import get_current_settings
from .context import get_current_context
from .utils import run_shellcommand

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import atexit

def on_exit():
    # skip if being called inside the sky execution
    if os.getenv("LANGRUNNER_SKYTASK", "false") == "true":
        return

    import sky

    logging.info("Shutting down all sky services..")
    try:
        sky.serve.down(all=True, purge=True)

        # wait for the services to go down
        while sky.serve.status():
            time.sleep(5)
    except Exception as e:
        logging.exception(e)

    try:
        logging.info("Shutting down all sky clusters..")
        # get all the clusters
        while (clusters := sky.status()):
            for cluster in clusters:
                logging.info(f"Shutting down sky cluster ({cluster['name']})")
                sky.down(cluster_name = cluster['name'])
            time.sleep(5)
    except Exception as e:
        logging.exception(e)

# Register the function to be called on exit
atexit.register(on_exit)

# this function must be cached so that we dont execute if there is no change.
def setup_langrunner(settings):
    """setup/select a runner based on the compute resources requested."""
    settings = settings or get_current_settings
    sk = SkyRunner()
    sk.setup(settings)
    return sk


class LangRunner(ABC):
    """Base class for different runners."""

    run_settings = get_current_settings
    runid = str(uuid.uuid4())
    rundir = os.path.join(os.path.expanduser("~"), f"./langrunner/{runid}")
    runname = "lgr-run"

    @abstractmethod
    def setup(self, settings):
        """setup a runner."""
        pass

    @abstractmethod
    def remotecall(self, func, *args, **kwargs):
        """to execute a func remotely."""
        pass


class SkyPatcher(ABC):
    @staticmethod
    def load_kubernetes_config():
        import kubernetes

        try:
            kubernetes.config.load_kube_config()
        except Exception as exc:
            print('Failed to load Kubernetes configuration. Please check if your kubeconfig file exists at ~/.kube/config')
            #raise exc

    @staticmethod
    def configure_skypilot_system_namespace(*args, **kwargs):
        # need not create skypilot-system namespace
        # https://github.com/skypilot-org/skypilot/blob/master/sky/utils/kubernetes/generate_kubeconfig.sh
        # doesnt add permission to create namespaces
        return

    @staticmethod
    def k8s_patchstart(enabled_clouds):
        import sky
        for cloud in enabled_clouds:
            if cloud == sky.clouds.Kubernetes:
                patch1 = mock.patch('sky.adaptors.kubernetes._load_config', SkyPatcher.load_kubernetes_config)
                patch2 = mock.patch('sky.provision.kubernetes.config._configure_skypilot_system_namespace', SkyPatcher.configure_skypilot_system_namespace)
                return patch1.start(), patch2.start()
        return None, None

    @staticmethod
    def k8s_patchstop(patches):
        for patch in patches:
            if patch is not None:
                patch.stop()

class SkyRunner(LangRunner):
    """Class to run lang resources on public cloud infra"""

    providers = ["aws", "gcp", "azure", "kubernetes"]


    def setup(self, settings):
        import yaml

        self.run_settings = settings
        use_runners = settings.use_runners

        if use_runners == "auto":
            use_runners = self.providers


        _cwd_creds_path = os.path.join(os.getcwd(), "credentials.yaml")
        _env_creds_path = os.getenv("LANGRUNNER_CREDS_FILEPATH", _cwd_creds_path)
        _creds_paths = [settings.credentials_file, _cwd_creds_path, _env_creds_path]

        _creds_file = _cwd_creds_path

        for path in _creds_paths:
            if os.path.exists(path):
                logging.info(f"Reading credentials from path > {path}")
                _creds_file = path
                break

        #with open(settings.credentials_file, 'r') as fp:
        with open(_creds_file, 'r') as fp:
            creds = yaml.safe_load(fp)

        for runner in use_runners:
            if creds[runner]["enabled"] is True:
                params = dict(filter(lambda item: item[0] != 'enabled', creds[runner].items()))
                fn = getattr(self, f"_configure_{runner}")
                fn(**params)

        os.makedirs(self.rundir, exist_ok=True)
        os.makedirs(os.path.join(self.rundir, "input"))
        os.makedirs(os.path.join(self.rundir, "output"))

        import sky
        print(f"Checking the compute {self.providers} configured/enabled.")
        with suppress(SystemExit):
            sky.check.check(clouds=self.providers, quiet=True)
        self.enabled_clouds = sky.check.get_cached_enabled_clouds_or_refresh()

    def _sky_servestate(self, service_name):
        import sky
        from sky.serve.serve_state import ServiceStatus, ReplicaStatus

        service = sky.serve.status(service_names = service_name)[0]
        serve_state = ServiceStatus(service['status'])
        replica_state = ReplicaStatus(service['replica_info'][0]['status'])
        return serve_state, replica_state

    def _serve_poll(self, service_name):
        import time
        from yaspin import yaspin
        from termcolor import colored
        from sky.serve.serve_state import ServiceStatus

        reset = '\033[0m'
        total_elapsed_time = 0
        begin_time = time.time()
        state = prevstate = ('', '')

        def break_condition(state, prevstate, total_elapsed_time):
            return prevstate != state or state[0] == ServiceStatus.READY or state[0] in ServiceStatus.failed_statuses() or total_elapsed_time > (3600/2)

        while True:
            if break_condition(state, prevstate, total_elapsed_time): break
            start_time = time.time()
            with yaspin() as spinner:
                while True:
                    if break_condition(state, prevstate, total_elapsed_time): break

                    state = self._sky_servestate(service_name)

                    elapsed_time = time.time() - start_time
                    spinner.text = f"service = {service_name} - service state = {state[0].colored_str()} - replica state = {state[1].colored_str()} ................ {elapsed_time:.2f}s"
                    time.sleep(5)

                #spinner.ok()
                prevstate = state = self._sky_servestate(service_name)

            total_elapsed_time = time.time() - begin_time

        state = self._sky_servestate(service_name)
        if state[0] == ServiceStatus.READY:
            print(colored(f'service = {service_name} is {state[0].colored_str()}.', 'green') + reset, end="\n")
        if state[0] in ServiceStatus.failed_statuses():
            print(colored(f'service = {service_name} failed {state[0].colored_str()}, please check runner logs for the issue', 'red') + reset, end="\n")
        if total_elapsed_time > (3600/2):
            print(colored(f'[{service_name}] is not ready for 30mins. Please check the runner logs for actual issue.', 'red') + reset, end="\n")

        
    def remotecall(self):
        import sky
        from sky.serve.service_spec import SkyServiceSpec
        from sky.serve import constants
        import unittest.mock as mock
        from unique_names_generator import get_random_name

        context = get_current_context()

        file_mounts = {"/mnt/input": context.inputdir}
        task_name = get_random_name(separator="-", style="lowercase")
        # task_name = f"{self.run_name} - {task_name}"


        temp_dir = tempfile.mkdtemp()
        workdir = os.path.expanduser(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        workdir = os.path.join(workdir, 'langrunner')
        shutil.copytree(workdir, os.path.join(temp_dir, 'langrunner'))

        for pycache in Path(temp_dir).rglob('__pycache__'):
            shutil.rmtree(pycache)

        workdir = temp_dir

        context_fp = os.path.join(context.inputdir, "remote_context.json")
        with open(context_fp, "w") as fp:
            json.dump(context.json(), fp)

        script = '''
import sys
import os

try:
    import langrunner
except ImportError:
    sys.path.append(\'./\')

from langrunner import remotes
remotes.remote_main()
'''

        rc = f'conda activate lgrenv; python -c "{script}"'

        basic_requirements = ['pydantic==2.8.2', 'pydantic_core==2.20.1']
        remote_requirements = ' '.join(context.REMOTE_REQUIREMENTS) + ' ' + ' ' .join(basic_requirements)
        setup_command = f"conda create -n lgrenv python=3.10.14 -y; conda activate lgrenv; pip install {remote_requirements}"

        sky_task = sky.Task(
            name=task_name,
            setup = setup_command,
            run=rc,
            workdir=workdir,
            envs={"LANGRUNNER_SKYTASK": "true"},
        )

        sky_task.set_file_mounts(file_mounts)

        accelerators = None

        if self.run_settings.use_accelerator == 'auto':
            # [MAK - TODO] - From sky get if accelerators are enabled.
            # if yes then use it. For now setting it to True
            accelerators = "T4:1"
        elif self.run_settings.use_accelerator is True:
            accl = self.run_settings.accelerator
            accl_count = self.run_settings.accelerator_count
            accelerators = f"{accl}:{accl_count}"

        sky_task.set_resources(
            sky.Resources(
                accelerators=accelerators,
                use_spot=self.run_settings.use_spot,
                memory=self.run_settings.memory,
                cpus=self.run_settings.cpus,
                ports=self.run_settings.ports
            ),
        )

        #sky_clustername = f"lgr-{self.runname}"
        sky_clustername = f"lgr-{task_name}"

        k8s_patchers = SkyPatcher.k8s_patchstart(self.enabled_clouds)

        if context.remote_tasktype == 'service':
            readiness_path = context.REMOTE_TASK_SERVICE_PROBE_URL
            service = SkyServiceSpec(readiness_path = readiness_path, 
                    readiness_timeout_seconds=constants.DEFAULT_READINESS_PROBE_TIMEOUT_SECONDS,
                    initial_delay_seconds=constants.DEFAULT_INITIAL_DELAY_SECONDS, 
                    min_replicas=constants.DEFAULT_MIN_REPLICAS, 
                    max_replicas=constants.DEFAULT_MIN_REPLICAS, 
                    upscale_delay_seconds=constants.AUTOSCALER_DEFAULT_UPSCALE_DELAY_SECONDS, 
                    downscale_delay_seconds=constants.AUTOSCALER_DEFAULT_DOWNSCALE_DELAY_SECONDS) 
            sky_task.set_service(service)

            services = []
            with suppress(sky.exceptions.ClusterNotUpError):
                services = sky.serve.status(service_names=sky_clustername)
            if len(services) == 0:
                from sky.sky_logging import silent
                with silent():
                    sky.serve.up(sky_task, service_name=sky_clustername)
                print("Lanuching compute with required resources. Waiting for it to be up.", flush=True, end="\n")
            else:
                print(f"Compute with name {sky_clustername} is already up. Checking if all services are READY", flush=True, end="\n")

            self._serve_poll(sky_clustername)

            services = sky.serve.status(service_names=sky_clustername) #refresh services list
            context.REMOTE_ENDPOINT = services[0]['replica_info'][0]['endpoint']

            #sky.serve.tail_logs(sky_clustername, target='replica', follow=True, replica_id=1)
        else:
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

            jobs = sky.queue(cluster_name=sky_clustername)
            job_status = 'invalid'
            for job in jobs:
                if job['job_name'] == task_name:
                    job_status = job['status']

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
            #logging.info("bringing down the sky cluster %s", sky_clustername)
            #sky.down(sky_clustername)
            shutil.rmtree(temp_dir)

        SkyPatcher.k8s_patchstop(k8s_patchers)

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
