"""Runners to run lang functions on remote clusters."""

from abc import ABC, abstractmethod

import os
import logging
import inspect
import importlib
import uuid
from typing import Type, Tuple, Dict, List, Any

from pydantic import BaseModel, validator
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES
import yaml
import sky

from langrunner.remote import check, prolog, epilog
from langrunner.utils import run_shellcommand

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LangRunner(ABC):
    """Base class runner for lang constructs"""

    lang_resource: Type
    lang_module: str
    resource_obj: Any = None
    run_name: str = str(uuid.uuid4())
    resource_params: dict = {}
    resource_isremote: bool = False
    rundir: str = None
    SUPPORTED_MODULES: Tuple[str, ...] = ("llama_index", "langchain", "autogen")
    MODULES_MAP: Dict[str, str] = {
        "llama_index": "li",
        "langchain": "lc",
        "autogen": "ag",
    }

    @abstractmethod
    def remotecall(self, func, *args, **kwargs):
        """to execute a func remotely."""
        pass

    @validator("lang_resource")
    @classmethod
    def validate_isclass(cls, v):
        """Validates if v is of type class"""
        if not isinstance(v, type):
            raise ValueError("lang_resource argument must be a class.")
        return v

    @validator("lang_module")
    @classmethod
    def validate_module(cls, v):
        """Validates if v is from supported module"""
        if v not in cls.SUPPORTED_MODULES:
            raise ValueError(
                "lang_resource should be from one of the supported module {SUPPORTED_MODULES}"
            )
        return v

    @classmethod
    def from_resource(cls, resource, **resource_params):
        """Return an object based on the lang resource"""

        obj = cls()
        obj.lang_resource = resource

        obj.resource_params = resource_params

        obj.lang_module = resource.__module__.split(".")[0]

        # check if this lang resource is supported for remote execution.
        # if yes then save the lang rsrc for remote init.

        module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), obj.lang_module, "__init__.py")
        print(f"{module_path}", flush=True)
        spec = importlib.util.spec_from_file_location(obj.lang_module, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        #imp = importlib.import_module(obj.lang_module, package="langrunner")
        # isremote = module.isremote(obj.lang_resource)
        # obj.resource_isremote = isremote

        isremote = check(cls=obj.lang_resource, fn=None)
        obj.resource_isremote = isremote

        if isremote is False:
            # initialize the object locally
            self.resource_obj = obj.lang_resource(**resource_params)
            return self.resource_obj
        else:
            # do not initialize object locally. Init on the fly when a function in remote class is called.
            obj.rundir = os.path.expanduser(f"~/.cloudrunner/{obj.run_name}")
            inputs_dir = os.path.join(obj.rundir, "input")
            os.makedirs(inputs_dir, exist_ok=True)
            outputs_dir = os.path.join(obj.rundir, "output")
            os.makedirs(outputs_dir, exist_ok=True)
        return obj

    def __getattribute__(self, item):
        try:
            # check if call is for any attr of this runner class
            return super().__getattribute__(item)
        except AttributeError as ae:
            # if the resource is not enabled for remote exec
            if self.resource_isremote is False:
                return self.resource_obj.__getattribute__(item)
            else:
                pubfns = [m[0] for m in inspect.getmembers(self.lang_resource, predicate=inspect.isroutine) if not m[0].startswith('_')]
                if item in pubfns:
                    # check if this pubfn is supported for remote exec
                    if check(self.lang_resource, item) is False:
                        raise ValueError(f"{self.lang_resource} is set for remote but public method {item} is not supported for remote.")
                    else:
                        def remote_wrapper(*args, **kwargs):
                            funcattr = getattr(self.lang_resource, item)
                            return self.remotecall(funcattr, *args, **kwargs)
                        return remote_wrapper
                else:
                    #at this point, it could be one of the parameter of the lang class
                    return self.resource_params[item]
        else:
            raise ae


class CloudRunner(LangRunner):
    """Class to run lang resources on public cloud infra"""

    cloud_providers = ["AWS", "GCP", "Azure", "Kubernetes"]

    @classmethod
    def from_resource(cls, resource, **resource_params):
        """Return an object based on the lang resource"""
        obj = super(CloudRunner, cls).from_resource(resource, **resource_params)

        if isinstance(obj, CloudRunner) and obj.resource_isremote is True:
            obj.setup_cloud()
        return obj

    def setup_cloud(self, credsf="credentials.yaml"):
        """Setup cloud and check credentials"""
        is_yamlfp = os.path.isfile(credsf) and credsf.lower().endswith(
            (".yaml", ".yml")
        )
        assert is_yamlfp, "data should point to a yaml file with cloud credentials in it, please see examples/document"

        with open(credsf, "r", encoding="utf-8") as file:
            creds = yaml.safe_load(file)

        for provider in self.cloud_providers:
            if (provider in creds) and creds[provider]["enabled"]:

                module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "__init__.py")
                spec = importlib.util.spec_from_file_location("utils", module_path)
                umod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(umod)

                del creds[provider]["enabled"]
                fn = getattr(umod, f"configure_{provider.lower()}")
                fn(**creds[provider])
                try:
                    cloud_provider_class = getattr(sky.clouds, provider.upper())
                    if cloud_provider_class().check_credentials():
                        logging.info(
                            "{provider.upper()} (OK). credentials provided are valid."
                        )
                    else:
                        logging.error(
                            "{provider.upper()} (SKIP). credentials provided are invalid."
                        )
                except Exception as e:
                    logging.error(
                        "%s is not a valid supported cloud provider. Exception %s",
                        provider,
                        e,
                    )
                    raise e

    def remotecall(self, func, *args, **kwargs):
        inputs_dir = os.path.join(self.rundir, "input")
        outputs_dir = os.path.join(self.rundir, "output")

        func_signature = inspect.signature(func)
        pos_arg_names = list(func_signature.parameters.keys())[: len(args)]
        fn_params = {**kwargs, **dict(zip(pos_arg_names, args))}

        prolog(
            self.lang_resource,
            func.__name__,
            inputs_dir,
            outputs_dir,
            cls_params=self.resource_params,
            fn_params=fn_params,
        )

        file_mounts = {"/mnt/input": inputs_dir}
        task_name = get_random_name(separator="-", style="lowercase")
        #task_name = f"{self.run_name} - {task_name}"
        workdir = os.path.expanduser("./")

        rc_args = " ".join([self.lang_resource.__name__, func.__name__, self.run_name])
        rc = f"python remote.py {rc_args}"

        sky_task = sky.Task(
            name=task_name,
            setup="pip install -r requirements.txt",
            run=rc,
            workdir=workdir,
        )

        sky_task.set_file_mounts(file_mounts)

        sky_task.set_resources(
            sky.Resources(
                #cloud=[
                    #getattr(sky.clouds, provider)() for provider in self.cloud_providers
                #],
                accelerators={"T4": 1},
                use_spot=True,
            )
        )

        #sky_clustername = f"lgr-{self.run_name}"
        sky_clustername = f"lgr-{task_name}"

        clusters = sky.status()
        if sky_clustername not in [cluster["name"] for cluster in clusters]:
            sky.launch(
                sky_task,
                cluster_name=sky_clustername,
                dryrun=False,
                stream_logs=True,
            )

        sky.exec(sky_task, cluster_name=sky_clustername, dryrun=False)

        remote_outputpath = "/mnt/output"
        rsync_command = (
            f"rsync -avz {sky_clustername}:{remote_outputpath} {outputs_dir}"
        )

        run_shellcommand(
            rsync_command,
            install_url="(e.g., 'sudo apt install rsync' on Debian-based systems)",
        )
        # bring cluster down. In case of exceptions leave as is, user can manually copy.
        logging.info("bringing down the sky cluster %s", sky_clustername)
        sky.down(sky_clustername)

        return epilog(self.lang_resource, func.__name__, outputs_dir)