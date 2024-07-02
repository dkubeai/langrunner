from abc import ABC, abstractmethod

import os
import json
import logging
import subprocess
import shutil

import yaml
import sky

from langrunner.llamaindex.finetuning import SentenceTransformersFinetuneEngine
from langrunner.llamaindex.finetuning import runner as lift_runner
from langrunner.utils import lgutils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


is_yamlfile = lambda fp: os.path.isfile(fp) and fp.lower().endswith((".yaml", ".yml"))
is_class = lambda rsrc: hasattr(rsrc, "__class__")


def _yaml_safeload(filepath):
    try:
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except (yaml.YAMLError, IOError):
        return None


def configure_aws(access_key_id, secret_access_key, region="us-west-2"):
    """Configure aws credentials in local system."""

    cmds = [
        f"aws configure set aws_access_key_id {access_key_id}",
        f"aws configure set aws_secret_access_key {secret_access_key}",
        f"aws configure set region {region}",
    ]

    for cmd in cmds:
        lgutils.run_shellcommand(cmd, install_url="https://aws.amazon.com/cli/")


def configure_gcp(sa_json_filepath):
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
    lgutils.run_shellcommand(cmd, install_url="https://cloud.google.com/sdk/")

    with open(gcloud_path, "r") as jfp:
        project_id = json.load(jfp)["project_id"]

    cmd = f"gcloud config set project {project_id}"
    lgutils.run_shellcommand(cmd, install_url="https://cloud.google.com/sdk/")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_path


def configure_azure(client_id, client_secret, tenant):
    """Configure azure creds in local system."""
    _, _, _ = client_id, client_secret, tenant
    pass


class LangRunner(ABC):
    """Base class runner for lang constructs"""

    @abstractmethod
    def setup(self, data):
        """Setup a lang runner."""

    @abstractmethod
    def run(self, resource, **params):
        """Run a resource class remotely"""


class CloudRunner(LangRunner):
    """Runner to run a resource on supported cloud providers."""

    def __init__(self, providers=("aws", "gcp", "azure")):
        self._providers = providers
        self._useclouds = []

    def _validate_cloudcreds(self, provider):
        try:
            cloud_provider_class = getattr(sky.clouds, provider)
            if cloud_provider_class().check_credentials():
                logging.info(
                    "{provider.upper()} (OK). credentials provided for aws cloud are valid."
                )
                self._useclouds.append(provider)
            else:
                logging.error(
                    "{provider.upper()} (SKIP). credentials provided for aws cloud are invalid."
                )
        except Exception as e:
            logging.error(
                f"{provider} is not a valid supported cloud provider. Execption {e.stderr}"
            )

    def _check_creds(self, provider, creds):
        if provider.lower() == "aws":
            # configure aws credentials in the local system
            configure_aws(
                creds["aws_access_key_id"],
                creds["aws_secret_access_key"],
                creds["region"],
            )
        if provider.lower() == "gcp":
            configure_gcp(creds["service_account_filepath"])
            # create default app credentials files in .gcp directory
        if provider.lower() == "azure":
            # create a creds file in .azure directory
            configure_azure(creds["client_id"], creds["client_secret"], creds["tenant"])

        self._validate_cloudcreds(provider.upper())

    def _load_credentials(self, creds):
        for provider in self._providers:
            if (provider in creds) and (creds[provider]["enabled"] == "true"):
                self._check_creds(provider, creds[provider])

    def setup(self, data):
        # Data has to point a file with cloud credentials
        credsf = data

        assert is_yamlfile(
            credsf
        ), "data should point to a yaml file with cloud credentials in it, please see examples/document"

        creds = _yaml_safeload(credsf)
        if any([provider in self._providers for provider in creds.keys()]) is False:
            logging.error(
                f"the credentials file passed is invalid, does not match supported providers {self._providers}"
            )
        else:
            self._load_credentials(creds)

            # the _credentials shouldnt be empty if the values supplied were valid
            assert (
                len(self._useclouds) == 0
            ), "the credentials validation failed, please rerun with valid creds"

    def run(self, resource, **params):
        if is_class(resource):
            if isinstance(resource, SentenceTransformersFinetuneEngine):
                lift_runner.run_oncloud(resource, **params)
            else:
                assert True, f"Unsuported resource type {resource.__class__}"
        else:
            assert True, f"Unsupported resource type {type(resource)}"
