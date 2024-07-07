"""Helper funcs"""

import os
import shutil
import subprocess
import pickle
import codecs
import dill
import json


def run_shellcommand(cmdstr, install_url=""):
    """Helper fn to run shell commands"""
    shell_command = cmdstr.split()
    binary = shell_command[0]
    try:
        # Run shell command
        subprocess.run(
            shell_command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed Executing shell cmd: {cmdstr} with error: {e.stderr}")
        raise e
    except FileNotFoundError as e:
        fnfe = f"{binary} CLI is not installed. Please install and retry. {install_url}"
        print(f"{fnfe}")
        raise e


def configure_aws(access_key_id, secret_access_key, region="us-west-2"):
    """Configure aws credentials in local system."""

    cmds = [
        f"aws configure set aws_access_key_id {access_key_id}",
        f"aws configure set aws_secret_access_key {secret_access_key}",
        f"aws configure set region {region}",
    ]

    for cmd in cmds:
        run_shellcommand(cmd, install_url="https://aws.amazon.com/cli/")


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
    run_shellcommand(cmd, install_url="https://cloud.google.com/sdk/")

    with open(gcloud_path, "r") as jfp:
        project_id = json.load(jfp)["project_id"]

    cmd = f"gcloud config set project {project_id}"
    run_shellcommand(cmd, install_url="https://cloud.google.com/sdk/")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcloud_path


def configure_azure(client_id, client_secret, tenant):
    """Configure azure creds in local system."""
    _, _, _ = client_id, client_secret, tenant
    pass


"""
def pickle_b64(picklable, filepath):
    serialized = codecs.encode(pickle.dumps(picklable), "base64").decode()
    with open(filepath, "w") as fp:
        fp.write(serialized)
    return serialized


def b64_unpickle(filepath):
    with open(filepath, "r") as fp:
        serialized = fp.read()
    return pickle.loads(codecs.decode(serialized.encode(), "base64"))
"""


def serialize(obj, filepath, mode="json"):
    """serialize a given obj and save in file"""
    if mode == "json":
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
    if mode == "pickle":
        serialized = dill.dumps((obj))
        with open(filepath, "wb") as fp:
            fp.write(serialized)


def deserialize(filepath, mode="json"):
    """load serialized obj from file"""
    if mode == "json":
        with open(filepath, "r", encoding="utf-8") as fp:
            return json.load(fp)
    if mode == "pickle":
        with open(filepath, "rb") as fp:
            return dill.loads(fp)
