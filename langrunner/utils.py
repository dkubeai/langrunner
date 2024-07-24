"""Helper funcs"""

import os
import shutil
import subprocess
import pickle
import codecs
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


def pickle_b64(picklable, filepath):
    serialized = codecs.encode(pickle.dumps(picklable), "base64").decode()
    with open(filepath, "w") as fp:
        fp.write(serialized)
    return serialized


def b64_unpickle(filepath):
    with open(filepath, "r") as fp:
        serialized = fp.read()
    return pickle.loads(codecs.decode(serialized.encode(), "base64"))


def serialize(obj, filepath, mode="json"):
    """serialize a given obj and save in file"""
    if mode == "json":
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(obj, fp)
    if mode == "pickle":
        import dill
        serialized = dill.dumps((obj))
        with open(filepath, "wb") as fp:
            fp.write(serialized)


def deserialize(filepath, mode="json"):
    """load serialized obj from file"""
    if mode == "json":
        with open(filepath, "r", encoding="utf-8") as fp:
            return json.load(fp)
    if mode == "pickle":
        import dill
        with open(filepath, "rb") as fp:
            return dill.loads(fp)
