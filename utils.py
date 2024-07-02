"""Helper funcs"""

import subprocess


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
