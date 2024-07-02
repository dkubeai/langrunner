import tempfile
import os
import subprocess
import logging

import sky

from langrunner.finetuning import SentenceTransformersFinetuneEngine
import langrunner.utils as lgutils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sky_clustercreate(name, task):
    """Create sky cluster with given name"""
    clusters = sky.status()
    if name not in [cluster["name"] for cluster in clusters]:
        sky.launch(
            task,
            cluster_name=name,
            dryrun=False,
            stream_logs=True,
        )


def run_oncloud(clouds, resource, **params):
    """Run the given resource on the best cloud"""
    params = params  # should use this parameter
    if isinstance(resource, SentenceTransformersFinetuneEngine):
        dataset = resource.dataset
        val_dataset = resource.val_dataset

        model_output_path = resource.model_output_path

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_fp = os.path.join(temp_dir, "train_dataset.json")
            # Create a file for the dataset argument
            dataset.save_json(dataset_fp)

            file_mounts = {dataset_fp: "/mnt/train_dataset.json"}
            resource.dataset = "/mnt/train_dataset.json"

            if val_dataset is not None:
                valdataset_fp = os.path.join(temp_dir, "val_dataset.json")
                # Create a file for the dataset argument
                val_dataset.save_json(valdataset_fp)

                file_mounts = {valdataset_fp: "/mnt/val_dataset.json"}
                resource.val_dataset = "/mnt/val_dataset.json"

            resource.model_output_path = "/mnt/finetuned_model"
            task_name = get_random_name(separator="-", style="lowercase")
            workdir = os.path.expanduser("./")

            command_args = " ".join(
                [f"{key}={value}" for key, value in resource.__dict__.items()]
            )
            run_command = f"python st.py {command_args}"

            sky_task = sky.Task(
                name=task_name,
                setup="pip install -r requirements.txt",
                run=run_command,
                workdir=workdir,
            )

            sky_task.set_file_mounts(file_mounts)

            sky_task.set_resources(
                sky.Resources(
                    cloud=[
                        getattr(sky.clouds, provider.upper()) for provider in clouds
                    ],
                    accelerators={"T4": 1},
                    use_spot=True,
                )
            )

            sky_clustername = "langrunner_cluster"
            sky_clustercreate(sky_clustername, sky_task)
            sky.exec(sky_task, cluster_name=sky_clustername, dryrun=False)

            os.makedirs(model_output_path, exist_ok=True)

            output_path = resource.model_output_path
            rsync_command = (
                f"rsync -avz {sky_clustername}:{output_path} {model_output_path}"
            )

            cmd = "rsync_command"
            lgutils.run_shellcommand(
                "rsync_command",
                install_url="(e.g., 'sudo apt install rsync' on Debian-based systems)",
            )
            # bring cluster down. In case of exceptions leave as is, user can manually copy.
            logging.info(f"Bringing down the sky cluster {sky_clustername}")
            sky.down(sky_clustername)
