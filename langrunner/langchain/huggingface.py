import subprocess
from typing import Any, Iterator, List, Mapping, Optional

from langchain_openai import OpenAI
from langrunner.remotes import RemoteRunnable
from langrunner import RunnerSettings, get_current_context

class HuggingFacePipelineRemote(RemoteRunnable):

    remote_attrs = ["deploy_model"]
    remote_requirements = ["vllm==0.5.0.post1", "langchain==0.2.10", "langchain-openai==0.1.17", "fastapi", "sse_starlette"]
    remote_default_settings = RunnerSettings(memory="64+", use_accelerator=True, accelerator='A10G', accelerator_count=1, ports=8080)
    initialize_baseclass = False
    llm: OpenAI = None

    @staticmethod
    def _serve_model():
        context = get_current_context()
        model_id = context.serve_model_id
        ngpus = 1

        vllm_command = f"python -m vllm.entrypoints.openai.api_server --tensor-parallel-size {ngpus} --host 0.0.0.0 --port 8080 --model {model_id}"

        result = subprocess.run(vllm_command.split(" "), capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Serving model {model_id} failed with error {result.stderr}")

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        backend: str = "default",
        device: Optional[int] = -1,
        device_map: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> Any:    
        obj = cls()
        obj.model_id = model_id
        obj.model_kwargs = model_kwargs
        obj.pipeline_kwargs = pipeline_kwargs
        obj.deploy_model()
        return obj.llm

    def deploy_model(self):
        context = get_current_context()
        context.serve_model_id = self.model_id
        context.remote_tasktype = 'service'
        context.REMOTE_TASK_SERVICE_PROBE_URL = "/v1/models"

        yield HuggingFacePipelineRemote._serve_model

        serve_endpoint = context.REMOTE_ENDPOINT

        self.llm = OpenAI(model_name=self.model_id, openai_api_base=f"{serve_endpoint}/v1", openai_api_key="langrunner")
        return self.llm
