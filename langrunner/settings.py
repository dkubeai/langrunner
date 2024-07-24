import os
import functools
from typing import Union, List, Any, Optional, Dict
from pydantic import BaseModel, ValidationInfo, field_validator, ConfigDict
from pathlib import Path


def set_currentsettings_oncreate(cls):
    cls_init = cls.__init__

    @functools.wraps(cls_init)
    def wrapped_init(self, *args, **kwargs):
        # Call the original constructor
        cls_init(self, *args, **kwargs)

        cls.set_current_settings(self)

    cls.__init__ = wrapped_init
    return cls


@set_currentsettings_oncreate
class RunnerSettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        json_encoders={
            Any: lambda v: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
        },
    )

    cpus: Union[None, int, str] = None
    memory: Union[None, str, int] = None
    accelerator: str = "T4"
    accelerator_count: int = 1
    instance_type: str = "auto"
    use_runners: Union[List[Any], str] = "auto"
    use_accelerator: Union[bool, str] = True #"auto"
    use_spot: bool = False
    ports: int = 8080
    envs: Optional[Dict[str, str]] = None
    credentials_file: Union[str, Path] = os.path.join(
        os.path.expanduser("~"), ".langrunner", "credentials.yaml"
    )

    @classmethod
    def set_current_settings(cls, settings):
        cls._current_settings = settings

    @classmethod
    def get_current_settings(cls):
        return cls._current_settings

    @field_validator("cpus", "memory", "accelerator_count")
    def field_must_be_ge_1(cls, value, info: ValidationInfo) -> int:
        if isinstance(value, int) and value < 1:
            raise ValueError(f"{info.field_name} must be set to value >= 1")
        return value

    @field_validator("credentials_file")
    def field_must_be_yaml(cls, value, info: ValidationInfo) -> Union[str, Path]:
        if os.path.isfile(value):
            if value.lower().endswith((".yaml", ".yml")):
                return value
        raise ValueError(f"{info.field_name} must be a path to a valid yaml file")


get_default_settings = RunnerSettings()
get_current_settings = RunnerSettings.get_current_settings() or get_default_settings
