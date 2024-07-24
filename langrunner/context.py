import json
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict
import contextlib
import functools


class JSONSerializable(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            Any: lambda v: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
        },
    )

    def __setattr__(self, key, value):
        try:
            json.dumps(value)
            super().__setattr__(key, value)
        except (TypeError, ValueError):
            raise ValueError(f"Value '{value}' is not JSON serializable")


def set_currentcontext_oncreate(cls):
    cls_init = cls.__init__

    @functools.wraps(cls_init)
    def wrapped_init(self, *args, **kwargs):
        # Call the original constructor
        cls_init(self, *args, **kwargs)

        cls.set_current_context(self)

    cls.__init__ = wrapped_init
    return cls

@set_currentcontext_oncreate
class Context(JSONSerializable):
    @classmethod
    def set_current_context(cls, context):
        cls._current_context = context

    @classmethod
    def get_current_context(cls):
        return cls._current_context

get_default_context = Context()
get_current_context = Context.get_current_context
set_current_context = Context.set_current_context
