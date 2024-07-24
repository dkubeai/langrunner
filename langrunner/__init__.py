"""Lagrunner module."""
from typing import Union, Type, Callable

from .runners import SkyRunner
from .settings import RunnerSettings
from .runners import setup_langrunner
from .settings import get_default_settings, get_current_settings
from .context import get_current_context, set_current_context
from .remotes import global_remotes_factory


__all__ = [
    "SkyRunner",
    "RunnerSettings",
    "get_default_settings",
    "get_current_settings",
    "setup_langrunner",
    "get_current_context",
    "set_current_context",
]


def runnable(class_or_func: Union[Type, Callable], allow_local: bool=False):
    return global_remotes_factory.get(class_or_func)
