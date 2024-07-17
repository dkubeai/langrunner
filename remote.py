"""Abstract methods for remote dispatching of a fn"""

from abc import ABC, abstractmethod

import json
import sys
import os
import types
import importlib
import functools
import inspect
from pydantic_core import PydanticSerializationError
from langrunner.context import get_current_context
from langrunner.runners import setup_langrunner

LANGREMOTES = dict()


def implements(langclass):
    def wrapper(remoteclass):
        LANGREMOTES.update({remoteclass.__name__: (langclass, remoteclass.__name__)})
        # LANGREMOTES.append((langclass, remoteclass.__name__))
        return remoteclass

    return functools.wraps(
        langclass,
        assigned=(
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
        ),
        updated=(),
    )(wrapper)


def import_remote_class(module_path, target_module, class_name, actual_class_name):
    """to import a remote class."""
    # Import the target module
    target = importlib.import_module(target_module)
    # Get the actual class from the target module
    cls = getattr(target, actual_class_name)

    # Split the module path into parts
    parts = module_path.split(".")

    for i in range(2, len(parts)):
        sub_path = ".".join(parts[:i])
        if sub_path not in sys.modules:
            sub_module = types.ModuleType(sub_path)
            sys.modules[sub_path] = sub_module

    # Set the final fake module to contain the class with the original name
    final_path = ".".join(parts[:-1])
    if final_path not in sys.modules:
        final_module = types.ModuleType(final_path)
        sys.modules[final_path] = final_module
    setattr(sys.modules[final_path], parts[-1], cls)
    return cls


def import_remote_classes():
    """import remote classes."""
    for _, value in LANGREMOTES.items():
        lang_package = value[0].split(".")[0]
        lang_class = value[0].split(".")[-1]
        module_path = f"langrunner.{value[0]}"
        # target_module = f"langrunner.{lang_package}.remotes"
        target_module = f"langrunner.{lang_package}"
        class_name = lang_class
        remote_class_name = value[1]
        import_remote_class(module_path, target_module, class_name, remote_class_name)


registry = {}
def remotefuncdispatch(func):
    """dispatch based on the input values"""

    join = lambda v1, v2=None: v1 + (f'.{v2}' or "")

    def register(cls: str, fn: str):
        value = join(cls, fn)

        def decorator_wrapper(decorator_func):
            registry[value] = decorator_func
            return decorator_func

        return decorator_wrapper

    def wrapper(cls, fn, *args, **kwargs):
        value = join(cls, fn)
        if value in registry:
            return registry[value](*args, **kwargs)
        else:
            raise ValueError(f"Unsupported value '{value}'")

    wrapper.register = register
    return wrapper


@remotefuncdispatch
def remotefunc(cls: str, fn: str):
    """abstract remote func definition"""
    raise NotImplementedError(
        f"remote func for class {cls} and func {fn} is not implemented."
    )


class LangrunnerRemoteResource(ABC):
    """Base class for remote implementation of lang resources."""

    def __init__(self, *args, **kwargs):
        self.resource_init_params = {}

        self.remote_class = self.__class__.__name__
        mapping = LANGREMOTES[self.remote_class]

        # create lang class instance without initializing it as init is set for remote
        parts = mapping[0].split(".")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        mod = __import__(module_path, globals(), locals(), [class_name])
        self.lang_class = getattr(mod, class_name)

        func_signature = inspect.signature(self.lang_class.__init__)
        pos_arg_names = list(func_signature.parameters.keys())[1:]  # Skip 'self'
        self.resource_init_params = {**kwargs, **dict(zip(pos_arg_names, args))}

        self.langclass_obj = self.lang_class.__new__(self.lang_class)

        for key, value in self.resource_init_params.items():
            setattr(self, key, value)
            setattr(self.langclass_obj, key, value)

        self.langrunner = setup_langrunner()

    def __getattribute__(self, name):
        remote_attrs = super().__getattribute__("remote_attrs")
        try:
            if name in remote_attrs:
                raise AttributeError
            return super().__getattribute__(name)
        except AttributeError:
            attr = self.langclass_obj.__getattribute__(name)

            if callable(attr):
                # update the fields to latest values
                for key, _ in self.resource_init_params.items():
                    new_value = getattr(self, key)
                    setattr(self.langclass_obj, key, new_value)

                if name in remote_attrs:
                    remoteattr = super().__getattribute__(name)
                    # handle the function call remotely
                    def remote_wrapperfn(*args, **kwargs):
                        return self._remote_wrapperfn(remoteattr, *args, **kwargs)
                    return remote_wrapperfn
                return attr
            return attr

    def _remote_wrapperfn(self, remoteattr, *args, **kwargs):
        ctx = get_current_context()
        ctx.runid = self.langrunner.runname
        ctx.rundir = self.langrunner.rundir
        ctx.inputdir = os.path.join(ctx.rundir, "input")
        ctx.outputdir = os.path.join(ctx.rundir, "output")
        ctx.settings = self.langrunner.run_settings.json()
        ctx.__REMOTE__ = True

        ctx.langclass_initparams = {}
        for key, _ in self.resource_init_params.items():
            try:
                value = getattr(self, key)
                ctx.langclass_initparams.update({key:value})
            except PydanticSerializationError:
                ctx.langclass_initparams.update(key, None)

        for remotefn in remoteattr(*args, **kwargs):
            # execute the returned func remotely
            #ctx.REMOTE_FUNC = remotefn.__name__
            ctx.REMOTE_FUNC = remoteattr.__name__
            ctx.REMOTE_CLASS = self.__class__.__name__

            parts = self.__module__.split('.')
            lang_package = parts[1]
            ctx.REMOTE_REQUIREMENTS_FILE = f"{lang_package}/requirements.txt"

            self.langrunner.remotecall()

    @staticmethod
    def remotefn_decorator(func):
        """Decorator to manage switch between local and remote exec"""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            ctx = get_current_context()
            if ctx.__REMOTE__ is False:
                return func(self, *args, **kwargs)
            else:
                ctx.REMOTE_FUNC = func.__name__
                ctx.REMOTE_CLASS = self.__class__.__name__
                self.langrunner.remotecall()
                return None

        return wrapper

def remote_main():
    with open("/mnt/input/remote_context.json", "r", encoding="utf-8") as fp:
        context = json.load(fp)
        context = json.loads(context)

        remote_context = get_current_context()
        for key, value in context.items():
            setattr(remote_context, key, value)

        rc = remote_context.REMOTE_CLASS
        rfn = remote_context.REMOTE_FUNC

        remotefunc(rc, rfn)
