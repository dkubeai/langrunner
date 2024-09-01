"""Abstract methods for remote dispatching of a fn"""

from abc import ABC, abstractmethod

import json
import sys
import os
import types
import importlib
import functools
import inspect
from contextlib import suppress
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError
from langrunner.context import get_current_context
from langrunner.runners import setup_langrunner


_llama_index_remote_wraps = {
        'SentenceTransformersFinetuneEngine': 'langrunner.llama_index.sentence_transformers.SentenceTransformersFinetuneEngineRemote',
    }

_langrunner_remote_wraps = {
        'SFTFinetuneEngine': 'langrunner.llama_index.sft.SFTFinetuneEngineRemote',
        'FlagEmbeddingFinetuneEngine': 'langrunner.llama_index.flagembedding.FlagEmbeddingFinetuneEngineRemote'
    }
_langchain_remote_wraps = {
        'RunnableBinding': 'langrunner.langchain.chains.RunnableRemote',
        'RunnableSequence': 'langrunner.langchain.chains.RunnableRemote',
        'HuggingFaceEmbeddings': 'langrunner.langchain.huggingface.HuggingFaceEmbeddingsRemote',
        'HuggingFacePipeline': 'langrunner.langchain.huggingface.HuggingFacePipelineRemote'
    }

_autogetn_remote_wraps = {}


_remote_wraps = {
        'llama_index': _llama_index_remote_wraps,
        'langrunner': _langrunner_remote_wraps,
        'langchain': _langchain_remote_wraps,
        'langchain_huggingface': _langchain_remote_wraps,
        'autogen': _autogetn_remote_wraps
    }


class Factory(BaseModel):
    def get(self, class_or_func):
        # return the remote implementation of class from the root package
        if isinstance(class_or_func, types.FunctionType):
            # function type. [MAK - TODO] this must be supported.
            msg = f"Lang runner support for execution of methods not implemented yet. please raise a FR on repo."
            raise NotImplementedError(msg)

        if isinstance(class_or_func, type) :
            # check if the remote wraps exist for the class or arg
            _class = class_or_func
            _class_name = _class.__name__
            root_package = _class.__module__.split('.')[0]
            if root_package not in _remote_wraps:
                msg = f"Lang runner support for {class_or_func} is not yet implemented. please raise a FR on repo."
                raise NotImplementedError(msg)

            if _class_name not in _remote_wraps[root_package]:
                msg = f"Lang runner support for {class_or_func} is not yet implemented. please raise a FR on repo."
                raise NotImplementedError(msg)

            _remote_class = _remote_wraps[root_package][_class_name]

            # import the remote class and return the class to the caller
            module_path, rclass_name = _remote_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[rclass_name])
            return getattr(module, rclass_name)

        if not inspect.isclass(class_or_func) and hasattr(class_or_func, '__class__') and not inspect.isbuiltin(class_or_func):
            _class_name = class_or_func.__class__.__name__
            root_package = class_or_func.__module__.split('.')[0]
            if root_package not in _remote_wraps:
                msg = f"Lang runner support for {class_or_func} is not yet implemented. please raise a FR on repo."
                raise NotImplementedError(msg)

            if _class_name not in _remote_wraps[root_package]:
                msg = f"Lang runner support for {class_or_func} is not yet implemented. please raise a FR on repo."
                raise NotImplementedError(msg)

            _remote_class = _remote_wraps[root_package][_class_name]

            # import the remote class and return the class to the caller
            module_path, rclass_name = _remote_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[rclass_name])
            rclass = getattr(module, rclass_name)
            robj = rclass.__new__(rclass)
            rclass.__init__(robj, class_or_func)

            return class_or_func

        msg = f"Input should be either a class or func and not {type(class_or_func)}"
        raise NotImplementedError(msg)

global_remotes_factory = Factory()


class RemoteRunnable(ABC):
    def __init__(self, *args, **kwargs):
        self.__langclass_initparams = {}
        self.__langclass = self.__class__
        for base in self.__class__.__bases__:
            if base is not RemoteRunnable:
                self.__langclass = base
                if self.initialize_baseclass == False:
                    instance = base.__new__(base)
                else:
                    instance = base(*args, **kwargs)

                func_signature = inspect.signature(base.__init__)
                pos_arg_names = list(func_signature.parameters.keys())[1:]  # Skip 'self'
                init_params = {**kwargs, **dict(zip(pos_arg_names, args))}

                for key, value in init_params.items():
                    setattr(instance, key, value)

                self.__langclass_instance = instance
                self.__langclass_initparams = init_params
                self.__dict__.update(instance.__dict__)

        if hasattr(self, 'remote_default_settings'):
            self.__langrunner = setup_langrunner(self.remote_default_settings)
        else:
            self.__langrunner = setup_langrunner(None)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        try:
            remote_attrs = super().__getattribute__('remote_attrs')
            if name in remote_attrs:
                if callable(attr) is False:
                    raise AttributeError(f"attribute {name} is set as remote attr, cannot access locally.")
                else:
                    def _remote_wrapper(*args, **kwargs):
                        return self._remotecall(attr,*args, **kwargs)
                    return _remote_wrapper
        except Exception as exc:
            pass
        return attr


    def _remotecall(self, fnattr, *args, **kwargs):
        ctx = get_current_context()

        ctx.runid = self.__langrunner.runname
        ctx.rundir = self.__langrunner.rundir
        ctx.inputdir = os.path.join(ctx.rundir, "input")
        ctx.outputdir = os.path.join(ctx.rundir, "output")
        ctx.settings = self.__langrunner.run_settings.json()
        ctx.__REMOTE = True
        ctx.langclass_initparams = {}

        for key, _ in self.__langclass_initparams.items():
            with suppress(PydanticSerializationError):
                value = getattr(self, key)
                self.__langclass_initparams[key] = value
                ctx.langclass_initparams[key] = value

        for rcfn in fnattr(*args, **kwargs):
            # execute the yielded function remotely
            ctx.REMOTE_FUNC = rcfn.__name__
            ctx.REMOTE_CLASS = rcfn.__qualname__.split('.')[0]
            ctx.REMOTE_MODULE = rcfn.__module__
            ctx.REMOTE_REQUIREMENTS = self.remote_requirements
            ctx.LANGCLASS = self.__langclass.__name__

            self.__langrunner.remotecall()        

def remote_main():
    with open("/mnt/input/remote_context.json", "r", encoding="utf-8") as fp:
        context = json.load(fp)
        context = json.loads(context)

        remote_context = get_current_context()
        for key, value in context.items():
            setattr(remote_context, key, value)

        _rclass = remote_context.REMOTE_CLASS
        _rfname = remote_context.REMOTE_FUNC
        _rmodule = remote_context.REMOTE_MODULE
        _langclass = remote_context.LANGCLASS


        # import the remote class and return the class to the caller
        module = __import__(_rmodule, fromlist=[_rclass])
        class_attr = getattr(module, _rclass)
        getattr(class_attr, _rfname)()
