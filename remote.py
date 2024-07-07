"""Abstract methods for remote dispatching of a fn"""

from typing import Tuple, Any
import fire

from langrunner.utils import fndispatch


@fndispatch
def check(cls, fn=None) -> bool:
    """check if remote execution is supported for the cls"""
    return False  # default value


@fndispatch
def prolog(cls, fn, inputdir, outputdir, cls_params=None, fn_params=None):
    """prepare for remote execution"""
    raise NotImplementedError("Unsupported resource type {cls}:{fn}")


@fndispatch
def run(cls, fn, run_name):
    """execute the resource remotely."""
    raise NotImplementedError("Unsupported resource type {cls}:{fn}")


@fndispatch
def epilog(cls, fn, outputdir) -> Tuple[Any, ...]:
    """prepare for remote execution"""
    raise NotImplementedError("Unsupported resource type {cls}:{fn}")


def execute(cls, fn, run_name):
    """wrapper func which executes remotely"""
    run(cls, fn, run_name)


if __name__ == "__main__":
    fire.Fire(execute)
