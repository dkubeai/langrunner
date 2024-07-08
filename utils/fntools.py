"""helpful function tool methods"""


def fndispatch(func):
    """dispatch based on the value not type."""
    registry = {}

    join = lambda v1, v2=None: v1 + (v2 or "")

    def register_value(cls: str, fn: str):
        value = join(cls.__name__, fn)

        def decorator_wrapper(decorator_func):
            registry[value] = decorator_func
            return decorator_func

        return decorator_wrapper

    def wrapper(cls, fn, *args, **kwargs):
        value = join(cls.__name__, fn)
        if value in registry:
            return registry[value](cls, fn, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported value '{value}'")

    wrapper.register = register_value
    return wrapper
