from functools import wraps

MBPS_TO_BPNS = 0.001  # 1 Mbps = 1 bits per 1000 nanosecond
BYTE_2_BIT = 8

def requires_attribute(attr, msg=None):
    """
    Decorator to ensure that a method is only called if a certain attribute is set on the instance.
    Args:
        attr (str): The name of the required attribute.
        msg (str, optional): Custom error message if the attribute is not set. Defaults to None.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            value = getattr(self, attr, None)

            if value is None:
                raise ValueError(
                    msg or f"{func.__name__} requires '{attr}' to be set."
                )

            return func(self, *args, **kwargs)

        return wrapper
    return decorator

def require_instance(func):
    """Decorator: call self.no_instance_assigned() if no usecase is active."""
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "usecase", None):
            return self.no_instance_assigned()
        return func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

