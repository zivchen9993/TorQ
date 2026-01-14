# torq/_like.py
from functools import wraps
from .Dtypes import resolve_like

def likeable(fn):
    @wraps(fn)
    def wrapper(*args, x=None, dtype=None, device=None, **kwargs):
        dc, dv = resolve_like(x, dtype, device)
        return fn(*args, dtype=dc, device=dv, **kwargs)
    return wrapper