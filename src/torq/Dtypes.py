import torch

_REAL_FROM_COMPLEX = {torch.complex64: torch.float32, torch.complex128: torch.float64}
_COMPLEX_FROM_REAL = {
    torch.float16: torch.complex64,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

def complex_dtype_like(x: torch.Tensor | None) -> torch.dtype:
    if x is None:
        return torch.complex64
    dt = x.dtype
    if dt.is_complex:
        return dt
    return _COMPLEX_FROM_REAL.get(dt, torch.complex64)

def resolve_like(x: torch.Tensor | None = None,
                 dtype: torch.dtype | None = None,
                 device=None) -> tuple[torch.dtype, torch.device | None]:
    dc = dtype or complex_dtype_like(x)
    dv = device if device is not None else (x.device if x is not None else None)
    return dc, dv

