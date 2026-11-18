"""DLPack views of driver-mapped pointers owned by the TileLang allocator."""

import ctypes as C
from functools import lru_cache
from pathlib import Path

import torch


_new_capsule = C.pythonapi.PyCapsule_New
_new_capsule.argtypes = (C.c_void_p, C.c_char_p, C.c_void_p)
_new_capsule.restype = C.py_object
_capsule_name = b"dltensor"
_valid_capsule = C.pythonapi.PyCapsule_IsValid
_valid_capsule.argtypes = (C.py_object, C.c_char_p)
_valid_capsule.restype = C.c_int
_dtype_codes = {
    "float16": (2, 16), "float32": (2, 32), "bfloat16": (4, 16),
    "int32": (0, 32), "uint32": (0, 32), "int64": (0, 64), "uint64": (0, 64),
}


@lru_cache(maxsize=1)
def _library():
    lib = C.CDLL(str(Path(__file__).with_name("native") / "_vmm_dlpack.so"))
    lib.tilelang_vmm_view.argtypes = (
        C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_int64), C.c_uint8, C.c_uint8,
    )
    lib.tilelang_vmm_view.restype = C.c_void_p
    lib.tilelang_vmm_release.argtypes = (C.c_void_p,)
    lib.tilelang_vmm_release.restype = None
    return lib


def vmm_tensor_view(ptr, shape, dtype, device, owner=None):
    """Wrap a CUDA VMM address whose device is already known by the allocator."""
    if not ptr or any(dim <= 0 for dim in shape):
        raise ValueError("a nonempty driver-mapped tensor is required")
    dims = (C.c_int64 * len(shape))(*shape)
    code, bits = _dtype_codes[dtype]
    lib = _library()
    address = lib.tilelang_vmm_view(ptr, device, len(shape), dims, code, bits)
    if not address:
        raise MemoryError("could not allocate DLPack descriptor")
    capsule = None
    try:
        capsule = _new_capsule(address, _capsule_name, None)
        tensor = torch.from_dlpack(capsule)
    except BaseException:
        if capsule is None or _valid_capsule(capsule, _capsule_name):
            lib.tilelang_vmm_release(address)
        raise
    if dtype in ("uint32", "uint64"):
        tensor = tensor.view(getattr(torch, dtype))
    if owner is not None:
        tensor.untyped_storage()._tilelang_managed_allocation = owner
    return tensor
