"""Make the pip-installed CUDA 12 libraries visible to both runtimes."""

import ctypes
import glob
import os
import site

_LIBS = (
    "nvidia/cublas/lib/libcublas.so.12",
    "nvidia/cublas/lib/libcublasLt.so.12",
    "nvidia/cudnn/lib/libcudnn.so.9",
)


def preload() -> None:
    """Load cuBLAS/cuDNN globally so CTranslate2 finds them, then let onnxruntime do the same.

    Neither runtime looks inside site-packages by itself; without this both fall back to the CPU
    or fail to load.
    """
    for pattern in _LIBS:
        for root in site.getsitepackages():
            for path in glob.glob(os.path.join(root, pattern)):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    import onnxruntime

    onnxruntime.preload_dlls()
