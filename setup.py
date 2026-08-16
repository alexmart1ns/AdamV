import os
import sys
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Cross-platform flags
if sys.platform == 'win32':
    cxx_args = ['/O2', '/fp:fast', '/openmp', '/std:c++17']
else:
    cxx_args = ['-O3', '-ffast-math', '-fopenmp', '-std=c++17']

nvcc_args = ['-O3', '-use_fast_math']

ext_modules = [
    # CPU Extension (Always compiled)
    CppExtension(
        name='adamv_cpp',
        sources=['adamv/csrc/adamv_kernel.cpp'],
        extra_compile_args=cxx_args
    )
]

# CUDA Extension (Compiled only if NVCC is available)
if torch.cuda.is_available() or os.environ.get('FORCE_CUDA', '0') == '1':
    # Avoid compilation errors on systems without CUDA compiler installed natively
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is not None:
            ext_modules.append(
                CUDAExtension(
                    name='adamv_cuda',
                    sources=['adamv/csrc/adamv_kernel.cpp', 'adamv/csrc/adamv_kernel.cu'],
                    define_macros=[('WITH_CUDA', None)],
                    extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
                )
            )
            print("CUDA Environment detected! Compiling 'adamv_cuda' extension...")
    except ImportError:
        pass

setup(
    name='adamv',
    version='2.0.3',
    packages=['adamv'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
