from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='adamv_cpp',
    ext_modules=[
        CppExtension(
            name='adamv_cpp',
            sources=['csrc/adamv_kernel.cpp'],
            extra_compile_args=['/O2'] # Max optimization flag for MSVC
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
