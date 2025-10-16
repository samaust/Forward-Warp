from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='forward_warp',
    packages=["forward_warp"],
    ext_modules=[
        CUDAExtension(
            name='forward_warp_cuda.forward_warp_cuda',
            sources=[
                'forward_warp_cuda/forward_warp_cuda.cpp',
                'forward_warp_cuda/forward_warp_cuda_kernel.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False),
    },
)
