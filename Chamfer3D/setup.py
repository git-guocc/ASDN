from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ],
        extra_compile_args={
            'cxx': ['-g'],
            'nvcc': ['-O2', '-arch=sm_75']  # 为2080 Ti指定正确的架构
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })