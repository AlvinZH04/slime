/work/nvme/bgqz/bzhang31/envs/slime/lib/python3.12/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/work/nvme/bgqz/bzhang31/envs/slime/lib/python3.12/site-packages/torch/cuda/__init__.py:827: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
W0517 11:55:03.735000 3525680 torch/utils/cpp_extension.py:117] No CUDA runtime is found, using CUDA_HOME='/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9'
/work/nvme/bgqz/bzhang31/envs/slime/lib/python3.12/site-packages/fla/utils.py:356: UserWarning: Triton is not supported on current platform, roll back to CPU.
  warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
    PyTorch 2.11.0+cu130 with CUDA 1301 (you have 2.9.1+cu129)
    Python  3.12.9 (you have 3.12.9)
  Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
  Memory-efficient attention, SwiGLU, sparse and more won't be available.
  Set XFORMERS_MORE_DETAILS=1 for more details
# Environment snapshot

Python: 3.12.9
torch: 2.9.1+cu129
torch.cuda.is_available: False
transformers: 5.8.1
flash_attn: 2.7.4.post1
flashinfer: 0.6.3
sglang: 0.5.11
sgl_kernel: 0.3.21
transformer_engine: 2.10.0
apex: ?
ray: 2.55.1
mbridge: 0.15.1

slime HEAD: 41dc3b6d21d3c75b212965077a1cc4117932f06d
Megatron-LM HEAD: 3714d81d418c9f1bca4594fc35f9e8289f652862
sglang HEAD: bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2
