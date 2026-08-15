# Forward Warp for PyTorch

A CUDA extension for forward-warping images with PyTorch and autograd support.

## Verified environment

This project is currently tested with:

- Python 3.14.6
- PyTorch 2.13.0+cu130
- CUDA Toolkit 13.0 (`nvcc` 13.0)
- NVIDIA GeForce RTX 4090 (compute capability 8.9)

Earlier Python, PyTorch, and CUDA versions are not tested or documented as supported.

## Requirements

Building this package compiles a CUDA extension. You need:

- An NVIDIA GPU and a compatible driver
- CUDA Toolkit 13.0, including `nvcc`
- A C++ compiler supported by the installed CUDA Toolkit
- Python 3.14
- PyTorch 2.13.0+cu130

Install PyTorch first, using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your platform and CUDA build. The CUDA Toolkit used to compile this extension must be compatible with that PyTorch build.

The build normally finds the CUDA Toolkit automatically. If it is installed in a nonstandard location, set `CUDA_HOME` before installing or building:

```bash
export CUDA_HOME=/usr/local/cuda
```

Check the toolchain before continuing:

```bash
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Install from source

Create and activate a virtual environment, install PyTorch, then install the build dependencies required by `pyproject.toml`:

```bash
python -m pip install --upgrade pip
# Install PyTorch 2.13.0+cu130 using the command from pytorch.org.
python -m pip install "setuptools>=61" "setuptools-scm[simple]>=8" numpy wheel build
```

Clone the package branch and install it. `--no-build-isolation` makes the CUDA extension compile against the PyTorch already installed in the environment:

```bash
git clone --branch python_package https://github.com/samaust/Forward-Warp.git
cd Forward-Warp
python -m pip install --no-build-isolation .
```

To install directly from Git instead:

```bash
python -m pip install --no-build-isolation \
  "forward_warp @ git+https://github.com/samaust/Forward-Warp.git@python_package"
```

Verify the installation:

```bash
python - <<'PY'
import torch
import forward_warp
from forward_warp import _cuda

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("forward_warp:", forward_warp.__file__)
print("extension:", _cuda.__file__)
PY
```

## CUDA architectures

When `TORCH_CUDA_ARCH_LIST` is unset, PyTorch builds the extension for the CUDA architectures of GPUs visible during compilation. On the verified RTX 4090 host, this produces native `sm_89` code and `compute_89` PTX.

This default is appropriate when the wheel will run only on the build machine or identical GPUs. To create a wheel for several GPU architectures, set `TORCH_CUDA_ARCH_LIST` before building. For example:

```bash
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"
```

This emits native code for compute capabilities 8.6, 8.9, and 9.0, plus PTX for 9.0. More architectures increase build time and wheel size. Choose only the architectures needed by the deployment fleet.

## Build a wheel

From the repository root, build a wheel for the visible GPU architecture:

```bash
unset TORCH_CUDA_ARCH_LIST
python -m build --wheel --no-isolation
```

For a multi-architecture wheel, specify the architectures for the build command:

```bash
TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" \
  python -m build --wheel --no-isolation
```

The wheel is written to `dist/` and can be installed with:

```bash
python -m pip install dist/forward_warp-*.whl
```

## Usage

`Forward_warp` accepts an image tensor with shape `[B, C, H, W]` and an optical-flow tensor with shape `[B, H, W, 2]`. Flow values are measured in pixels.

```python
from forward_warp import Forward_warp, Forward_warp_rescaled

# Bilinear interpolation is the default.
forward_warp = Forward_warp()
warped = forward_warp(image, flow)

# Select nearest-neighbor interpolation when constructing the module.
nearest_forward_warp = Forward_warp(interpolation_mode="Nearest")
nearest_warped = nearest_forward_warp(image, flow)

# Produce a warp rescaled by a forward-warped visibility mask.
rescaled_warp = Forward_warp_rescaled()(image, flow)
```

`Forward_warp_max_motion` is also available for CUDA tensors and provides forward warping with maximum-motion occlusion handling.

## Tests

The image demo uses the bundled `im0.png`, `im1.png`, and `flow.pkl` files, and requires OpenCV:

```bash
python -m pip install opencv-python
cd test
python test.py
```

The pytest suite requires a CUDA-capable environment:

```bash
python -m pip install pytest
pytest test/test_max_motion.py
```

## Troubleshooting

- **`nvcc` not found:** install CUDA Toolkit 13.0 and set `CUDA_HOME` if the installer cannot find it.
- **CUDA/PyTorch version mismatch:** install the intended CUDA-enabled PyTorch build before installing this project, then use `--no-build-isolation`.
- **Unsupported GPU architecture:** build again with `TORCH_CUDA_ARCH_LIST` containing the target GPU capability.
- **Compilation fails:** confirm that the system C++ compiler is supported by CUDA 13.0 and that `nvcc --version` reports the expected toolkit.

## License

This project is distributed under the [MIT License](LICENSE).
