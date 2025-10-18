## Foward Warp Pytorch Version

Has been tested in torch=2.8.0+cu126, python=3.12, CUDA=12.6

### Install

```bash
export CUDA_HOME=/usr/local/cuda #use your CUDA instead
pip install forward_warp@git+https://github.com/samaust/Forward-Warp@python_package
```

### Test

```bash
cd test
python test.py
```

### Usage

```python
from forward_warp import Forward_warp

fw = Forward_warp()
# default interpolation mode is Bilinear
im2_bilinear = fw(im0, flow) 
# use interpolation mode Nearest
# Notice: Nearest input-flow's gradient will be zero when at backward.
im2_nearest = fw(im0, flow, interpolation_mode="Nearest") 
```
