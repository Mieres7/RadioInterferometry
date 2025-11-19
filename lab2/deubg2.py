from numba import cuda
import numpy as np

@cuda.jit
def test_kernel(a):
    i = cuda.grid(1)
    if i < a.size:
        a[i] += 1

x = np.arange(10, dtype=np.float32)
d = cuda.to_device(x)

test_kernel[1, 32](d)
cuda.synchronize()

print(d.copy_to_host())
