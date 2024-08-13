from splatmul import matmat, splatsplat, splatmat
from ml_dtypes import bfloat16
import numpy as np
# a = np.linspace(-1000, 1000, 1024).astype(bfloat16)
# print(a)
# print(a.view(np.uint16).view(bfloat16))
# print(a.view(np.uint16).view(bfloat16).view(np.uint16).view(bfloat16))
# b = a + np.linspace(0, 1000, 1024).astype(bfloat16)[:, None]
# print(b)
# print(b.view(np.uint16).view(bfloat16).max())

w_size = 1024
hidden_size = 1024
weight_enc, weight_dec = matmat(w_size, hidden_size)
print(weight_dec.dtype)
weight_dec = weight_dec.view(bfloat16).copy()
weight_dec = np.ascontiguousarray(weight_dec)
adam_state = splatsplat(w_size, hidden_size,
                        1e-4, 0.9, 0.999, 1e-8, 16)
grad = weight_dec * 1e-3
grad = grad.copy()
grad = np.ascontiguousarray(grad)
print(grad.dtype)
print("Max grad:", np.max(grad))
# print(grad.view(np.uint16).shape)
print("Max grad:", np.max(grad.view(np.uint16).view(bfloat16)))
# 1/0
# splatmat(adam_state, grad.view(np.uint16), weight_dec.view(np.uint16))