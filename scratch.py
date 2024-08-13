from splatmul import matmat, splatsplat, splatmat
from ml_dtypes import bfloat16
import numpy as np

w_size = 1024
hidden_size = 1024
weight_enc, weight_dec = matmat(w_size, hidden_size)
weight_dec = weight_dec.view(bfloat16).copy()
weight_dec = np.ascontiguousarray(weight_dec)
adam_state = splatsplat(w_size, hidden_size,
                        1e-4, 0.9, 0.999, 1e-8, 16)
grad = weight_dec * np.array(1e-3, dtype=bfloat16)
print(np.abs(grad).max())
splatmat(adam_state, grad.view(np.uint16), weight_dec.view(np.uint16))
print(np.abs(weight_dec).max())