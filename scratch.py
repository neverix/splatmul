from splatmul import matmat, splatsplat, splatmat
from ml_dtypes import bfloat16
import numpy as np
w_size = 1024
hidden_size = 1024
weight_enc, weight_dec = matmat(w_size, hidden_size)
weight_dec = weight_dec.view(bfloat16)
adam_state = splatsplat(w_size, hidden_size,
                        1e-4, 0.9, 0.999, 1e-8, 16)
grad = weight_dec * 1e-3
print("Max grad:", np.max(grad))
splatmat(adam_state, grad.view(np.uint16), weight_dec.view(np.uint16))