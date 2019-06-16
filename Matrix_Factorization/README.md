## 代码来自下面的网址
http://www.albertauyeung.com/post/python-matrix-factorization/

## 使用
打开ipython
``` python
import numpy as np
from MF import MF
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
mf = MF(R, K=2, alpha=0.1, beta=0.01, iterations=20)
training_process = mf.train()
print("P x Q:", mf.full_matrix())
print("Global bias:", mf.b)
print("User bias:", mf.b_u)
print("Item bias:", mf.b_i)
```
 
