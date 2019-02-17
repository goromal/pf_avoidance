# Potential Field Avoidance

Functions for facilitating dynamic obstacle avoidance in a path-following setting.

For API documentation, see `Potential Field Avoidance.pdf`.

# Example Usage

```python3

from pf_avoidance import PotentialField as PF
import autograd.numpy as np

x = np.array([[1.0], [2.0], [1.0]]) # THESE NEED TO BE FLOATS!!!
s = np.array([[1.0], [1.0], [0.0]])

PF.addObstacle(1.0, 2, 3, 0, 0)
PF.addObstacle(0.0, 0, 0, 1, 1)

print(PF.Potential(x))
print(PF.Gradient(x))
print(PF.Hessian(x))
print(PF.directionalDerivative(x, s))
print(PF.secondDirectionalDerivative(x, s))

```

Output:

```
14.0
[[4.]
 [8.]
 [4.]]
[[4. 0. 0.]
 [0. 4. 0.]
 [0. 0. 4.]]
8.48528137423857
14.14213562373095
```
