"""
Modeling a layer batch dynamically using NumPy
"""

import numpy as np

inputs = [
            [1.2, 5.1, 2.1, 2.5],
            [2.0, 5.0,-1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]

weights = [
            [3.1, 2.1, 8.7, 1.0], 
            [0.5, -0.91, 0.2, -0.5], 
            [-0.26, -2.1, 0.17, 0.87]
        ]

biases = [2, 3, 0.5]

output = (
    np.dot(inputs, np.array(weights).T) + biases
)  # *transposing since `inputs` is `shape(3, 4)` and `weights` is `shape(3,4)` as well and when we transpose `weights`, it becomes `shape(4, 3)` hence the columns of `inputs` match the rows of `weights` and a dot product is computed.

print(output)
