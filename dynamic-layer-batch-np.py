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

weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = (
    np.dot(inputs, np.array(weights).T) + biases
)

layer2_outputs = np.dot(layer1_outputs, np.array(weights).T) + biases

print(layer2_outputs)
