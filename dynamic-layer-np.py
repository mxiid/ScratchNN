"""
Modeling a layer dynamically using NumPy
"""

import numpy as np

inputs = [1.2, 5.1, 2.1, 2.5]

weights = [[3.1, 2.1, 8.7, 1.0], [0.5, -0.91, 0.2, -0.5], [-0.26, -2.1, 0.17, 0.87]]

biases = [2, 3, 0.5]

outputs = []

# lmao, i thought i did something when i wrote this code myself as an iteration of the previous dynamic implementation->

for w, b in zip(weights, biases):
    n_output = 0
    for i, n_w in zip(inputs, w):
        n_output += np.dot(i, n_w)
    n_output += b 
    outputs.append(n_output)

print(outputs)

# ->turns out, you could just replace all of that with this xd but it's okay, we learn! honestly i was just confused if you could add a list with the result of a np.dot product but my dumbass didn't realize that np.dot returns a list and list can be added to a list

output = (
    np.dot(weights, inputs) + biases
)  # also, the reason why we're passing weights first in np.dot (even though, the order doesn't matter) is because the shape of the output depends on the first parameter that you're passing and since `weights` is a matrix with the shape of `(3, 4)` we're passing the weights first as the output or the number of neurons is 3 while the number of inputs is 4. also rule of multiplying matrices is that the number of columns of matrix `A` should match the number of rows of matrix `B`. 

print(output)
