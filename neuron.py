"""
Setting up a simple neuron
"""

inputs = [1.2, 5.1, 2.1] #input from the previous neuron/true input (can be true inputs or outputs from neurons)
weights = [3.1, 2.1, 8.7] #every input has a weight associated with it
bias = 3 #every unique neuron has a unique bias

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias # y=Wx+b

print(output)