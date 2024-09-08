"""
Modeling a layer dynamically
"""

inputs = [1.2, 5.1, 2.1, 2.5]

weights = [[3.1, 2.1, 8.7, 1.0], [0.5, -0.91, 0.2, -0.5], [-0.26, -2.1, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = [] #output of the current layer
for neuron_weights, neuron_bias in zip(weights, biases): #basically taking neuron's weights and it's respective bais
    neuron_output = 0 #initializing neuron output
    for n_input, weight in zip(inputs, neuron_weights): #we're taking weights and then taking it's respective input now so that we can multiply it
        neuron_output+=n_input*weight
    neuron_output+=neuron_bias #finally adding the bias at the end
    layer_outputs.append(neuron_output) #appending all outputs (so, if it's 3 outputs, the output will have a list containing 3 elements)

print(layer_outputs)