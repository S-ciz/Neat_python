import numpy as np
import random

INITIAL_MUTATION_RATE = 0.075

# Genome Class with Speciation and Compatibility
class Genome:
    innovation_counter = 0  # Global innovation number counter
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.connections = []  # Track connection genes (with innovation numbers)
        self.fitness = 0
        self.species_id = None

    def mutate(self):
        # Mutate weights
        for matrix in [self.weights_input_hidden, self.weights_hidden_output]:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if random.random() < INITIAL_MUTATION_RATE:
                        matrix[i][j] += np.random.uniform(-0.5, 0.5)

        # Add new connection or node mutation
        if random.random() < INITIAL_MUTATION_RATE:
            if random.random() < 0.5:
                self.add_connection_mutation()
            else:
                self.add_node_mutation()

    def add_connection_mutation(self):
        # Randomly add a new connection with a unique innovation number
        input_neuron = random.randint(0, self.input_size - 1)
        output_neuron = random.randint(0, self.hidden_size - 1)
        self.connections.append((input_neuron, output_neuron, Genome.innovation_counter))
        Genome.innovation_counter += 1

    def add_node_mutation(self):
        # Randomly split an existing connection by adding a new node
        if self.connections:
            conn = random.choice(self.connections)
            input_neuron, output_neuron, _ = conn
            self.hidden_size += 1
            new_weights = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
            self.weights_hidden_output = np.concatenate((self.weights_hidden_output, new_weights), axis=0)

    def crossover(self, other):
        child = Genome(self.input_size, self.hidden_size, self.output_size)
        child.weights_input_hidden = np.where(np.random.rand(*self.weights_input_hidden.shape) < 0.5,
                                              self.weights_input_hidden, other.weights_input_hidden)
        child.weights_hidden_output = np.where(np.random.rand(*self.weights_hidden_output.shape) < 0.5,
                                               self.weights_hidden_output, other.weights_hidden_output)
        return child
    
  
    def forward(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_input_hidden)
        hidden_layer_activation = np.tanh(hidden_layer)
        output_layer = np.dot(hidden_layer_activation, self.weights_hidden_output)
        return np.tanh(output_layer)

    def compatibility_distance(self, other):
        # Calculate genetic distance (based on innovation numbers)
        matching_genes = sum(1 for conn in self.connections if conn in other.connections)
        excess_genes = len(self.connections) + len(other.connections) - 2 * matching_genes
        return excess_genes
