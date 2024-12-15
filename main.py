class Optimizer:
    def __init__(self, learning_rate=0.01):
        pass

    # tells us what direction to go
    def compute_loss(self, node):
        pass

    def take_gradient_step(self, node):
        pass

# Computation Graph Node
class Node:

    def __init__(self, data, op=None, input_nodes=None, gradient=0):
        pass

    def __str__(self):
        pass

    # instantiates back prop
    def compute_gradient(self):
        pass

    # reset the gradient
    def zero_gradient(self):
        pass
    # recursively calls backward
    def autograd(self, grad=1):
        pass

# regular node but with optimizer
class LossNode(Node):

    optimizer = Optimizer()

    def __init__(self, data, op=None, input_nodes=None, gradient=0):
        super().__init__(data, op, input_nodes, gradient)


# interface for forward and backward
class Operation:
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError("Implement Forward Function")

    def backward(self, grad):
        raise NotImplementedError("Implement Backward Function")


class Add(Operation):
    def __init__(self, *nodes):
        pass

    def forward(self):
        pass

    def backward(self, grad):
        pass


class MSELoss(Operation):
    def __init__(self, prediction_node, ground_truth):
        pass

    def forward(self):
        pass

    def backward(self, grad=1):
        pass


