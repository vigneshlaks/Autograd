from typing import Optional, List

import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def take_gradient_step(self, node):
        # handles weight updates for particular node
        if node.op:
            node.op.take_gradient_step(self.learning_rate)

        for input_node in node.input_nodes:
            self.take_gradient_step(input_node)

        # compute new guess
        if node.op:
            node.op.forward()


# Computation Graph Node
class Node:

    def __init__(self, data=None, op: Optional["Operation"] = None, input_nodes=None
                 , gradient=0):
        # ADDED AS DATA
        self.data = data
        self.gradient = gradient

        # For backprop
        self.input_nodes = input_nodes if input_nodes is not None else []
        self.op = op

    def __str__(self):
        return f'Data: {self.data}'

    # reset the gradient
    def zero_gradient(self):
        self.gradient = 0
        for node in self.input_nodes:
            node.zero_gradient()

    # recursively calls backward
    def autograd(self, grad=1):
        # add to global grad
        self.gradient += grad

        if self.op:
            # calculate amount of gradient to send to each node
            new_grads = self.op.backward(grad)

            for node, grad in zip(self.input_nodes, new_grads):
                node.autograd(grad)


# regular node but with optimizer
class LossNode(Node):

    optimizer = Optimizer()

    def __init__(self, data, op=None, input_nodes=None, gradient=0):
        super().__init__(data, op, input_nodes, gradient)

    # instantiates back prop
    def compute_gradient(self):
        # recursively zeroes the gradients
        self.zero_gradient()

        # calculate gradient
        self.autograd()

        # take gradient step
        self.optimizer.take_gradient_step(self)


# interface for forward and backward
class Operation:
    def __init__(self):
        pass

    def get_node(self):
        raise NotImplementedError("Implement Forward Function")

    def backward(self, grad):
        raise NotImplementedError("Implement Backward Function")

    def take_gradient_step(self, learning_rate):
        raise NotImplementedError("Implement Gradient Step")


class Add(Operation):
    def __init__(self, *nodes):
        self.valueList = np.array([node.data for node in nodes])
        self.weights = np.array([1.0] * len(self.valueList))

        # init node
        data = np.dot(self.valueList, self.weights)
        self.node = Node(data, self, nodes)

    def get_node(self):
        return self.node

    def take_gradient_step(self, learning_rate):
        self.weights -= float(self.node.gradient * learning_rate)
    def forward(self):
        self.node.data = np.dot(self.valueList, self.weights)

    def backward(self, grad):
        # scalar multiply our weight vector
        return self.weights * grad


class MSELoss(Operation):
    def __init__(self, prediction_node, ground_truth):
        self.prediction_node = prediction_node
        self.ground_truth = ground_truth
        self.node = None

        loss = (self.prediction_node.data - self.ground_truth) ** 2
        self.node = LossNode(loss, self, [self.prediction_node])

    def take_gradient_step(self, learning_rate):
        self.node.data -= learning_rate * self.node.gradient

    def get_node(self):
        return self.node

    def forward(self):
        loss = (self.prediction_node.data - self.ground_truth) ** 2
        self.node.data = loss

    def backward(self, grad=1):
        grad = 2 * (self.prediction_node.data - self.ground_truth)
        return [grad]


# create prediction node
n2 = Node(2)
n1 = Node(5)
n3 = Add(n1, n2).get_node()
n4 = Node(90)
n5 = Add(n4, n3).get_node()

# create loss node
mse_loss_node = MSELoss(n5, 8).get_node()

# check initial loss
for i in range(100):
    mse_loss_node.compute_gradient()
    print(f"\non iteration {i+1}")
    print(mse_loss_node)
    print("n3 node " + str(n3))
    print(n5)