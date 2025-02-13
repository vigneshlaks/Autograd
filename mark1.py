from typing import Optional, List


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def take_gradient_step(self, node):
        if node.gradient:
            node.data -= self.learning_rate * node.gradient

        for input_node in node.input_nodes:
            self.take_gradient_step(input_node)

        if node.op:
            node.op.forward()


# Computation Graph Node
class Node:

    def __init__(self, data, op: Optional["Operation"] = None, input_nodes=None, gradient=0):
        self.data = data
        self.gradient = gradient

        # For backprop
        self.input_nodes = input_nodes or []
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

    def forward_node(self):
        raise NotImplementedError("Implement Forward Function")

    def backward(self, grad):
        raise NotImplementedError("Implement Backward Function")


class Add(Operation):
    def __init__(self, *nodes):
        self.nodeList = list(nodes)
        self.node = None

    def forward_node(self):
        self.node = Node(sum(node.data for node in self.nodeList), op=self,
                    input_nodes=self.nodeList)
        return self.node

    def forward(self):
        self.node.data = sum(node.data for node in self.nodeList)

    def backward(self, grad):
        return [grad for _ in self.nodeList]


class MSELoss(Operation):
    def __init__(self, prediction_node, ground_truth):
        self.prediction_node = prediction_node
        self.ground_truth = ground_truth
        self.node = None

    def forward_node(self):
        loss = (self.prediction_node.data - self.ground_truth) ** 2
        self.node = LossNode(loss, self, [self.prediction_node])
        return self.node

    def forward(self):
        loss = (self.prediction_node.data - self.ground_truth) ** 2
        self.node.data = loss

    def backward(self, grad=1):
        grad = 2 * (self.prediction_node.data - self.ground_truth)
        return [grad]


# create prediction node
n1 = Node(5)
n2 = Node(-2)
n3 = Add(n1, n2).forward_node()
# create loss node
mse_loss_node = MSELoss(n3, 6).forward_node()

# check initial loss
print(mse_loss_node)
for i in range(1000):
    mse_loss_node.compute_gradient()
    print(f"\non iteration {i+1}")
    print(mse_loss_node)
    print(n1)
    print(n2)
    print(n3)
