# Autograd

# Creating an Autograd Engine (Implementation)

## System Simplifying

Structuring your code at a high level can be intimidating but we can break it down into its simplest anatomical parts. At the highest level we planning to do -> make a guess,  check that guess, change our guess.

Sharpening things a little, we then think of things in the language of neural networks. The guess is encoded in the forward pass of our algorithm. When we make our check, we compare against a loss function. We correct our guess through back propagation and gradient descent.

We can encode things through classes of Nodes, Operations (loosely analogous to edges in a canonical neural network depiction), and Optimizer to store and update gradients, propagate changes through back propagation, and define how we go about taking our gradient step respectively.

Something to note is there are likely different schools of thought when it comes to defining such a system each with their pros and cons. We can think of things as a balancing act where we are generally working to have thought things through enough such that ideas are "well thought out" but not spending too much time causing stunted progress and falling into the trap of premature optimization.
## Autograd

### Back Propagation

We first notice that the definition at each time step of our algorithm doesn't change. In other words, the problem remains the same for any node within our graph. This hints to us we can use recursion. For the $nth$ node along our path we recurse from the $n-1th$, further we can store the gradient calculated along our path and proceed in a bottom up dynamic programming fashion across paths, across nodes. 

#### Why Zero out our stored gradient?

We don't want information from an earlier iteration of our algorithm to affect how we compute the gradient for the current one.

#### Why recalculate our gradient at all?

After taking our gradient step, we can think of ourselves on a different part of our loss landscape. We need to find out the best way to make another step. In the code, this is shown through having a new loss value, which in turn affects our accumulated gradients, affecting the gradient step.
## **Class-Based Design and Its Role in Autograd**

In designing this autograd engine, we structured our code using object-oriented programming principles. Each class in the system has a clear role, mirroring fundamental neural network concepts. This allowed us to map our mental model of a neural network more directly into code and organize our thoughts. This also improved readability of code and allows us extend our implementation to more operations and optimization techniques.

# Creating an Autograd Engine (Implementation)

## System Simplifying

Structuring your code at a high level can be intimidating but we can break it down into its simplest anatomical parts. At the highest level we planning to do -> make a guess,  check that guess, change our guess.

Sharpening things a little, we then think of things in the language of neural networks. The guess is encoded in the forward pass of our algorithm. When we make our check, we compare against a loss function. We correct our guess through back propagation and gradient descent.

We can encode things through classes of Nodes, Operations (loosely analogous to edges in a canonical neural network depiction), and Optimizer to store and update gradients, propagate changes through back propagation, and define how we go about taking our gradient step respectively.

Something to note is there are likely different schools of thought when it comes to defining such a system each with their pros and cons. We can think of things as a balancing act where we are generally working to have thought things through enough such that ideas are "well thought out" but not spending too much time causing stunted progress and falling into the trap of premature optimization.
## Autograd

### Back Propagation

We first notice that the definition at each time step of our algorithm doesn't change. In other words, the problem remains the same for any node within our graph. This hints to us we can use recursion. For the $nth$ node along our path we recurse from the $n-1th$, further we can store the gradient calculated along our path and proceed in a bottom up dynamic programming fashion across paths, across nodes. 

#### Why Zero out our stored gradient?

We don't want information from an earlier iteration of our algorithm to affect how we compute the gradient for the current one.

#### Why recalculate our gradient at all?

After taking our gradient step, we can think of ourselves on a different part of our loss landscape. We need to find out the best way to make another step. In the code, this is shown through having a new loss value, which in turn affects our accumulated gradients, affecting the gradient step.
## **Class-Based Design and Its Role in Autograd**

In designing this autograd engine, we structured our code using object-oriented programming principles. Each class in the system has a clear role, mirroring fundamental neural network concepts. This allowed us to map our mental model of a neural network more directly into code and organize our thoughts. This also improved readability of code and allows us extend our implementation to more operations and optimization techniques.

### References