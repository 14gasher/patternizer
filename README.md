# patternizer
____________

## Introduction


Hey y'all, so this is my repo for learning about machine learning. Everything is written in c++ using only the standard
library. I know, I know, there are fast libraries out there, but the point of this exercise was not to build something
as fast as possible, but rather to learn, and what better way than to do it from scratch? With that in mind, there are
additional classes associated with each of the machine learning types for support.

With that jargon out of the way, here is a current list of machine learning things I have built.

1. <a href="#perceptron">Perceptron</a>
2. <a href="#simpleNN">Simple Neural Network</a> (Basic cost function, 1 hidden layer, prebuilt to read MINST data).
3. <a href="#future"> Future </a>

As I continue teaching myself, this list will grow, so enjoy!


__________
## <a id="perceptron">Perceptron</a>


Being the among the oldest flavors of machine learning, I've implemented one of my one.
It works like you'd expect. Train it by giving an input and a target output, and it will adjust its
weight until it gets where it should be.

In main.cpp, there is a function demonstrating how the class works called perceptronDemo.


_____________________

## <a id="simpleNN">Simple Neural Network</a>

Moving up the chain, I've implemented a simple neural network. It uses the Quadratic Cost Function ((target - output)^2)/2,
allows you to define a learning rate, and to define the number of inputs, hidden layer neurons, and outputs. It's pretty slick...
I've managed to get over 90% accuracy on the MINST set within 30 training epochs.

In order to make this work, I've created a Matrix library. Unfortunately, it does not use the hardware to speed things up (GPU anyone?),
but that may come with time. Also, there is a class to convert the MINST data from big to little endian and store it.

In main.cpp, you will find main.cpp implementing, training, and testing a SimpleNeuralNetwork instance.


______
## <a id="future">Future</a>


The next step will be making the "Simple" Neural Network more complex. This includes adding regularization techniques,
having the option of different cost functions including the cross-entropy cost function, and adding momentum to the gradient
descent.

Once these additional things are added, I believe I will be home free to continue onwards to a "deep" Network.

Other areas to be pursued:

**Genetic algorithms** -- Finds probable optimizations on weird data sets
                   (Most value if can take 1 pound, 500 different things with different weights and values...)

**Reinforcement Learning** -- Agents learn in environment meaning you don't need initial training data

**Generative Learning** -- The ability to "copy". Given a bunch of hand written 5's, output your own 5.

**Networks with Memory** -- Traditional NN "forgets" previous learning when switched to a new task. Can you stop that?
                     This includes: long-short term memory networks, elastic weight consolidation, and progressive neural networks

**Less Data, Smaller Models** -- Can simpler NN's learn from complex ones to reduce the training time? Transfer training...

**Environment Creation** -- We can't really just let a robot ai run wild... create a virtual environment and have ai learn there...





