# neural-net

This library is my from-scratch neural net. Its purpose is to be educational. Writing it helped me fill in some gaps in my knowledge, and hopefully my code / story will point others in the right direction as well.

## Goal

The goal was to create a library that was capable of classifying images from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The MNIST dataset is a common "hello, world" of image classification, so it's a good place to start.

## The Neural Net

Classifying the images from our dataset can be done with a relatively small, easily understood network. We're just going to use two densly connected layers. In Keras, the network can be defined as...

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

The input is a 28x28 matrix that'll get flatted into a 784 element vector. The first dense network will consist of 128 neurons, each connected to every element in the input vector. The second dense layer will be the head of the network. It has outputs 10 elements, one for each class present in the dataset.

The math behind this is very simple. The flatten layer has no math at all. The dense layers each output `activation(weights * input + bias)`. The RELU activation function is just `max(0, input)`. It introduces non-linearity to the network, which is important for image recognition. The softmax function is just `exp(input)/sum(exp(input))`. It normalizes our outputs into a probability vector.

If you have optimized weights and biases, generating the predictions for an input image is basic math. The entire trick to machine learning is in figuring out the right weights and biases. And in this network, there are 101,632 weights and 138 biases to optimize.

See the [Tensorflow equivalent here](examples/fashion-mnist/tensorflow-equivalent.py) if you want to confirm that this network design does actually work and see what the output looks like.

## Stochastic Gradient Descent

[Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is one of the basic techniques used to arrive at good weights and biases for the network.

In a nutshell, stochastic gradient descent goes like this:

1. Pick a random training image.
2. Calculate the "gradient" of the loss function with respect to each trainable parameter.
3. Adjust the trainable parameter based on that gradient and a learning rate.
4. Repeat with the next image.

The loss function is a metric for how close the network is to the correct prediction. For this network, we'll use categorical cross-entropy. When images map to just one category, the math for that is just `-ln(prediction for that category)`. So if the prediction for the correct category was 1.0, the loss would be 0.0.

The gradient is the derivative of the loss function. It tells us which direction to adjust each weight in to make the loss decrease.

## Calculus

Okay so now on each step, we need derivatives for all 101,770 of our trainable parameters, and we need them *fast*. If you run the [Tensorflow equivalent](examples/fashion-mnist/tensorflow-equivalent.py), you'll see that it only needs about 30 *micro*seconds for each step. This is where the magic lies.

When I first started writing code, I hadn't yet looked at what Tensorflow was actually doing. Instead, I brushed up on calculus using guides like [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/).

My first working version of the network used all the rules laid out in that document to perform training. The network was literally 100,000 times slower than Tensorflow. Something was wrong.

Turns out Tensorflow doesn't use any of the math I was using. In fact, at the moment it's not even capable of deriving true Jacobian matrices.

## Automatic Differentiation

"Automatic differentiation" was the buzz-word I needed. As soon as I looked at the [Wikipedia page](https://en.wikipedia.org/wiki/Automatic_differentiation), I realized I had been wasting my time with all that matrix calculus.

Specifically, reverse-mode differentiation is the key. This allows us to get the gradients for all of our variables in a single traversal of the graph.

I leaned heavily on Tensorflow's [tf.gradients](https://www.tensorflow.org/api_docs/python/tf/gradients) API to write unit tests for my implementations here, and by grepping for "RegisterGradient" in the Tensorflow codebase, I could find reference for all of the operations I needed.

## The Result

The network successfully trains on the dataset with results similar to the Tensorflow equivalent. The performance is acceptably close. Tensorflow is still faster, but not by multiple orders of magnitude.

You can go to the [examples/fashion-mnist](examples/fashion-mnist) directory and `cargo run --release` to take it for a spin.
