
<div align="center">
    <h1 align="center"><i>The Art of Code</i></h1>
    <div align="center">
        <a align="center" href='https://jeremycavallo.com/blog'><img src="https://img.shields.io/badge/Blog-white?logo=ghostty&logoColor=blue" alt="blog"></a>
        <a align="center" href='https://github.com/jjcavallo5/the-art-of-code'><img src="https://img.shields.io/github/stars/jjcavallo5/the-art-of-code" alt="stars"></a>
    </div>
    <p align="center"><i>No dependencies. Just raw code.</i></h1>
</div>

<div align="center">
    <img src="media/header.jpg" alt="header" width="100%">
</div>

<br>

**Inspired by Andrej Karpathy's [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)**

## Philosophy

In the era of AI, code is becoming an art. Developers are shipping faster than ever, yet writing less code themselves.

It's easy to slip away from our roots. We started writing code because we wanted to build something. Along the way, there were lots of struggles. Late nights scrolling through StackOverflow forums, trying to find the solution to some niche bug. 

Why did we endure this pain?

_Because eventually, we would solve the problem_

That's what's exciting. That's whats rewarding. That's what brings us back. Letting the agent solve all of our problems? Yes, it's much more efficient. But that rewarding feeling is gone. And because of that, developers are burning out and forgetting their roots.

Sometimes, we need to go back to our roots, and:

**Just. Write. Code.**

## About The Code

The code trains and evaluates a simple neural network. It utilizes the MNIST hand-written digit dataset, which consists of 60,000 training samples and 10,000 test samples.

There are no dependencies to this code, which means everything has to be built from scratch, including:

- Random number generation for weight initialization
- Downloading the dataset
- Reading samples and labels from the downloaded dataset
- Autograd engine, including gradient tracking
- Common machine learning components, such as:
  - Cross-entropy loss function
  - Stochastic gradient descent optimizer
  - Linear layers
  - Softmax
- Training and evaluation loops

Training with the default hyper parameters results in a test accuracy of ~96%. I didn't spend much time tuning them, so there are likely performance gains to be found.

## Blog Post

I plan to write a detailed blog post explaining the code. When that's complete, I'll link to it here. For now, check out my [other blog posts](https://jeremycavallo.com/blog)
