## Lecture 6

1. Activation functions
    - `Sigmoid`: saturate for large abs(x); kill off gradient flow when saturated; output is not zero-centered (otherwise the gradient will continue to be positive or negative, thus you'll take a zig-zag path to get to the optimization); `exp` is computational expensive.
    - `tanh`: zero centered; still has saturation issue. 
    - `ReLU`: doesn't saturate; computationally efficient; more biological plausible; not zero-centered; kill of the gradient when $$x<0$$. 
    - `Leaky ReLU`: slightly slope in negative region, and gradient doesn't die!
    - `PReLU` (Parametric): the slope of left branch is a parameter!
    - `ELU` (Exponential): the left side is $$\alpha (e^x - 1)$$. Negative saturation hence robustness to noise; computationally expensive. `ELU` is between `ReLU` and `Leaky ReLU`. 
    - `Maxout` (Goodfellow): $$\mathrm{max}(w_1x+b_1, w_2x+b_2)$$. Now four parameters. 

    **TLDR**: Use `ReLU`, try leaky/maxout/ELU, try `tanh` but don't expect much, never use `sigmnoid`. 

2. Data pre-processing
    - Zero-mean and normalize. In CV, we don't normalize images. We usually subtract the mean image.
    - PCA, Whitening. But for images, we only do zero-mean. 

3. Initialize parameters
    - If $$W1,\dots W_n$$ is randomly assigned with amplitude 0.01 and `tanh`: standard devitation of output shrinks rapidly, because the output has been multiplied with small numbers $$W$$ many times. Then we get all the output to zero, which will not activate any neurons!!! The gradient of W will also be very small!! 
    - **upstreaming** is backprop!
    - If $$W1,\dots W_n$$ is randomly assigned with large amplitude and `tanh`: every activation will be saturated! 
    - **Xavier initialization** (can be implemented in PyTorch): `W = np.random.randn(in, out) / np.sqrt(in)`. But `ReLU` will kill this initialization scheme! Then we use `W = np.random.randn(in, out) / np.sqrt(in/2)` (half neuron got killed).
    - Batch normalization (BN): normalizing every image within one batch. The input X has shape (N, D), we calculate `mean` and `std` along N, and normalize each column (we have D columns) accroding to the mean and std. BN is usually inserted before activation function.

4. Babysitting the training
    - Pre-processing the data
    - Choose an architechture
    - Make sure the loss is reasonable. By adding regularization, we expect to see loss goes up.
    - Start up with very small development set, turn off regularization and use vanilla `SGD`, to make sure you overfit this training data, get high score. 
    - Start training! Start with small regularization and find learning rate taht makes the loss go down. `nan` and `inf` indicate loss is too large! Rough range: `[1e-5, 1e-3]`. 
    - Hyperparameters: including learning rate, decay rate, regularization scheme, network architecture...   coarse -> fine cross validation. First do only a few epochs to get the sense of the params, then do longer runs. It's better to search hyperparameters in log space! Randomly sample hyperparameters (don't do grid search)! 
    
