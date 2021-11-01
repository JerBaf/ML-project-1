import numpy as np
import matplotlib.pyplot as plt

### LOSSES

def compute_mse(y,tx,w):
    """ Compute mean square error."""
    error = y - tx@w
    return error.T@error/(2*error.shape[0])

def compute_log_loss(y, tx, w):
    """ Compute the loss for logistic regression."""
    pred = sigmoid(tx@w)
    loss = y.T@(np.log(pred)) + (1 - y).T@(np.log(1 - pred))
    return -np.squeeze(loss)

### MODELS 

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters = 1000, gamma = 0.01):
    """ Fit a logistic regression model with gradient descent and regularization."""
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_gradient_reg_lr(y, tx, w, lambda_, gamma)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters = 1000, gamma = 0.01):
    """ Fit a logistic regression model with gradient descent."""
    w = initial_w
    for iter in range(max_iters):
        loss, w = learning_by_gradient_lr(y, tx, w, gamma)
    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    best_w = w
    best_loss = np.inf
    for iter in range(max_iters):
        gradient, error = compute_gradient_ls(y,tx,w)
        loss = compute_mse(y,tx,w)
        w = w - gamma*gradient
        if loss < best_loss:
            best_w = w
            best_loss = loss
    return best_w, best_loss

def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 1
    w = initial_w
    best_w = w
    best_loss = np.inf
    for i in range(max_iters):
        for y_minibatch, tx_minibatch in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient_ls(y_minibatch,tx_minibatch,w)
        loss = compute_mse(y,tx,w)
        w = w - gamma*gradient
        if loss < best_loss:
            best_w = w
            best_loss = loss
    return best_w, best_loss

def ridge_regression(y, tx, lambda_):
    """Fit a ridge regression model."""
    w = np.linalg.solve(tx.T@tx + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1]), tx.T@y)
    loss = compute_mse(y,tx,w)
    return w, loss

def least_squares(y, tx):
    """Fit a least squares regression model"""
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    loss = compute_mse(y,tx,w)
    return w, loss


### GRADIENTS FUNCTIONS

def compute_gradient_ls(y, tx, w):
    """Compute the gradient of the MSE for linear regression."""
    e = y - tx@w
    N = y.shape[0]
    return -1/N * (tx.T @ e), e

def compute_stoch_gradient_ls(y, tx, w):
    """Compute the gradient of the MSE for linear regression on a small subset."""
    e = y - tx@w
    N = y.shape[0]
    return -1/N * (tx.T @ e)

def calculate_gradient_lr(y, tx, w):
    """Calculate the gradient of the log likelihood"""
    pred = sigmoid(tx@w)
    grad = tx.T@(pred - y)
    return grad

def learning_by_gradient_lr(y, tx, w, gamma):
    """Compute a single iteration of GD for logistic regression."""
    loss = compute_log_loss(y,tx,w)
    grad = calculate_gradient_lr(y,tx,w)
    new_w = w - gamma * grad
    return loss, new_w

def learning_by_gradient_reg_lr(y,tx,w,lambda_,gamma):
    """ Compute a single iteration of GD for penalized logistic regression."""
    loss = compute_log_loss(y,tx,w) + lambda_*np.squeeze(w.T@w)
    grad = calculate_gradient_lr(y,tx,w) + 2*lambda_*w
    new_w = w - gamma * grad
    return loss, new_w

### MISC 

def sigmoid(x):
    """ Compute the sigmoid function on x."""
    return np.exp(x)/(1+np.exp(x))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    METHOD TAKEN FROM THE COURSE CS-433 of EPFL
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]