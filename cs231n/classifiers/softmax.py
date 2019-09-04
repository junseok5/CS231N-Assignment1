from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        fi = np.dot(X[i], W)
        sum_j = np.sum(np.exp(fi))
        
        for k in range(num_classes):
            pk = np.exp(fi[k]) / sum_j
            
            if y[i] == k:
                dW[:, k] += (pk - 1) * X[i]
            else:
                dW[:, k] += pk * X[i]
            
        loss += -np.log(np.exp(fi[y[i]]) / sum_j)
        
#     loss /= num_train + reg * np.sum(W * W)
#     dW /= num_train + reg * W
        
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # f: (500, 10), sum_f: (500, ), W: (3073, 10), pk: (500, 10)
    num_train = X.shape[0]
    
    f = np.dot(X, W)
    sum_f = np.sum(np.exp(f), axis=1)
    
    loss = -np.log(np.exp(f[np.arange(num_train), y]) / sum_f)
    loss = np.sum(loss) / num_train + reg * np.sum(W * W)
    
    pk = np.exp(f) / sum_f[:, np.newaxis]
    P = np.zeros(pk.shape)
    P = pk
    P[np.arange(num_train), y] -= 1
    
    dW += np.dot(X.T, P)
#     dW /= num_train + reg * W
    dW /= num_train
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
