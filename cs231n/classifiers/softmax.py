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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    ex_sum = np.sum(np.exp(scores))
    ex_truth = np.exp(scores[y[i]])
    correct_ratio = ex_truth / ex_sum
    this_loss = - np.log(correct_ratio) 
    for j in xrange(num_classes):
      if j == y[i]:
        dW.transpose()[j] += (correct_ratio * X[i]) - X[i]
      else:
        dW.transpose()[j] += (np.exp(scores[j]) / ex_sum) * X[i]
    loss += this_loss
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += reg * np.sum(W * W)
  loss /= X.shape[0]
  dW += (reg * W * 2)
  dW /= X.shape[0]

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
  scores = X.dot(W)
  correct_scores = scores[np.arange(len(y)),(y)]
  ex_sums = np.sum(np.exp(scores), axis=1)
  ex_truth = np.exp(correct_scores)
  corr_ratios = ex_truth / ex_sums
  loss = np.sum(- np.log(corr_ratios))
  # fill dW with the coefficients for the data
  grad_mat.transpose()[np.arange(len(y)),(y)] = - truth_sums
  dW = grad_mat.dot(X).T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += reg * np.sum(W * W)
  loss /= X.shape[0]
  dW += (reg * W * 2)
  dW /= X.shape[0]

  return loss, dW
