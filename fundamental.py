import numpy as np
from numpy.typing import NDArray  #optinal for type hints (type signature)
from typing import Callable


def square(x:np.ndarray)-> np.ndarray:
  return np.power(x,2)

def deriv(f: Callable[[NDArray], NDArray], x: NDArray, h: float = 1e-5) -> NDArray:
    """
    Computes the numerical derivative of a function f at point x using central difference method.
    
    Parameters:
    - f: The function to differentiate.
    - x: The point at which to evaluate the derivative.
    - h: A small step size for the numerical approximation.
    
    Returns:
    - The numerical derivative of f at x.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid function for each element in the input array.
    
    Parameters:
    - x: Input array.
    
    Returns:
    - Sigmoid of each element in x.
    """
    return 1 / (1 + np.exp(-x))

function = Callable[[NDArray], NDArray] #here i define the function which takes an NDArray and returns an NDArray
Chain = list[function]  # A chain of functions, each taking an NDArray and returning an NDArray

def chain_rule_2(chain:Chain,x : np.ndarray) -> NDArray:
  assert len(chain) == 2, "Chain must contain exactly two functions."
  assert x.ndim == 1, "Input x must be a one-dimensional array."
  
  first_func = chain[0]
  second_func = chain[1]
  # Compute the derivative of the first function at x
  first_derivative = deriv(first_func, x)
  # Compute the output of the first function at x
  first_output = first_func(x)
  # Compute the derivative of the second function at the output of the first function
  second_derivative = deriv(second_func, first_output)

  # Apply the chain rule: derivative of second function at first output times derivative of first function
  return second_derivative * first_derivative

def matrice_multiplication_forward(x: NDArray, W: NDArray) -> NDArray:
    """
    Computes the forward pass of matrix multiplication.
    
    Parameters:
    - x: Input array (1D or 2D).
    - W: Weight matrix (2D).
    
    Returns:
    - Result of multiplying x with W.
    """
    assert x.shape[0] == W.shape[1], "Input x must have the same number of elements as W has rows."
    return np.dot(x, W)

def martrice_multiplication_backward(x: NDArray, W: NDArray) -> NDArray:
    """
    Computes the backward pass of matrix multiplication.
    
    Parameters:
    - x: Input array (1D or 2D).
    - W: Weight matrix (2D).
    
    Returns:
    - Gradient of the loss with respect to x.
    """
    assert x.shape[0] == W.shape[1], "Input x must have the same number of elements as W has rows."
    dxdn= np.transpose(W,(1,0))
    return dxdn



