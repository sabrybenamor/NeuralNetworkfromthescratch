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