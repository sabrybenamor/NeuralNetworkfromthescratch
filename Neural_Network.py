import numpy as np
from numpy.typing import NDArray  #optinal for type hints (type signature)
from typing import Tuple, Dict
from fundamental import sigmoid

# Two layers neural network
def forward_loss(X: NDArray, Y: NDArray, weights:Dict[str,NDArray]) -> Tuple[Dict[str,NDArray],float]:
    """
    Computes the forward pass of a two-layer neural network and calculates the loss.
    
    Parameters:
    - X: Input array (2D).
    - Y: Target array .
    - weights: Weights array (2D).
    
    Returns:
    - The loss value (float).
    """
    assert X.shape[0] == Y.shape[0], "Input and target arrays must have the same number of observations."
    assert weights.shape[0] == X.shape[1], "Weights must match the number of features in X."
    
    # Forward pass
    N1= np.dot(X, weights['W1']) + weights['B1']  # First layer linear combination
    first_layer_output = sigmoid(N1)  # Sigmoid activation
    N2 = np.dot(first_layer_output, weights['W2']) + weights['B2']  # Second layer linear combination
    P = sigmoid(N2)  # Sigmoid activation for output layer

    # Compute the loss (mean squared error)
    loss = np.mean(np.power(P - Y, 2))  # Mean Squared Error
    # Store the output and weights in a dictionary
    forward_output: Dict[str, NDArray] = {}
    forward_output['X'] = X
    forward_output['Y'] = Y
    forward_output['N1'] = N1
    forward_output['first_layer_output'] = first_layer_output
    forward_output['N2'] = N2
    forward_output['P'] = P
    return forward_output, loss
