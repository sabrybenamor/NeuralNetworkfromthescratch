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
    #assert X.shape[0] == Y.shape[0], "Input and target arrays must have the same number of observations."
    #assert weights.shape[0] == X.shape[1], "Weights must match the number of features in X."
    
    # Forward pass
    M1= np.dot(X, weights['W1'])  # First layer linear combination
    N1= M1 + weights['B1']  # First layer linear combination
    first_layer_output = sigmoid(N1)  # Sigmoid activation
    M2 = np.dot(first_layer_output, weights['W2'])  # Second layer linear combination
    N2 = M2 + weights['B2']  # Second layer linear combination
    P = sigmoid(N2)  # Sigmoid activation for output layer

    # Compute the loss (mean squared error)
    loss = np.mean(np.power(P - Y, 2))  # Mean Squared Error
    # Store the output and weights in a dictionary
    forward_output: Dict[str, NDArray] = {}
    forward_output['X'] = X
    forward_output['Y'] = Y
    forward_output['M1'] = M1
    forward_output['N1'] = N1
    forward_output['first_layer_output'] = first_layer_output
    forward_output['M2'] = M2
    forward_output['N2'] = N2
    forward_output['P'] = P
    return forward_output, loss

def loss_gradient_2_layers(forward_output: Dict[str, NDArray], weights: Dict[str, NDArray]) -> Dict[str, NDArray]:
    """
    Computes the gradient of the loss with respect to the weights and biases of a two-layer neural network.
    
    Parameters:
    - forward_output: Output from the forward pass.
    - weights: Weights dictionary containing 'W1', 'B1', 'W2', and 'B2'.
    
    Returns:
    - Gradient of the loss with respect to weights and biases.
    """
    X = forward_output['X']
    Y = forward_output['Y']
    M1 = forward_output['M1']
    N1 = forward_output['N1']
    first_layer_output = forward_output['first_layer_output']
    M2 = forward_output['M2']
    N2 = forward_output['N2']
    P = forward_output['P']

    # Compute gradients
    dL_dP = 2 * (P - Y)   # Gradient of loss w.r.t. predictions
    dP_dN2 = sigmoid(N2) * (1 - sigmoid(N2))  # Derivative of sigmoid for output layer
    dN2_dM2 = np.ones_like(M2)  # Derivative of linear combination w.r.t. output layer
    dN2_dB2 = np.ones_like(weights['B2'])  # Derivative of linear combination w.r.t. bias of output layer
    dM2_dW2 = np.transpose(first_layer_output, (1, 0))  # Derivative of linear combination w.r.t. weights of output layer
    dM2_dO1 = np.transpose(weights['W2'], (1, 0))  # Derivative of linear combination w.r.t. output of first layer
    dO1_dN1 = sigmoid(N1) * (1 - sigmoid(N1))  # Derivative of sigmoid for first layer
    dN1_dM1 = np.ones_like(M1)  # Derivative of linear combination w.r.t. first layer output
    dN1_dB1 = np.ones_like(weights['B1'])  # Derivative of linear combination w.r.t. bias of first layer
    dM1_dW1 = np.transpose(X, (1, 0))  # Derivative of linear combination w.r.t. weights of first layer
    dM1_dX = np.transpose(weights['W1'], (1, 0))  # Derivative of linear combination w.r.t. output of input layer


    # Chain rule to compute gradients
    grad_W2 = np.dot(dM2_dW2, dL_dP * dP_dN2 * dN2_dM2)  # Gradient of loss w.r.t. W2
    grad_B2 = np.sum(dL_dP * dP_dN2 * dN2_dB2, axis=0)  # Gradient of loss w.r.t. B2
    grad_W1 = np.dot(dM1_dW1, (dL_dP * dP_dN2).dot(weights['W2'].T) * dO1_dN1 * dN1_dM1)  # Gradient of loss w.r.t. W1
    grad_B1 = np.sum ((dL_dP * dP_dN2).dot(weights['W2'].T) * dO1_dN1 * dN1_dB1, axis=0)  # Gradient of loss w.r.t. B1

    return {
        'grad_W1 ': grad_W1,
        'grad_B1 ': grad_B1,
        'grad_W2 ': grad_W2,
        'grad_B2 ': grad_B2
    }


