import numpy as np
from numpy.typing import NDArray  #optinal for type hints (type signature)
from typing import Tuple, Dict

def forward_linear_regression(X: NDArray, Y: NDArray, weights:Dict[str,NDArray]) ->Tuple[float,Dict[str,NDArray]] :
    """
    Computes the forward pass of a linear regression model.
    
    Parameters:
    - x: Input array .
    - w: Weights array .
    - y: targets .
    
    Returns:
    - The output of the linear regression model.
    """
    assert X.shape[0]== Y.shape[0], "Input and target arrays must have the same length (number of observations)."
    assert X.shape [1]== weights['W'].shape[0] ,"assert matrix multiplication is valid"
    assert weights['B'].shape[0]== weights['B'].shape[1]==1, "Bias must be a scalar."

    # Compute the output of the linear regression model
    N=np.dot(X,weights['W']) 
    P=N + weights['B']  # Add bias to the linear combination
    # Compute the loss (mean squared error)
    loss = np.mean(np.power(P - Y, 2))  # Mean Squared Error
    # Store the output and weights in a dictionary
    forward_output : Dict[str, NDArray] = {}
    forward_output['X']=  X
    forward_output['Y']= Y
    forward_output['N'] = N
    forward_output['P'] = P

    return loss, forward_output

def loss_gradient (forward_output: Dict[str, NDArray], weights: Dict[str, NDArray]) -> Dict[str, NDArray]:
    """
    Computes the gradient of the loss with respect to the weights and bias.
    
    Parameters:
    - forward_output: Output from the forward pass.
    - weights: Weights dictionary containing 'w' and 'b'.
    
    Returns:
    - Gradient of the loss with respect to weights and bias.
    """
    X = forward_output['X']
    Y = forward_output['Y']
    N = forward_output['N']
    P = forward_output['P']

    # Compute gradients
    dL_dP = 2 * (P - Y)   # Gradient of loss w.r.t. predictions
    dP_dN = np.ones_like(N)  # Derivative of predictions w.r.t. linear combination
    dN_dw = np.transpose(X,(1,0)) # Derivative of linear combination w.r.t. weights
    dP_db = np.ones_like(weights['B'])  # Derivative of predictions w.r.t. bias
    dL_dN = dL_dP * dP_dN  # Chain rule to combine gradients
    # Chain rule to compute gradients
    grad_w = np.dot(dN_dw, dL_dN)  # Gradient of loss w.r.t. weights    
    grad_b = (dL_dP * dP_db).sum(axis=0)  # Gradient of loss w.r.t. bias

    return {'grad_w': grad_w, 'grad_b': grad_b}
