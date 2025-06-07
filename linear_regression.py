import numpy as np
from numpy.typing import NDArray  #optinal for type hints (type signature)
from typing import Tuple, Dict, List

def forward_linear_regression(x: NDArray, y: NDArray, weights:Dict[str,NDArray]) ->Tuple[float,Dict[str,NDArray]] :
    """
    Computes the forward pass of a linear regression model.
    
    Parameters:
    - x: Input array .
    - w: Weights array .
    - y: targets .
    
    Returns:
    - The output of the linear regression model.
    """
    assert x.shape[0]== y.shape[0], "Input and target arrays must have the same length."
    assert x.shape [0]== weights['w'].shape[1] ,"assert matrix multiplication is valid"
    assert weights['b'].shape[0]== weights['b'].shape[1]==1, "Bias must be a scalar."

    # Compute the output of the linear regression model
    N=np.dot(weights['w'], x) 
    P=N + weights['b']
    # Compute the loss (mean squared error)
    loss = np.mean((P - y) ** 2)
    # Store the output and weights in a dictionary
    forward_output : Dict[str, NDArray] = {}
    forward_output['X']=x
    forward_output['Y']=y
    forward_output['N'] = N
    forward_output['P'] = P

    return loss, forward_output
