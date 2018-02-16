"""
In this module we modify the multivariate normal distribution
1. We find the product of 2 multivariate normal distributions
2. We find the conditional distribution of a multivariate normal distribution
"""

import numpy as np

def multiply_two_multivariate_normal_distributions_pdfs(mean1, mean2, cov1, cov2):
    """
    Find the product of the p.d.f's of two multivariate normal distributions

    The parameters should be np matrices / arrays
    """
    try:
        cov1_inv = np.linalg.inv(cov1)
    except np.linalg.LinAlgError:
        cov1_inv = np.linalg.pinv(cov1)

    try:
        cov2_inv = np.linalg.inv(cov2)
    except np.linalg.LinAlgError:
        cov2_inv = np.linalg.pinv(cov2)
    
    try:
        cov_res = np.linalg.inv(cov1_inv + cov2_inv)
    except np.linalg.LinAlgError:
        cov_res = np.linalg.pinv(cov1_inv + cov2_inv)
        
    mean_res = np.dot(cov_res, (np.dot(cov1_inv, mean1) + np.dot(cov2_inv, mean2)))

    return mean_res, cov_res


def conditional_distribution_multivariate_normal(mean1, mean2, cov11, cov12, cov21, cov22, a):
    """
    Given a multivariate normal distribution of 12 variables.
    We condition on 6 of these variables.
    Then we find the parameters of the resulting distribution

    We are conditioning on the varibles represented by mean2
    'a' is the value taken by these variables, i.e. the value we condition on 
    cov = [cov11 cov12 ; cov21 cov22]
    """
    try:
        cov_22_inv = np.linalg.inv(cov22)
    except np.linalg.LinAlgError:
        cov_22_inv = np.linalg.pinv(cov22)

    int_matrix = np.dot(cov12, cov_22_inv)
    
    mean_conditional = mean1 + np.dot(int_matrix, (a-mean2))
    cov_conditional = cov11 - np.dot(int_matrix, cov21)

    return mean_conditional, cov_conditional