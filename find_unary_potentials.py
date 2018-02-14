import numpy as np

def find_unary_potential_gaussian_per_part(pose_vector, joint_locations):
    """
    Fits the data in pose_vector and centroid to a multivariate normal distribution
    
    This data corresponds to the rotation vectors and centroid for a single body part
    We have multiple data points of these two parameters

    N -> Number of data points for each body part

    IMPORTANT : Each row of the pose_vector and the joint_locations matrices should
        correspond to the same data point

    Inputs:
    ------------
    pose_vector, Nx3 Matrix
        each row corresponds to the rotation vector for the body part

    joint_locations, Nx3 matrix
        Each row corresponds to the centroid for the body part


    Outputs
    ------------
    mean, 6x1 array
        the mean for the data

    Covariance, 6x6 matrix
        Covariance Matrix 
    """

    data = np.column_stack((pose_vector, joint_locations))

    # The axis=0 means we find mean along the columns
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)   

    return mean, cov
