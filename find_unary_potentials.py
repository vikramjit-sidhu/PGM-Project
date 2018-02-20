import numpy as np
import ipdb

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



def find_unary_potential_gaussian_per_part_only_pose(pose_data):
    """
    Fits the data in pose_data to a multivariate normal distribution
    
    N -> Number of data points for the body part

    Inputs:
    ------------
    pose_data, Nx3 Matrix
        each row corresponds to the rotation vector for the body part


    Outputs
    ------------
    mean, 3x1 array
        the mean for the data

    Covariance, 3x3 matrix
        Covariance Matrix 
    """
    # The axis=0 means we find mean along the columns
    mean = np.mean(pose_data, axis=0)
    cov = np.cov(pose_data.T)   

    return mean, cov

from sklearn import mixture

def find_unary_potential_mix_gaussian_per_part_only_pose(pose_data):
    """
    Fits the data in pose_data to a multivariate normal distribution

    N -> Number of data points for the body part

    Inputs:
    ------------
    pose_data, Nx3 Matrix
        each row corresponds to the rotation vector for the body part


    Outputs
    ------------
    mean, 3x3 array
        1st index for mixture component, second for pose variables

    Covariance, 3x3x3 matrix
        1st index for mixture component, last two covariance matrix for that gaussian
    """

    # Fit a Gaussian mixture with EM using 3 components
    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(pose_data)
    mean = gmm.means_
    cov = gmm.covariances_

    return mean, cov