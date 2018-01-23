import numpy as np

def find_pariwise_potential_gaussian(part1_Rotation_vectors, part1_centroids, part2_Rotation_vectors, part2_centroids):
    """
    Fit a multiavariate Gaussian to the parameters describing two parts

    N -> Number of data points

    Each row of data in the input variables should match the body model 
        that they are extracted from

    Inputs:
    ------------
    part1_Rotation_vectors, Nx3 Matrix
        each row corresponds to a data point for the rotation vector of the body part

    part1_centroids, Nx3 matrix
        Each row corresponds to a data point for the centroid for the body part

    The inputs for the other part are similar

    Outputs
    ------------
    mean, 12x1 array
        the mean for the data

    Covariance, 12x12 matrix
        Covariance Matrix 
    """
    data = np.column_stack((part1_Rotation_vectors, part2_Rotation_vectors, part1_centroids, part2_centroids))

    N = data.shape[0]

    # The axis=0 means we find mean along the columns
    mean = np.mean(data, axis=0)
    Covariance = np.cov(data)

    return mean, Covariance
