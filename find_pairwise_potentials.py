import numpy as np


def get_relative_rotation(r1, r2):
    """
    Find relative rotation between two rotation vectors

    Inputs:
    ------------
    r1
        3x1 rotation vector

    r2
        3x1 rotation vector

    Outputs:
    ------------
    r_relative
        Relative rotation vectors
    """
    from cv2 import Rodrigues

    R1 = Rodrigues(src=r1)[0]   # [0] is used because Rodrigues returns a tuple
    R2 = Rodrigues(src=r2)[0]

    R_relative = R1.T * R2
    # Finding corresponding rotation vector from Matrix
    r_relative = Rodrigues(src=R_relative)[0]
    return r_relative


def find_pariwise_potential_gaussian(part1_Rotation_vectors, part1_joints, part2_Rotation_vectors, part2_joints):
    """
    Fit a multiavariate Gaussian to the parameters describing two parts

    N -> Number of data points

    Each row of data in the input variables should match the body model 
        that they are extracted from

    Inputs:
    ------------
    part1_Rotation_vectors, Nx3 Matrix
        each row corresponds to a data point for the rotation vector of the body part

    part1_joints, Nx3 matrix
        Each row corresponds to a data point for the centroid for the body part

    The inputs for the other part are similar

    Outputs
    ------------
    mean, 12x1 array
        the mean for the data

    Covariance, 12x12 matrix
        Covariance Matrix 
    """

    N = part1_Rotation_vectors.shape[0]
    combined_rotation = []
    for row_index in range(N):
        combined_rotation.append(get_relative_rotation(part1_Rotation_vectors[row_index,:], part2_Rotation_vectors[row_index,:]))
    combined_rotation = np.array(combined_rotation)

    combined_joint_locations = part1_joints - part2_joints

    data = np.column_stack((combined_rotation, combined_joint_locations))

    # The axis=0 means we find mean along the columns
    mean = np.mean(data, axis=0)
    cov = np.cov(data)

    return mean, cov
