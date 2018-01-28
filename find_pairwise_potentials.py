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


def get_relative_rotation_all_data_pts(rot_vectors_1, rot_vectors_2):
    N = rot_vectors_1.shape[0]
    combined_rotation = np.empty((N, 3))
    for row_index in range(N):
        combined_rotation[row_index, :] = get_relative_rotation(rot_vectors_1[row_index,:], rot_vectors_2[row_index,:]).reshape(1,3)
    return combined_rotation


def find_pariwise_potential_gaussian(part1_pose, part1_joints, part2_pose, part2_joints):
    """
    Fit a multiavariate Gaussian to the parameters describing two parts

    N -> Number of data points

    Each row of data in the input variables should match the body model 
        that they are extracted from

    Inputs:
    ------------
    part1_pose, Nx3 Matrix
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

    combined_rotation = get_relative_rotation_all_data_pts(part1_pose, part2_pose)
    combined_joint_locations = part1_joints - part2_joints

    data = np.column_stack((combined_rotation, combined_joint_locations))

    # The axis=0 means we find mean along the columns
    mean = np.mean(data, axis=0)
    cov = np.cov(data)

    return mean, cov


def find_ternary_potentials(part1_pose, part1_joints, part2_pose, part2_joints, part3_pose, part3_joints, part4_pose, part4_joints):
    combined_pose_1_2 = get_relative_rotation_all_data_pts(part1_pose, part2_pose)
    combined_joints_1_2 = part1_joints - part2_joints

    combined_pose_1_2_3 = get_relative_rotation_all_data_pts(combined_pose_1_2, part3_pose)
    combined_joints_1_2_3 = combined_joints_1_2 - part3_joints

    return (find_pariwise_potential_gaussian(combined_pose_1_2_3, combined_joints_1_2_3, part4_pose, part4_joints))


def find_quarternary_pots(part1_pose, part1_joints, part2_pose, part2_joints, part3_pose, part3_joints, part4_pose, part4_joints, part5_pose, part5_joints):
    combined_pose_1_2 = get_relative_rotation_all_data_pts(part1_pose, part2_pose)
    combined_joints_1_2 = part1_joints - part2_joints

    return (find_ternary_potentials(combined_pose_1_2, combined_joints_1_2, part3_pose, part3_joints, part4_pose, part4_joints, part5_pose, part5_joints))