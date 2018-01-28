import numpy as np

from load_data import load_data

number_of_body_parts = 24
number_of_data_points = 50


def get_data_for_body_part(list_all_data, body_part_num):
    """
    all_data should be a list of np arrays
    """
    body_part_data = np.empty((number_of_data_points, 3))
    data_index = 0
    for body_data in list_all_data:
        body_part_data[data_index, :] = body_data[body_part_num,:]
        data_index += 1
    return body_part_data


def prepare_data():
    """
    The joint_data and pose_data lists contain in each element a np array.
    Each np array is 24x3 and corresponds to the data for a single pose
    Each row of the numpy array corresponds to a body part


    This method 

    OUTPUTS:
    ------------
    partwise_data_pose:
        A python list, each element of the list contains all the data 
        for a part.
        e.g. the first element of the list is a 50x3 np array.
        This array will contain all the data for the first body part

    partwise_data_joints:
        Similar to partwise_data_pose for the joints

    """
    joint_data, pose_data = load_data()

    partwise_data_pose = []
    partwise_data_joints = []

    for body_part_num in range(number_of_body_parts):
        partwise_data_pose.append(get_data_for_body_part(pose_data, body_part_num))
        partwise_data_joints.append(get_data_for_body_part(joint_data, body_part_num))
    
    return partwise_data_pose, partwise_data_joints