import numpy as np

data_path = "data/"

joint_locations_filename_intial = "joint_data_2"
pose_data_filename_initial = "pose_data_194389"
file_extension = ".npy"

num_pose_data_points = 194388
num_joint_data_points = 918

num_body_parts = 24


def load_data():
    """
    Loads the joint locations and the rotation vectors (pose data) for each body model.

    The data is set in such a way that the pose data and the joint locations correspond 
    for each data point.


    OUTPUTS:
    ------------
    joint_data
        A list of length amount_of_data
        Each element of the list is a 24x3 np array.
        Each row of the np array is the joint location in x,y,z for a body part 

    pose_data
        Similar to joint_data for pose, i.e. rotation vectors

    """
    joint_filename = data_path + joint_locations_filename_intial + file_extension
    pose_filename = data_path + pose_data_filename_initial + file_extension

    joint_data = np.load(open(joint_filename, "rb"))
    joint_data = joint_data.reshape(num_joint_data_points, num_body_parts, 3)
    
    pose_data = np.load(open(pose_filename, "rb"))
    pose_data = pose_data.reshape(num_pose_data_points, num_body_parts, 3)

    return joint_data, pose_data
