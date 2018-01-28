import numpy as np

data_path = "result/"
amount_of_data = 50

joint_locations_filename_intial = "joint_data_"
pose_data_filename_initial = "pose_data_"
file_extension = ".npy"


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
    joint_data = []
    pose_data = []
    for index in range(1, amount_of_data+1):
        joint_filename = data_path + joint_locations_filename_intial + str(index) + file_extension
        joint_data.append(np.load(open(joint_filename, "rb")))

        pose_data_filename = data_path + pose_data_filename_initial + str(index) + file_extension
        pose_data.append(np.load(open(pose_data_filename, "rb")))
    return joint_data, pose_data
