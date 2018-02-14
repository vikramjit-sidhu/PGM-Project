
# from prepare_data import prepare_data
from load_data import load_data
from find_unary_potentials import find_unary_potential_gaussian_per_part
from find_pairwise_potentials import find_pariwise_potential_gaussian
from factor_graph_node import FactorGraphNode
from perform_inference import get_pose_joint_for_each_part
from visualize_point_cloud import visualize_point_cloud

# Each key of the below dictionary is a body part index
# The list elements are the neighbors of the body parts
nodes_with_neighbors = {
                    0:[1,2,3], 1:[0,4], 2:[0,5], 3:[0,6], 4:[1,7], 5:[2,8], 6:[3,9]
                    , 7:[4,10], 8:[5,11], 9:[6,12,13,14], 10:[7], 11:[8], 12:[15,9]
                    , 13:[0,4], 14:[9,17], 15:[12], 16:[13,18], 17:[14,19], 18:[16,20]
                    , 19:[17,21], 20:[18,22], 21:[19,23], 22:[20], 23:[21]
                    }


import numpy as np

def get_unary_pots_each_part(partwise_data_pose, partwise_data_joints):
    """
    Gives us the unary potentials of each body part

    INPUTS:
    ------------
    partwise_data_pose, partwise_data_joints
        The data points for each body part

    OUTPUTS:
    ------------
    mean_all_parts
    cov_all_parts

    The above variables are lists.
    The data corresponding to each body part is stored in each index of the list
    """
    number_of_body_parts = len(partwise_data_joints)
    mean_all_parts = []
    cov_all_parts = []

    for body_part_index in range(number_of_body_parts):
        mean, cov = find_unary_potential_gaussian_per_part(
                partwise_data_pose[body_part_index], partwise_data_joints[body_part_index])
        mean_all_parts.append(mean)
        cov_all_parts.append(cov)
    return mean_all_parts, cov_all_parts


def create_factor_graph(mean_all_body_parts, cov_all_body_parts):
    """
    Creates a representation of the Factor Graph
    Lists and the FactorGraphNode class is used

    INPUTS:
    ------------
    mean_all_body_parts, cov_all_body_parts:
        The mean and covariance lists for each body part

    OUTPUTS:
    ------------
    factor_graph_list:
        List of the number of body parts
        Each element corresponds to the body part at that index
        Each element is an object of the FactorGraphNode class
    """
    factor_graph_list = []
    number_of_body_parts = len(mean_all_body_parts)

    for body_part_index in range(number_of_body_parts):
        node = FactorGraphNode(body_part_index, 
                mean_all_body_parts[body_part_index], cov_all_body_parts[body_part_index])
        factor_graph_list.append(node)
    return factor_graph_list


def update_pairwise_potentials(body_part_node, partwise_data_pose, partwise_data_joints):
    neighbors = nodes_with_neighbors[body_part_node.node_index]
    body_part_node.update_neighbors(neighbors)

    curr_node_joint_data = partwise_data_joints[body_part_node.node_index]
    curr_node_pose_data = partwise_data_pose[body_part_node.node_index]

    for neighbor_index in neighbors:
        neighbor_joint_data = partwise_data_joints[neighbor_index]
        neighbor_pose_data = partwise_data_pose[neighbor_index]

        if neighbor_index < body_part_node.node_index:
            mean, cov = find_pariwise_potential_gaussian(
                    neighbor_pose_data, neighbor_joint_data, curr_node_pose_data, curr_node_joint_data)
        else:
            mean, cov = find_pariwise_potential_gaussian(
                    curr_node_pose_data, curr_node_joint_data, neighbor_pose_data, neighbor_joint_data)
        body_part_node.update_pairwise_pot(neighbor_index, mean, cov)


def main():
    joint_data_nodewise, pose_data_nodewise = load_data()

    joint_data_combined = joint_data_nodewise.reshape(918, 72)
    pose_data_combined = pose_data_nodewise.reshape(918, 72)

    joint_mean = np.mean(joint_data_combined, axis=0)
    joint_cov = np.cov(joint_data_combined.T)

    pose_mean = np.mean(pose_data_combined, axis=0)
    pose_cov = np.cov(pose_data_combined.T)

    inferred_pose = np.random.multivariate_normal(mean=pose_mean, cov=pose_cov)
    inferred_joint = np.random.multivariate_normal(mean=joint_mean, cov=joint_cov)

    # Visualize data
    from smpl.serialization import load_model
    male_model = load_model("data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    vertices = male_model.r
    faces = male_model.f
    # visualize_point_cloud(vertices, faces)
    # raw_input("press enter to continue")

    male_model.pose[:] = inferred_pose
    male_model.J = inferred_joint.reshape(24,3)
    vertices = male_model.r
    faces = male_model.f

    visualize_point_cloud(vertices, faces)
    raw_input("press enter to continue")

    # # Unary Potentials
    # mean_all_body_parts, cov_all_body_parts = get_unary_pots_each_part(
    #                         partwise_data_pose, partwise_data_joints)

    # factor_graph_list = create_factor_graph(mean_all_body_parts, cov_all_body_parts)

    # # Update pairwise potentials for each body part in the factor graph
    # for body_part_node in factor_graph_list:
    #     update_pairwise_potentials(body_part_node, partwise_data_pose, partwise_data_joints)
    
    # Performing inference
    # inferred_pose_each_part, inferred_joints_each_part = get_pose_joint_for_each_part(factor_graph_list)


if __name__ == "__main__":
    main()