from prepare_data import prepare_data_new_data
from find_unary_potentials import find_unary_potential_gaussian_per_part_only_pose, find_unary_potential_mix_gaussian_per_part_only_pose
from factor_graph_node import FactorGraphNodeMixtureGaussian
from find_pairwise_potentials import find_pariwise_potential_gaussian_only_pose, find_pariwise_potential_mix_gaussian_only_pose

import numpy as np

nodes_with_neighbors = {
                    0:[1,2,3], 1:[0,4], 2:[0,5], 3:[0,6], 4:[1,7], 5:[2,8], 6:[3,9]
                    , 7:[4,10], 8:[5,11], 9:[6,12,13,14], 10:[7], 11:[8], 12:[15,9]
                    , 13:[0,4], 14:[9,17], 15:[12], 16:[13,18], 17:[14,19], 18:[16,20]
                    , 19:[17,21], 20:[18,22], 21:[19,23], 22:[20], 23:[21]
                    }

pickle_filename = "mixture_gaussians_factor_graph.obj"


def get_unary_pots_each_part(partwise_data_pose):
    """
    Gives us the unary potentials of each body part

    INPUTS:
    ------------
    partwise_data_pose
        The data points for each body part
        It is a python list of length = # of body parts
        Each element of the list is a numpy array of shape = (# data points x 3)

    OUTPUTS:
    ------------
    gauss_mixture_all_parts
        It is a list of length = # of body parts
        Each element is an instance of a sklearn.mixture.GaussianMixture
    """
    number_of_body_parts = len(partwise_data_pose)
    gauss_mixture_all_parts = []

    for body_part_index in range(number_of_body_parts):
        gauss_mixture_body_part = find_unary_potential_mix_gaussian_per_part_only_pose(partwise_data_pose[body_part_index])
        gauss_mixture_all_parts.append(gauss_mixture_body_part)
    return gauss_mixture_all_parts


def create_factor_graph(unary_potentials_all_parts):
    """
    Creates a representation of the Factor Graph
    Lists and the FactorGraphNode class is used

    INPUTS:
    ------------
    mean_all_body_parts, cov_all_body_parts:
        The unary potential parameters for each body part

    OUTPUTS:
    ------------
    factor_graph_list:
        List of the number of body parts
        Each element corresponds to the body part at that index
        Each element is an object of the FactorGraphNodeMixtureGaussian class
    """
    factor_graph_list = []
    number_of_body_parts = len(unary_potentials_all_parts)

    for body_part_index in range(number_of_body_parts):
        node = FactorGraphNodeMixtureGaussian(body_part_index, unary_potentials_all_parts[body_part_index])
        factor_graph_list.append(node)
    return factor_graph_list


def update_pairwise_potentials(body_part_node, partwise_data_pose):
    """
    Updates the pairwise potentials for a node, body_part_node
    This method depends on the global variable: nodes_with_neighbors

    INPUTS:
    ------------
    body_part_node:
        An object of the class FactorGraphNode
        It contains the data of the node whose pairwise potential we currently calculate

    partwise_data_pose
        A python list of length = # of body parts
        Each element of the list corresponds to the data for a body part
        Each element is a numpy array of shape = (# of data points x 3)

    OUTPUTS:
    ------------
    There are NO OUTPUTS
    However the body_part_node object is updated by reference
    """
    neighbors = nodes_with_neighbors[body_part_node.node_index]
    body_part_node.update_neighbors(neighbors)

    curr_node_pose_data = partwise_data_pose[body_part_node.node_index]

    for neighbor_index in neighbors:
        neighbor_pose_data = partwise_data_pose[neighbor_index]

        if neighbor_index < body_part_node.node_index:
            pairwise_pot_for_part = find_pariwise_potential_mix_gaussian_only_pose(
                neighbor_pose_data, curr_node_pose_data)
        else:
            pairwise_pot_for_part = find_pariwise_potential_mix_gaussian_only_pose(
                neighbor_pose_data, curr_node_pose_data)
        body_part_node.update_pairwise_pot(neighbor_index, pairwise_pot_for_part)


def main():
    partwise_data_pose, partwise_data_joints = prepare_data_new_data()
    # We ignore the joint data for now

    # Unary Potentials
    unary_pots_each_part = get_unary_pots_each_part(partwise_data_pose)

    # Creating the factor graph
    factor_graph_list = create_factor_graph(unary_pots_each_part)

    # Update pairwise potentials for each body part in the factor graph
    for body_part_node in factor_graph_list:
        update_pairwise_potentials(body_part_node, partwise_data_pose)

    # Saving the factor graph to perform inference later
    import pickle
    file = open(pickle_filename, "w")
    pickle.dump(factor_graph_list, file)
    file.close()


if __name__ == "__main__":
    main()