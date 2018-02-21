
from multivariate_normal_dist import multiply_two_multivariate_normal_distributions_pdfs, conditional_distribution_multivariate_normal

import numpy as np

num_samples = 2400


def generate_initial_joint_locations(num_body_parts):
    """
    For the joint locations, we simply initialize it randomly for now.
    Better option -> Sample from unary potentials?
    """
    return np.random.rand(num_body_parts, 3)


def generate_initial_pose_matrix(num_body_parts):
    """
    For the pose, we use the rest pose
    The matrix we initialize is all zeroes.
    Its size is (num_body_parts x 3)
    """
    return np.zeros(shape=(num_body_parts, 3))


def get_node_val(node_index, pose_matrix, joint_location_matrix):
    """
    Extract the pose and joint data from the matrices for a node
    """
    pose = pose_matrix[node_index]
    joint_location = joint_location_matrix[node_index]
    return pose, joint_location


def condition_on_pairwise_potential(node_index, conditioning_node_index, mean, cov, cond_var_value):
    """
    conditioning_node_index is the node that we condition on
    mean, cov is the mean and covariance of both the nodes
    """
    slicing_index_limit = mean.shape[0] // 2

    # Dividing the mean into the first and second halves
    mean_1 = mean[0:slicing_index_limit]
    mean_2 = mean[slicing_index_limit:]

    cov_11 = cov[0:slicing_index_limit, 0:slicing_index_limit]
    cov_12 = cov[0:slicing_index_limit, slicing_index_limit:]
    cov_21 = cov[slicing_index_limit:, 0:slicing_index_limit]
    cov_22 = cov[slicing_index_limit:, slicing_index_limit:]    

    if node_index < conditioning_node_index:
        # The last half of the variables are the ones conditioned on
        # mean_cond = mean_1 + cov_12 * inv(cov_22) * (cond_var_value - mean_2)
        # cov_cond = cov_11 - cov_12 * inv(cov_22) * cov_21
        cond_mean, cond_cov = conditional_distribution_multivariate_normal(
                    mean_1, mean_2
                    , cov_11, cov_12, cov_21, cov_22, cond_var_value)
    else:
        # The first half of the variables are the ones conditioned on
        # mean_cond = mean_2 + cov_21 * inv(cov_11) * (cond_var_value - mean_1)
        # cov_cond = cov_22 - cov_21 * inv(cov_11) * cov_12
        cond_mean, cond_cov = conditional_distribution_multivariate_normal(
                    mean_2, mean_1
                    , cov_22, cov_21, cov_12, cov_11, cond_var_value)
    return cond_mean, cond_cov


def get_pose_joint_for_each_part(factor_graph_list):
    """
    Performs inference on the factor graph to get pose and joint
    data for each node

    N -> Number of body parts

    INPUTS:
    ------------
    factor_graph_list:
        A list of N elements
        Each element corresponds to the body part at that index
        Each element is an object of the FactorGraphNode class
        It contains the unary, pairwise potentials and the neighbors for that node

    OUTPUTS:
    ------------
    inferred_pose_each_part, inferred_joints_each_part:
        Nx3 matrices
        Each row corresponds to the data for that body part
    """
    num_body_parts = len(factor_graph_list)
    # Generate the intial sample to start Gibbs Sampling
    pose_matrix = generate_initial_pose_matrix(num_body_parts)
    joint_location_matrix = generate_initial_joint_locations(num_body_parts)

    # Generate nodes which will be selected for Gibbs Sampling
    node_indices_array = np.ndarray.tolist(np.random.randint(0, 24, size=num_samples))

    # Start Gibbs Sampling over Factor Graph
    for node_index in node_indices_array:
        node_to_update = factor_graph_list[node_index]

        # The below variables mean, the combined distribution of the node which we 
        # sample from
        resulting_mean = node_to_update.unary_pot_mean
        resulting_cov = node_to_update.unary_pot_cov

        # Building a resulting distribution by combining all the distributions
        for neighbor_index in node_to_update.neighbors:
            # We find parameters of joint distribution of current node and neighbor
            pairwise_mean = node_to_update.pairwise_pots_mean[neighbor_index]
            pairwise_cov = node_to_update.pairwise_pots_cov[neighbor_index]

            # Finding value of conditioned variable (neighbor)
            pose, joint_location = get_node_val(neighbor_index, pose_matrix, joint_location_matrix)
            cond_var_value = np.concatenate([pose, joint_location])

            # Conditioning on the pairwise potentials to get resulting distribution
            cond_mean, cond_cov = condition_on_pairwise_potential(node_index, neighbor_index
                        , pairwise_mean, pairwise_cov, cond_var_value)

            resulting_mean, resulting_cov = multiply_two_multivariate_normal_distributions_pdfs(
                    cond_mean, resulting_mean, cond_cov, resulting_cov)

        # Perform Gibbs Sampling on the resulting distribution
        # pose, joint_locations = 
    return pose_matrix, joint_location_matrix


def get_node_val_pose_only(node_index, pose_matrix):
    """
    Extract the pose from the matrix for a node with index node_index
    """
    pose = pose_matrix[node_index]
    return pose


def infer_pose_each_part(factor_graph_list):
    """
    Performs inference on the factor graph to get pose data for each node

    N -> Number of body parts

    INPUTS:
    ------------
    factor_graph_list:
        A list of N elements
        Each element is an object of FactorGraphNode class
        Each object corresponds to the node at that index in the list

    OUTPUTS:
    ------------
    inferred_pose_each_part:
        Nx3 matrix
        Each row corresponds to the data for that body part
    """
    num_body_parts = len(factor_graph_list)

    # Generate the intial sample to start Gibbs Sampling
    inferred_pose = generate_initial_pose_matrix(num_body_parts)

    # Generate nodes which will be selected for Gibbs Sampling
    node_indices_array = np.ndarray.tolist(np.random.randint(0, 24, size=num_samples))

    # Start Gibbs Sampling over Factor Graph
    for node_index in node_indices_array:
        node_to_update = factor_graph_list[node_index]

        # The below variables mean, the combined distribution of the node which we 
        # sample from
        resulting_mean = node_to_update.unary_pot_mean
        resulting_cov = node_to_update.unary_pot_cov

        # import ipdb; ipdb.set_trace()
        # Building a resulting distribution by combining all the distributions
        for neighbor_index in node_to_update.neighbors:
            # We find parameters of joint distribution of current node and neighbor
            pairwise_mean = node_to_update.pairwise_pots_mean[neighbor_index]
            pairwise_cov = node_to_update.pairwise_pots_cov[neighbor_index]

            # Finding value of conditioned variable (neighbor)
            pose = get_node_val_pose_only(neighbor_index, inferred_pose)


            # Conditioning on the pairwise potentials to get resulting distribution
            cond_mean, cond_cov = condition_on_pairwise_potential(node_index, neighbor_index
                        , pairwise_mean, pairwise_cov, pose)

            resulting_mean, resulting_cov = multiply_two_multivariate_normal_distributions_pdfs(
                    cond_mean, resulting_mean, cond_cov, resulting_cov)

        # Perform Sampling on the resulting distribution
        pose_curr_node = np.random.multivariate_normal(mean=resulting_mean, cov=resulting_cov)
        
        # Updating the value in the overall pose matrix
        inferred_pose[node_to_update.node_index] = pose_curr_node

    return inferred_pose



def find_distribution_to_sample(mixture_weights):
    """
    Finds which Gaussian Distribution to choose from in a mixture of 3 Gaussians
    The chosen distribution can then be sampled from

    INPUTS:
    ------------
    mixture_weights
        A list of 3 elements, which correspond to the weights for each of the 3 distributions


    OUTPUTS:
    ------------
    choosen_gaussian_index
        The index of the distribution to use
        This index corresponds to the index of the weight in the mixture_weights list
    """
    if isinstance(mixture_weights, np.ndarray):
        mixture_weights = mixture_weights.tolist()

    mixture_weights_sorted = sorted(mixture_weights)
    uniform_sample = np.random.uniform()

    # Finding which interval the uniform sample lies in
    curr_range_end = 0
    for i in range(len(mixture_weights_sorted)):
        curr_range_end += mixture_weights_sorted[i]
        if uniform_sample < curr_range_end:
            weight_chosen = mixture_weights_sorted[i]
            break

    # Extracting the original index from the list
    choosen_gaussian_index = mixture_weights.index(weight_chosen)

    return choosen_gaussian_index


def infer_pose_each_part_mix_gaussian(factor_graph_list):
    """
    Performs inference on the factor graph to get pose data for each node

    N -> Number of body parts

    INPUTS:
    ------------
    factor_graph_list:
        A list of N elements
        Each element is an object of FactorGraphNode class
        Each object corresponds to the node at that index in the list

    OUTPUTS:
    ------------
    inferred_pose_each_part:
        Nx3 matrix
        Each row corresponds to the data for that body part
    """
    num_body_parts = len(factor_graph_list)

    # Generate the intial sample to start Gibbs Sampling
    inferred_pose = generate_initial_pose_matrix(num_body_parts)

    # Generate nodes which will be selected for Gibbs Sampling
    node_indices_array = np.ndarray.tolist(np.random.randint(0, 24, size=num_samples))
    
    # Start Gibbs Sampling over Factor Graph
    for node_index in node_indices_array:
        node_to_update = factor_graph_list[node_index]

        # Finding which mixture Gaussian distribution to use for unary pot
        index_unary_distribution = find_distribution_to_sample(node_to_update.unary_pot.weights_)

        # Creating combined variables for mean and covariance
        resulting_mean = node_to_update.unary_pot.means_[index_unary_distribution]
        resulting_cov = node_to_update.unary_pot.covariances_[index_unary_distribution]

        # Building a resulting distribution by combining all the distributions
        for neighbor_index in node_to_update.neighbors:
            # Gaussian Mixture instance for pairwise potential
            gauss_mixture_pairwise_pots = node_to_update.pairwise_pots[neighbor_index]

            # Finding which mixture Gaussian distribution to use for pairwise pot
            distribution_index = find_distribution_to_sample(gauss_mixture_pairwise_pots.weights_)

            pairwise_mean = gauss_mixture_pairwise_pots.means_[distribution_index]
            pairwise_cov = gauss_mixture_pairwise_pots.covariances_[distribution_index]

            # Finding value of conditioned variable (neighbor)
            pose = get_node_val_pose_only(neighbor_index, inferred_pose)

            # Conditioning on the pairwise potentials to get resulting distribution
            cond_mean, cond_cov = condition_on_pairwise_potential(node_index, neighbor_index
                                                                  , pairwise_mean, pairwise_cov, pose)

            resulting_mean, resulting_cov = multiply_two_multivariate_normal_distributions_pdfs(
                cond_mean, resulting_mean, cond_cov, resulting_cov)

        # Perform Sampling on the resulting distribution
        pose_curr_node = np.random.multivariate_normal(mean=resulting_mean, cov=resulting_cov)

        # Updating the value in the overall pose matrix
        inferred_pose[node_to_update.node_index] = pose_curr_node

    return inferred_pose