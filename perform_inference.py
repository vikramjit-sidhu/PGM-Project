
from multivariate_normal_dist import conditional_distribution_multivariate_normal, multiply_two_multivariate_normal_distributions_pdfs
from gaussian_distributions_with_weights import GaussianDistributionsWithWeights

import numpy as np

num_samples = 10000


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

        # Getting GaussianMixture object for unary pot
        unary_pot = node_to_update.unary_pot

        # Prepare object to represent mixture of Gaussians
        mix_gaussians_obj = GaussianDistributionsWithWeights()
        mix_gaussians_obj.set_weights_means_covs(unary_pot.weights_, unary_pot.means_, unary_pot.covariances_)

        # Conditioning on all the neighbors to get resulting distributions
        for neighbor_index in node_to_update.neighbors:
            # Finding value of conditioned variable (neighbor)
            pose = get_node_val_pose_only(neighbor_index, inferred_pose)
            
            # Gaussian Mixture instance for pairwise potential
            pairwise_pots_gauss_mixture_obj = node_to_update.pairwise_pots[neighbor_index]

            # Condition on all mixtures in Pairwise potentials
            weights = pairwise_pots_gauss_mixture_obj.weights_

            means_conditioned = []
            covs_conditioned = []
            for index in range(len(weights)):
                pairwise_mean = pairwise_pots_gauss_mixture_obj.means_[index]
                pairwise_cov = pairwise_pots_gauss_mixture_obj.covariances_[index]
                mean_cond, cov_cond = condition_on_pairwise_potential(
                    node_index, neighbor_index, pairwise_mean, pairwise_cov, pose)
                
                means_conditioned.append(mean_cond)
                covs_conditioned.append(cov_cond)

            # Add these mixtures to the object
            mix_gaussians_obj.add_gaussian_distributions(weights, means_conditioned, covs_conditioned)

        # Selecting the Gaussian distribution to sample from in the Gaussian Mixture
        mean_gaussian, cov_gaussian = mix_gaussians_obj.get_gaussian_to_sample_from()

        # Perform Sampling 
        pose_curr_node = np.random.multivariate_normal(mean=mean_gaussian, cov=cov_gaussian)

        # Updating the value in the overall pose matrix
        inferred_pose[node_to_update.node_index] = pose_curr_node

    return inferred_pose

