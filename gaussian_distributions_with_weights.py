
from multivariate_normal_dist import conditional_distribution_multivariate_normal, multiply_two_multivariate_normal_distributions_pdfs

import numpy as np


class GaussianDistributionsWithWeights:
    """
    This class is used to keep track of various multivariate Gaussian Distributions
    Each distribution is asscociated with a weight representing how likely it is to be chosen
    We keep track of all the weights in weights_list
    Corresponding to each weight we have a means and cov dictionary (means_dict, covs_dict)
    """
    def __init__(self):
        self.weight_list = []
        self.means_dict = {}
        self.covs_dict = {}

  
    def set_weights_means_covs(self, weights, means, covs):
        """
        Set the weights list
        Corresponding to the weights, Set the means and covariances dictionaries.
        """
        if isinstance(weights, np.ndarray):
            weights = weights.tolist()

        self.weight_list = weights

        for index, weight in enumerate(self.weight_list):
            self.means_dict[weight] = means[index]
            self.covs_dict[weight] = covs[index]


    def add_gaussian_distributions(self, weights_new, means_new, covs_new):
        """
        Add a mixture of Gaussian Distributions to the current object
        It updates the instance variables weight_list, means_dict, covs_dict

        INPUTS:
        ------------
        weights_new -> List of weights
        means_new, covs_new -> List of means and covariances
        Each of these elements correspond to each other

        """
        if isinstance(weights_new, np.ndarray):
            weights_new = weights_new.tolist()

        weight_list_curr = self.weight_list
        means_dict_curr = self.means_dict
        covs_dict_curr = self.covs_dict

        self.weight_list = []
        self.means_dict = {}
        self.covs_dict = {}

        # We have to take all combinations of the weights, which a loop of loop does
        for weight_curr in weight_list_curr:
            mean_curr = means_dict_curr[weight_curr]
            cov_curr = covs_dict_curr[weight_curr]

            for index, weight_new in enumerate(weights_new):
                mean_new = means_new[index]
                cov_new = covs_new[index]

                weight_combined = weight_curr * weight_new
                self.weight_list.append(weight_combined)

                mean_combined, cov_combined = multiply_two_multivariate_normal_distributions_pdfs(
                        mean_curr, mean_new, cov_curr, cov_new)
                self.means_dict[weight_combined] = mean_combined
                self.covs_dict[weight_combined] = cov_combined


    def get_gaussian_to_sample_from(self):
        """
        Choose a Gaussian Distribution to sample from
        This is done using the weights of all the Gaussians, i.e. weight_list

        OUTPUTS:
        ------------
        The parameters of the chosen Gaussian distribution

        mean_chosen_gaussain
            A numpy array, dimension nx1

        cov_chosen_gaussian
            A numpy ndarray, dimension nxn
        """
        chosen_weight = self.__choose_weight_from_list_weights(self.weight_list)
        mean_chosen_gaussian = self.means_dict[chosen_weight] 
        cov_chosen_gaussian = self.covs_dict[chosen_weight]
        return mean_chosen_gaussian, cov_chosen_gaussian


    def __choose_weight_from_list_weights(self, weight_list):
        """
        Finds which Gaussian Distribution to choose from in a mixture of n Gaussians
        The chosen distribution can then be sampled from

        INPUTS:
        ------------
        weight_list
            A list of 'n' elements, each of which are the weights of the distribution


        OUTPUTS:
        ------------
        weight_chosen
            The weight we have chosen
        """
        if isinstance(weight_list, np.ndarray):
            weight_list = weight_list.tolist()

        weight_list_sorted = sorted(weight_list)
        uniform_sample = np.random.uniform()

        # Finding which interval the uniform sample lies in
        curr_range_end = 0
        for weight in weight_list_sorted:
            curr_range_end += weight
            if uniform_sample < curr_range_end:
                weight_chosen = weight
                break
        else:
            weight_chosen = weight_list_sorted[-1]
        return weight_chosen
