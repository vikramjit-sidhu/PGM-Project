
class FactorGraphNode:
    def __init__(self, node_index, unary_pot_mean, unary_pot_cov):
        # Keep variables here for the joint locations and rotation matrices?

        self.node_index = node_index
        self.neighbors = []
        self.unary_pot_mean = unary_pot_mean
        self.unary_pot_cov = unary_pot_cov
        self.pairwise_pots_mean = {}
        self.pairwise_pots_cov = {}


    def update_neighbors(self, neighbors):
        self.neighbors = neighbors

    def update_pairwise_pot(self, neighbor_index, mean, cov):
        self.pairwise_pots_mean[neighbor_index] = mean
        self.pairwise_pots_cov[neighbor_index] = cov



class FactorGraphNodeMixtureGaussian:
    def __init__(self, node_index, unary_pot_gauss_mixture):
        # Keep variables here for the joint locations and rotation matrices?
        self.node_index = node_index
        self.neighbors = []
        self.unary_pot = unary_pot_gauss_mixture
        self.pairwise_pots = {}


    def update_neighbors(self, neighbors):
        self.neighbors = neighbors

    def update_pairwise_pot(self, neighbor_index, pairwise_pots_gauss_mixture):
        self.pairwise_pots[neighbor_index] = pairwise_pots_gauss_mixture