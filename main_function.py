
from prepare_data import prepare_data
from find_unary_potentials import find_unary_potential_gaussian_per_part
from find_pairwise_potentials import find_pariwise_potential_gaussian, find_ternary_potentials, find_quarternary_pots

# These are the indices of the body parts over which we calculate the pairwise potentials
potential_pairs = {1:4, 2:4, 3:6, 4:7, 5:8, 7:10, 8:11, 12:17, 13:16, 14:17, 16:18, 17:19, 18:20, 19:21, 20:22, 21:23}

def get_unary_pots_each_part(partwise_data_pose, partwise_data_joints):
    """
    Gives us the unary potentials of each body part

    INPUTS:
    ------------
    partwise_data_pose, partwise_data_joints
        The data points for each body part
    """
    number_of_body_parts = len(partwise_data_joints)
    mean_all_parts = []
    cov_all_parts = []

    for body_part_index in range(number_of_body_parts):
        mean, cov = find_unary_potential_gaussian_per_part(partwise_data_pose[body_part_index], partwise_data_joints[body_part_index])
        mean_all_parts.append(mean)
        cov_all_parts.append(cov)
    return mean_all_parts, cov_all_parts


def get_pairwise_pots(partwise_data_pose, partwise_data_joints):
    pairwise_pots_mean = {}
    pairwise_pots_cov = {}
    for part1_index, part2_index in potential_pairs.iteritems():
        pairwise_pots_mean[part1_index], pairwise_pots_cov[part1_index] = find_pariwise_potential_gaussian(partwise_data_pose[part1_index], partwise_data_joints[part1_index], partwise_data_pose[part2_index], partwise_data_joints[part2_index])
    return pairwise_pots_mean, pairwise_pots_cov


def main():
    partwise_data_pose, partwise_data_joints = prepare_data()
    mean_all_parts, cov_all_parts = get_unary_pots_each_part(partwise_data_pose, partwise_data_joints)

    # Pairwise potentials
    pairwise_pots_mean, pairwise_pots_cov = get_pairwise_pots(partwise_data_pose, partwise_data_joints)
    # Ternary and quarternary potentials
    pairwise_pots_mean[0], pairwise_pots_cov[0] = find_ternary_potentials(partwise_data_pose[0], partwise_data_joints[0], partwise_data_pose[1], partwise_data_joints[1], partwise_data_pose[2], partwise_data_joints[2], partwise_data_pose[3], partwise_data_joints[3])
    pairwise_pots_mean[9], pairwise_pots_cov[9] = find_quarternary_pots(partwise_data_pose[9], partwise_data_joints[9], partwise_data_pose[6], partwise_data_joints[6], partwise_data_pose[12], partwise_data_joints[12], partwise_data_pose[13], partwise_data_joints[13], partwise_data_pose[14], partwise_data_joints[14])


if __name__ == "__main__":
    main()