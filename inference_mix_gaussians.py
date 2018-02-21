from perform_inference import infer_pose_each_part_mix_gaussian
from visualize_point_cloud import visualize_point_cloud

import numpy as np

pickle_filename = "mixture_gaussians_factor_graph.obj"


def visualize_body_model(poses):
    from smpl.serialization import load_model
    male_model = load_model("data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")

    # Visualize base pose
    # vertices = male_model.r
    # faces = male_model.f
    # visualize_point_cloud(vertices, faces)
    # raw_input("press enter to continue")

    # Visulizing inferred pose
    male_model.pose[:] = poses.flatten()
    # male_model.J = inferred_joint.reshape(24,3)
    vertices = male_model.r
    faces = male_model.f
    visualize_point_cloud(vertices, faces)
    raw_input("press enter to continue")


def main():
    # Getting the factor graph from the pickle file
    import pickle
    file = open(pickle_filename, "r")
    factor_graph_list = pickle.load(file)
    file.close()

    # Performing inference
    inferred_pose_each_part = infer_pose_each_part_mix_gaussian(factor_graph_list)

    # Visualizing the result
    visualize_body_model(inferred_pose_each_part)


if __name__ == "__main__":
    main()