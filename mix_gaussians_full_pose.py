from prepare_data import prepare_data_new_data
from visualize_point_cloud import visualize_point_cloud

import numpy as np

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


def test_alternate_approach():
    partwise_data_pose, _ = prepare_data_new_data()

    # Prepare data
    pose_data = partwise_data_pose[0]
    for part_pose_data in partwise_data_pose[1:]:
        pose_data = np.column_stack((pose_data, part_pose_data))

    # Perform training
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=3)
    gmm.fit(pose_data)

    pose = gmm.sample()[0].reshape(72,1)
    visualize_body_model(pose)


if __name__ == "__main__":
    test_alternate_approach()