import numpy as np

def load_body_model():
    vertices = np.load('data/verts.npy')
    faces = np.load('data/faces.npy')
    return vertices, faces

def load_smpl_body_model():
    from smpl.serialization import load_model
    male_model = load_model("data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    
    # changing the pose of the model
    male_model.pose[:] = np.random.rand(male_model.pose.size) * .2

    vertices = male_model.r
    faces = male_model.f
    return vertices, faces