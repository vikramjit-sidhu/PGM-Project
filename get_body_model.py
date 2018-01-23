import numpy as np
from visualize_point_cloud import visualize_point_cloud
import cv2
from extract_parts import *
import ipdb
def load_body_model():
    #vertices = np.load('data/verts.npy')
    #faces = np.load('data/faces.npy')
    pose_data = np.load('/home/garvita/Documents/PGM/PGM-Project-master/result/pose_data_1.npy')
    print(pose_data)
    centroid_data = np.load('/home/garvita/Documents/PGM/PGM-Project-master/result/centroid_data_1.npy')
    print(centroid_data)
    return pose_data

def load_smpl_body_model():
    from smpl.serialization import load_model
    male_model = load_model("data/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
    # changing the pose of the model
    raw_input('press to continue')
    k = 1
    part_num = 0
    var_list = np.linspace(0.01,0.2,50)
    for var in var_list:
        male_model.pose[:] = np.random.rand(male_model.pose.size) * 0
        #print(dir(male_model))
        #ipdb.set_trace()
        part_pose = pose_part(male_model.pose[:])
        vertices = male_model.r
        centroid = centroid_part(vertices)
        faces = male_model.f
        #print(part_pose)
        visualize_point_cloud(vertices,centroid, faces)
        raw_input('press')

        ### only for testing
        R_mat = np.float64(np.array([[0,1,0], [-1,0,0], [0,0,1]]))
        #R_vec = cv2.Rodrigues(src=R_mat)[0]
        #print(R_vec)
        i = 0
        while(i < 72):
            male_model.pose[i:i+3] = np.array([0,0,-1.57])
            vertices = male_model.r
            faces = male_model.f
            male_model.pose[i:i+3] = np.array([0,0,0])

            i = i + 3
            print(i)
            visualize_point_cloud(vertices,centroid, faces)

            raw_input('press')

        ########
        v_file = 'result/vdata_' + str(k) + '.npy'
        f_file = 'result/fdata_' + str(k) + '.npy'
        p_file = 'result/pose_data_' + str(k) + '.npy'
        c_file = 'result/centroid_data_' + str(k) + '.npy'
        np.save(v_file,vertices)
        np.save(f_file,faces)
        np.save(p_file,part_pose)
        np.save(c_file,centroid)
        k = k + 1
    return vertices, faces


if __name__ == "__main__":
    load_smpl_body_model()
    load_body_model()