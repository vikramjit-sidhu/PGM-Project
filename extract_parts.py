import numpy as np
def extract_each_part():
    labels = np.load('data/partnames_per_vertex.pkl')
    body_part_names = list(set(labels))
    verts_body_part = []
    # for boxdy_part_name in body_part_names:

def centroid_part(X):
    centroid_p = {}
    labels = np.load('data/partnames_per_vertex.pkl')
    label_names = list(set(labels))
    print(label_names)
    label_size = len(label_names)
    num_pts,_ = X.shape
    for j in range(0,label_size):
        X_tmp = []
        for i in range(0,num_pts):
            if labels[i] == label_names[j]:
                X_tmp.append(X[i])
            X_tmp_arr = np.array(X_tmp)
        #print(X_tmp_arr.shape)
        centroid_p[j] = (np.mean(X_tmp_arr, axis=0))
        #print('mean ',centroid_p[j])
    return centroid_p

def pose_part(pose_arr):
    part_pose = np.empty((24,3), dtype=float)
    part_pose_tmp = []
    part_num = 0
    while(part_num < 72):
        pose_tmp = pose_arr[part_num:part_num+3]
        part_pose_tmp.append(np.asarray(pose_tmp))                
        part_num = part_num + 3

    #### same order as in vertices
    part_pose[0] = part_pose_tmp[21]
    part_pose[1] = part_pose_tmp[0]
    part_pose[2] = part_pose_tmp[23]
    part_pose[3] = part_pose_tmp[5]
    part_pose[4] = part_pose_tmp[7]
    part_pose[5] = part_pose_tmp[22]
    part_pose[6] = part_pose_tmp[9]
    part_pose[7] = part_pose_tmp[2]
    part_pose[8] = part_pose_tmp[4]
    part_pose[9] = part_pose_tmp[13]
    part_pose[10] = part_pose_tmp[8]
    part_pose[11] = part_pose_tmp[15]
    part_pose[12] = part_pose_tmp[11]
    part_pose[13] = part_pose_tmp[1]
    part_pose[14] = part_pose_tmp[18]
    part_pose[15] = part_pose_tmp[19]
    part_pose[16] = part_pose_tmp[14]
    part_pose[17] = part_pose_tmp[12]
    part_pose[18] = part_pose_tmp[3]
    part_pose[19] = part_pose_tmp[6]
    part_pose[20] = part_pose_tmp[16]
    part_pose[21] = part_pose_tmp[10]
    part_pose[22] = part_pose_tmp[20]
    part_pose[23] = part_pose_tmp[17]

    print(part_pose)
    return part_pose


def extract_each_part(vertices, faces):
    labels = np.array(np.load('data/partnames_per_vertex.pkl'))
    body_part_names = list(set(labels))

    for part in set(labels):
        body_part = vertices[labels == part]
        from visualize_point_cloud import visualize_body_part
        print(part)
        visualize_body_part(body_part)
        raw_input("Press Enter")
        
