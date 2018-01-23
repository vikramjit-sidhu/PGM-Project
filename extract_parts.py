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
	label_size = len(label_names)
	num_pts,_ = X.shape
	for j in range(0,label_size):
		X_tmp = []
		for i in range(0,num_pts):
			if labels[i] == label_names[j]:
				X_tmp.append(X[i])
				ipdb.set_trace()
		centroid_p[j] = np.mean(X_tmp)
		print('mean ',np.mean(X_tmp))
	return centroid_p

def pose_part(pose_arr):
	part_pose = []
	for part_num in range(0,23):
		pose_tmp = pose_arr[part_num:part_num+3]
		part_pose.append(np.asarray(pose_tmp))                
		part_num = part_num + 3
	return part_pose

