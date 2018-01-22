
import numpy as np
from visualize_point_cloud import visualize_body_part

def extract_each_part(vertices, faces):
    labels = np.array(np.load('data/partnames_per_vertex.pkl'))
    body_part_names = list(set(labels))

    for part in set(labels):
        body_part = vertices[labels == part]