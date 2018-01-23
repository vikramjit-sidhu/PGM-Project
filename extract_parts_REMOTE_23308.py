
import numpy as np

def extract_each_part(vertices, faces):
    labels = np.array(np.load('data/partnames_per_vertex.pkl'))
    body_part_names = list(set(labels))

    for part in set(labels):
        body_part = vertices[labels == part]
        from visualize_point_cloud import visualize_body_part
        # print(part)
        # visualize_body_part(body_part)
        # raw_input("Press Enter")
        