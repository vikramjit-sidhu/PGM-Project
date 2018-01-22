import numpy as np

from extract_parts import extract_each_part

def show_point_cloud(verts, faces):
    from visualize_point_cloud import visualize_point_cloud
    visualize_point_cloud(verts, faces)
    raw_input("Press Enter to Exit")


if __name__ == "__main__":
    from get_body_model import load_smpl_body_model
    vertices, faces = load_smpl_body_model()
    extract_each_part(vertices, faces)