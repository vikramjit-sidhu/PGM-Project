import numpy as np

def show_point_cloud(verts, faces):
    from visualize_point_cloud import visualize_point_cloud
    visualize_point_cloud(verts, faces)
    raw_input("Press Enter to Exit")


if __name__ == "__main__":
    from get_body_model import load_smpl_body_model
    vertices, faces = load_smpl_body_model()
    print(vertices.shape)
    show_point_cloud(vertices, faces)
    extract_each_part()
