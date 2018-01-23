import numpy as np

<<<<<<< HEAD
=======
from extract_parts import extract_each_part

>>>>>>> c308081824a852ddd377db727c779f5cd0cbc3ca
def show_point_cloud(verts, faces):
    from visualize_point_cloud import visualize_point_cloud
    visualize_point_cloud(verts, faces)
    raw_input("Press Enter to Exit")


if __name__ == "__main__":
    from get_body_model import load_smpl_body_model
    vertices, faces = load_smpl_body_model()
<<<<<<< HEAD
    print(vertices.shape)
    show_point_cloud(vertices, faces)
    extract_each_part()
=======
    extract_each_part(vertices, faces)
>>>>>>> c308081824a852ddd377db727c779f5cd0cbc3ca
