
def extract_each_part():
    labels = np.load('data/partnames_per_vertex.pkl')
    body_part_names = list(set(labels))
    verts_body_part = []
    # for boxdy_part_name in body_part_names:
