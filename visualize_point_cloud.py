
def visualize_point_cloud(X, faces=None, mv=None):
    """
    Visualize a point cloud, X

    Inputs:
    ------------
    X
        matrix of the point cloud of dimension Nx3

    faces 
        if different than None is a 3xN matrix indicating which verts are connected

    mv
        MeshViewer

    Outputs
    ------------
    Mesh visualization
    """
    from mayavi.mlab import triangular_mesh,points3d,figure,clf

    if not mv:
        mv = figure(size=(800,800))
    fig = mv
    clf(fig)
    verts1 = X.T
    tm1 = triangular_mesh(verts1[0], verts1[1], verts1[2], faces, color=(.7, .7, .9), figure=fig)
    # line1 = points3d(centroid[0][0], centroid[0][1], centroid[0][2], scale_factor=0.02, figure=fig)
    fig.scene.reset_zoom()


def visualize_body_part(part):
    """
    Visualize a point cloud which represents a body part

    Inputs:
    ------------
    part
        Nx3 numpy array containing point locations of body part

    Outputs
    ------------
    Body Part visualization
    
    """
    from mayavi.mlab import points3d, figure
    fig = figure(size=(800, 800))
    vertices = part.T
    part_fig = points3d(vertices[0], vertices[1], vertices[2], color=(.7, .7, .9), figure=fig)
    fig.scene.reset_zoom()
