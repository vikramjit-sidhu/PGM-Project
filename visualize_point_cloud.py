def visualize_point_cloud(X, centroid,faces=None, mv=None):
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
    line1 = points3d(centroid[0][0], centroid[0][1], centroid[0][2], scale_factor=0.02, figure=fig)
    fig.scene.reset_zoom()
