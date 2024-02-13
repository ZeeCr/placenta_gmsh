def test_edge_set() -> None:
    
    print(f"curr_pt: {model.occ.getMaxTag(0)}")
    v_no = cotyledon_edge_set.edge[0,:]
    print(f"nodal height 1: {cotyledon_edge_set.node_set.nodal_wall_height[v_no[0]]}")
    print(f"nodal height 2: {cotyledon_edge_set.node_set.nodal_wall_height[v_no[1]]}")
    print(f"node heights: {cotyledon_edge_set.node_set.nodal_wall_height[cotyledon_edge_set.edge[0,:]]}")
    v0 = cotyledon_edge_set.node_set.node[v_no[0],:]
    v1 = cotyledon_edge_set.node_set.node[v_no[1],:]
    model.occ.addPoint(*v0,5.0,meshSize = 1.0)
    model.occ.addPoint(*v1,5.0,meshSize = 1.0)

    # Check local -> global edge numbering gives right vertices
    [vp0,vp1] = cotyledon_edge_set.get_vertices_from_cell_edge(0,0)
    model.occ.addPoint(*vp0,6.0,meshSize = 1.0)
    model.occ.addPoint(*vp1,6.0,meshSize = 1.0)

    # Check ratio at 0, 1 is right
    xy = cotyledon_edge_set.calc_pt_along_edge(0,0.0)
    hgt = cotyledon_edge_set.calc_rel_height_along_edge(0,0.0)
    print(f"hgt at ratio 0.0: {hgt}")
    model.occ.addPoint(*xy,7.0,meshSize=1.0)
    xy = cotyledon_edge_set.calc_pt_along_edge(0,1.0)
    hgt = cotyledon_edge_set.calc_rel_height_along_edge(0,1.0)
    print(f"hgt at ratio 1.0: {hgt}")
    model.occ.addPoint(*xy,7.0,meshSize=1.0)


    # First check rel wall height of v_no[0] is right, then nodal
    xy = cotyledon_edge_set.calc_pt_along_edge(0,0.1)
    rel_hgt = cotyledon_edge_set.node_set.rel_wall_height[v_no[0]]
    sse = sphere_surface_eval(*cotyledon_edge_set.calc_pt_along_edge(0,0.0),initial_sphere_radius,0.0)
    model.occ.addPoint(*xy,sse + rel_hgt,meshSize=1.0)

    xy = cotyledon_edge_set.calc_pt_along_edge(0,0.2)
    node_hgt = cotyledon_edge_set.node_set.nodal_wall_height[v_no[0]]
    model.occ.addPoint(*xy,node_hgt,meshSize=1.0)

    r = 0.15
    xy = cotyledon_edge_set.calc_pt_along_edge(0,r)
    hgt = cotyledon_edge_set.calc_rel_height_along_edge(0,r)
    sse = sphere_surface_eval(*xy,initial_sphere_radius,0.0)
    model.occ.addPoint(*xy,sse,meshSize=1.0)
    model.occ.addPoint(*xy,sse+hgt,meshSize=1.0)


    model.occ.synchronize()
    gmsh.fltk.run()
    gmsh.finalize()
    sys.exit(-1)