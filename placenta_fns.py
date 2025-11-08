import sys
import math
import importlib

import numpy
import copy

import gmsh
import foronoi
from foronoi.contrib import ConcavePolygon

import placenta_fns as fns
import placenta_plots as plots
import placenta_const as const

importlib.reload(fns)
importlib.reload(plots)
importlib.reload(const)

from placenta_const import *

def cal_centroid(no_pts,pts):
    
    centroid = numpy.zeros(2)
    
    for i in range(0,no_pts):
        centroid = centroid+pts[i,:]
    
    centroid = centroid/no_pts
    
    return centroid

def cal_centre_polygon(no_vertices,vertices):
    """
    @brief Centre of mass of convex polygon based on subdividing polygon from centroid
    @param no_vertices Number of vertices of the polygon
    @param vertices Array of vertex coordinates
    @return Centroid coordinate of the polygon (2D)
    """
    polygon_area = 0.0
    weighted_sum = numpy.zeros(2)
    
    centroid = cal_centroid(no_vertices,vertices)
    
    tri_nodes = numpy.empty([3,2])
    tri_nodes[0,:] = centroid
    
    for tri_no in range(no_vertices-1,-1,-1):
        
        tri_nodes[1,:] = vertices[tri_no,:]
        if (tri_no > 0):
            tri_nodes[2,:] = vertices[tri_no-1,:]
        else:
            tri_nodes[2,:] = vertices[no_vertices-1,:]
        
        tri_centroid = cal_centroid(3,tri_nodes)
        
        tri_area = abs(0.5*( \
                        tri_nodes[0,0]*(tri_nodes[1,1]-tri_nodes[2,1]) + \
                        tri_nodes[1,0]*(tri_nodes[2,1]-tri_nodes[0,1]) + \
                        tri_nodes[2,0]*(tri_nodes[0,1]-tri_nodes[1,1]) \
                       ))
        
        polygon_area = polygon_area+tri_area
        weighted_sum = weighted_sum+tri_area*tri_centroid

    return (weighted_sum/polygon_area)

def uniform_points_on_circle(no_pts,r,offset=0.0):
    """
    @brief Place (no_pts) points uniformly on circle of radius r
    @param no_pts Number of points
    @param r Radius of circle
    @param offset Angular offset (radians)
    @return Array of points (no_pts x 2)
    """
    pts = numpy.zeros([no_pts,2])
    
    for pt_no in range(0,no_pts):
        theta = ((pt_no)/(no_pts))*2.0*math.pi+offset
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        pts[pt_no,:] = [x,y]
        
    return pts

def uniform_points_in_unit_circle_with_centre(no_pts,r):
    """
    @brief Test function - place no_pts-1 points at distance r from the centre, 1 point in centre
    @param no_pts Total number of points
    @param r Radius for outer points
    @return Array of points (no_pts x 2)
    """
    pts = numpy.zeros([no_pts,2])
    pts[0,:] = [0.0,0.0]
    
    for pt_no in range(1,no_pts):
        theta = ((pt_no-1)/(no_pts-1))*2.0*math.pi
        
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        pts[pt_no,:] = [x,y]
        
    return pts

def convert_foronoi_site(site):
    """
    @brief Convert foronoi object to arrays of vertices and edges
    @param site Foronoi site object
    @return [no_vertices,no_edges,vertices,edges]
    """
    no_vertices = len(site.vertices())
    no_edges = len(site.borders())

    vertices = numpy.zeros((no_vertices,2))
    edges = numpy.zeros((no_edges,2),dtype = int)

    first_edge = site.first_edge
    edge = first_edge

    edge_no = 0
    while edge_no < no_edges:

        origin = edge.origin.xy[:]
        target = edge.target.xy[:]
        edge_length = math.sqrt( \
            (origin[0] - target[0])**2 + \
            (origin[1] - target[1])**2)

        if (edge_length > tol):
            vertices[edge_no,:] = copy.deepcopy(origin)

            if (edge_no != no_edges-1):
                edges[edge_no,:] = \
                    (edge_no,edge_no+1)
            # Final edge - loop the vertex numbers back over
            else:
                edges[edge_no,:] = (edge_no,0)

            edge_no = edge_no+1
        else:
            vertices=numpy.delete(vertices,edge_no,0)
            edges=numpy.delete(edges,edge_no,0)
            no_vertices=no_vertices-1
            no_edges=no_edges-1
            # Safety check incase the last edge is being removed
            if (edge_no == no_edges):
                edges[edge_no-1,1] = 0

        edge = edge.next
        
    return [no_vertices,no_edges,vertices,edges]

def does_cell_intersect_circle(no_vertices,no_edges,vertices,edges,circle_r):
    """
    @brief Checks every edge and whether the 2 points lie on opposite sides of circle
    @details Returns True if intersects with 2 edges, False else. Assumes centre of circle is origin.
    @param no_vertices Number of vertices
    @param no_edges Number of edges
    @param vertices Vertex coordinates
    @param edges Edge connectivity array
    @param circle_r Circle radius
    @return [bool, intersection_edges]
    """
    sum_val = 0
    intersection_edges = numpy.zeros(2, dtype=int)
    
    for edge_no in range(0,no_edges):
        
        vertex_nos = numpy.array([edges[edge_no,0],edges[edge_no,1]])
        
        vertex_1 = vertices[vertex_nos[0],:]
        vertex_2 = vertices[vertex_nos[1],:]
        if ( \
                ((vertex_1[0]**2+vertex_1[1]**2)<=(circle_r-geom_tol)**2 and \
                   (vertex_2[0]**2+vertex_2[1]**2)>=(circle_r+geom_tol)**2) or \
                ((vertex_1[0]**2+vertex_1[1]**2)>=(circle_r+geom_tol)**2 and \
                   (vertex_2[0]**2+vertex_2[1]**2)<=(circle_r-geom_tol)**2) \
               ):
            sum_val = sum_val+1
            if (sum_val < 3):
                intersection_edges[sum_val-1]=edge_no
        
    if (sum_val == 2):
        #print("Obvious intersection")
        return [True,intersection_edges]
    elif (sum_val == 0):
        #print("Obvious NONintersection")
        return [False,intersection_edges]
    else:
        #print("Not obvious whether intersect or not")
        return [False,intersection_edges]

def check_orientation_of_polygon(nodes):
    """
    @brief Get oriented area of polygon then check if clockwise or anticlockwise
    @details Returns nodes reversed if they are clockwise so output is anticlockwise.
    @param nodes Array of polygon node coordinates (N x 2)
    @return nodes in anticlockwise order
    """
    if (nodes.shape[0] < 3):
        print("Error: check_orientation_of_polygon")
        print("Not enough nodes to form polygon")
        sys.exit(-1)
    elif (nodes.shape[1] != 2):
        print("Error: check_orientation_of_polygon")
        print("Nodes not in 2D, I'm not sure if extension to 3D works (might be fine)")
        sys.exit(-1)
            
    # Check if polygon is oriented clockwise or anticlockwise
    # If the polygon is oriented clockwise, the area will be negative
    # If the polygon is oriented anticlockwise, the area will be positive
    area = 0.0
    for i in range(0,nodes.shape[0]-1):
        area = area + (nodes[i,0]*nodes[i+1,1]-nodes[i+1,0]*nodes[i,1])
    area = area + (nodes[nodes.shape[0]-1,0]*nodes[0,1]-nodes[0,0]*nodes[nodes.shape[0]-1,1])
    area = 0.5*area
    
    reversed_nodes = copy.deepcopy(nodes)
    if (area < 0):
        reversed_nodes = reversed_nodes[::-1]
        
    return reversed_nodes            

def setup_standard_loop_ordering(no_vertices):
    """
    @brief [0,1],[1,2],[n,0], .. etc
    @param no_vertices Number of vertices
    @return loop array of shape (no_vertices,2)
    """
    loop = numpy.empty((no_vertices,2),dtype = int)
    
    for i in range(0,no_vertices - 1):
        loop[i,0] = i
        loop[i,1] = i + 1
        
    loop[no_vertices - 1,0] = no_vertices - 1
    loop[no_vertices - 1,1] = 0
    
    return loop

def add_vertex_after_idx_to_cell(add_idx,orig_no_vertices,orig_no_edges, \
        orig_vertices,orig_edges,pt_to_add):
    """
    @brief Add a vertex after add_idx
    @details Assumes [no_v,dim] index ordering - probably error if pass in array not like this
    @param add_idx Index after which to add vertex
    @param orig_no_vertices Original number of vertices
    @param orig_no_edges Original number of edges
    @param orig_vertices Original vertices array
    @param orig_edges Original edges array
    @param pt_to_add Point coordinates to add
    @return [no_vertices,no_edges,temp_vertices,temp_edges]
    """
    pt_dim = len(orig_vertices[0,:])
    
    no_vertices = orig_no_vertices + 1
    no_edges = orig_no_edges + 1
    
    temp_vertices = numpy.empty((no_vertices,pt_dim))
    temp_edges = numpy.empty((no_edges,2),dtype=int)
    
    if (len(pt_to_add) != pt_dim):
        print(f"Error: add_vertex_to_cell")
        print(f"Incompatible lengths")
        sys.exit(-1)
    
    if (add_idx == -1):
        print(f"Error: add_vertex_to_cell")
        print(f"add_idx = -1 not allowed due to 1st vertex needing to be intersection vertex")
        sys.exit(-1)
    elif (add_idx > orig_no_vertices - 1 or add_idx < -1):
        print(f"Error: add_vertex_to_cell")
        print(f"add_idx = {add_idx} is out of bounds, orig_no_vertices = {orig_no_vertices}")
        sys.exit(-1)
    elif (add_idx == orig_no_vertices - 1):
        temp_vertices[0:no_vertices - 1,:] = copy.deepcopy(orig_vertices)
        temp_edges[0:no_edges - 1,:] = copy.deepcopy(orig_edges)
        
        # Repoint previous start loop to newly created vertex
        temp_edges[no_edges - 2,1] = no_vertices - 1
        
        # Add new vertex, edge
        temp_vertices[no_vertices - 1,:] = copy.deepcopy(pt_to_add)
        temp_edges[no_edges - 1,:] = numpy.array([no_vertices - 1,0],dtype=int)
    else:
        temp_vertices[0:add_idx + 1,:] = copy.deepcopy(orig_vertices[0:add_idx + 1,:])
        
        temp_vertices[add_idx + 1,:] = copy.deepcopy(pt_to_add)
        temp_vertices[add_idx + 2:no_vertices,:] = copy.deepcopy(orig_vertices[add_idx + 1:orig_no_vertices,:])
        
        temp_edges = setup_standard_loop_ordering(no_vertices)
        
    return [no_vertices,no_edges,temp_vertices,temp_edges]
    
def reorder_vertices_first_edge_intersects(no_vertices,no_edges,vertices,edges, \
        circle_r):
    """
    @brief Assumes centre of circle is origin   
    @details Only vertices need to be changed here, edges by construction always s.t. e1 = [0,1], e2 = [1,2], ... 
    @param no_vertices Number of vertices
    @param no_edges Number of edges
    @param vertices Vertex coordinates
    @param edges Edge connectivity
    @param circle_r Circle radius
    @return [vertices,edges]
    """
    temp_vertices = numpy.zeros((no_vertices,2))
    temp_edges = numpy.zeros((no_edges,2),dtype=int)
    
    first_edge_index = -1
    for edge_index in range(0,no_edges):
        
        vertex_nos = numpy.array([edges[edge_index,0],edges[edge_index,1]])

        vertex_1 = vertices[vertex_nos[0],:]
        vertex_2 = vertices[vertex_nos[1],:]

        vertex_1_r = vertex_1[0]**2+vertex_1[1]**2
        vertex_2_r = vertex_2[0]**2+vertex_2[1]**2
        
        if ( (vertex_1_r<=(circle_r-geom_tol)**2 and \
              vertex_2_r>=(circle_r+geom_tol)**2) ):
            first_edge_index = edge_index
            break

    if (first_edge_index == no_edges-1):
        
        temp_vertices[1:no_vertices,:] = vertices[0:no_vertices-1,:]
        temp_vertices[0,:] = vertices[no_vertices-1,:]
        temp_edges[1:no_edges,:] = edges[0:no_edges-1,:]
        temp_edges[0,:] = edges[no_edges-1,:]
        vertices = numpy.copy(temp_vertices)
        #edges = numpy.copy(temp_edges)
        
    elif (first_edge_index > 0 and first_edge_index < no_edges-1):
        
        remaining_edges = no_edges-first_edge_index
        
        temp_vertices[0:remaining_edges,:] = vertices[first_edge_index:no_vertices,:]
        temp_vertices[remaining_edges:no_vertices,:] = vertices[0:first_edge_index,:]
        temp_edges[0:remaining_edges,:] = edges[first_edge_index:no_vertices,:]
        temp_edges[remaining_edges:no_vertices,:] = edges[0:first_edge_index,:]
        vertices = numpy.copy(temp_vertices)
        #edges = numpy.copy(temp_edges)
        
    if (first_edge_index == -1):
        print("ERROR: reorder_vertices_first_edge_intersects")
        print("first_edge_index == -1")
        sys.exit(-1)
    
    return [vertices,edges]


def del_vertices_between_intersection_vertices(no_vertices,no_edges,vertices,edges,circle_r):
    """
    @brief Assumes points have been reordered s.t. the edge intersection with origin in circle and target out circle is the first edge
    @details Deletes intermediate vertices between intersection vertices if necessary.
    @param no_vertices Number of vertices
    @param no_edges Number of edges
    @param vertices Vertex coordinates
    @param edges Edge connectivity
    @param circle_r Circle radius
    @return [no_vertices,no_edges,vertices,edges]
    """
    vertex_intersect_no = -1
    for edge_no in range(1,no_edges):
        
        vertex_nos = numpy.array([edges[edge_no,0],edges[edge_no,1]])
        
        vertex_1 = vertices[vertex_nos[0],:]
        vertex_2 = vertices[vertex_nos[1],:]
        
        vertex_1_r = vertex_1[0]**2+vertex_1[1]**2
        vertex_2_r = vertex_2[0]**2+vertex_2[1]**2
        
        if ( vertex_1_r>=(circle_r+geom_tol)**2 and \
                   vertex_2_r<=(circle_r-geom_tol)**2 \
                ): 
            # Vertex number of the vertex which is the origin of the edge going from out to in circle
            vertex_intersect_no = edge_no
            break
        
    # There's only vertices to remove if we're on edge index >= 3
    if (vertex_intersect_no > 2):
        
        print("del_vertices_between_intersection_vertices: removing points")
        print("This hasn't been properly tested, so if code dies it could be fault of this")
        
        for vertex_no in range(vertex_intersect_no-1,1,-1):
            
            vertices=numpy.delete(vertices,vertex_no,0)
            edges=numpy.delete(edges,vertex_no,0)
            no_vertices=no_vertices-1
            no_edges=no_edges-1
            
            edges[vertex_no:no_edges,:]=edges[vertex_no:no_edges,:]-1
            
    elif (vertex_intersect_no == -1):
        print("ERROR: del_vertices_between_intersection_vertices")
        print("vertex_intersect_no == -1")
        sys.exit(-1)

    edges[no_edges-1,1]=0

    return [no_vertices,no_edges,vertices,edges]

def shrink_outside_vertices(no_vertices,no_edges,vertices,edges,circle_r,intersection_edges):
    """
    @brief Assumes vertices ordered clockwise, only two vertices outside circle, circle centred (0,0)
    @details DO NOT USE INTERSECTION_EDGES FOR ARRAY INDEXING, entries haven't been shifted since reordering vertices.
             Intersection edge vertices are moved along their line. Intermediate ones are moved along the line between it and centre of circle (origin).
    @param no_vertices Number of vertices
    @param no_edges Number of edges
    @param vertices Vertex coordinates (modified in-place)
    @param edges Edge connectivity
    @param circle_r Circle radius
    @param intersection_edges Indices of intersection edges
    @return [vertices,edges,vertices_for_centroid]
    """
    vertices_for_centroid = numpy.copy(vertices)
    
    # Distance beyond circle_r which the vertices will be moved to
    circle_offset_r = placenta_voronoi_outer_radius_offset
    
    circle_combined_r = circle_r+circle_offset_r
    
    if (intersection_edges[0] == intersection_edges[1]):
        print("Warning: shrink_outside_vertices")
        print("Haven't implemented method to shrink single vertex outside circle")
        print("Skipping shrink")
        return [vertices,edges,vertices_for_centroid]
    
    for vertex_no in range(1,intersection_edges[1]+1):
        
        edge_vertices = numpy.zeros([2,2])
        if (intersection_edges[1]+1 == no_edges):
            end_vertex_no = 0
        else:
            end_vertex_no = vertex_no+1
        
        if (vertex_no == 1):
            edge_vertices[0,:] = vertices[vertex_no,:]
            edge_vertices[1,:] = vertices[vertex_no-1,:]
        elif vertex_no == intersection_edges[1]:
            edge_vertices[0,:] = vertices[vertex_no,:]
            edge_vertices[1,:] = vertices[end_vertex_no,:]
        else:
            edge_vertices[0,:] = vertices[vertex_no,:]
         
        # Check to make sure the vertex isn't already near circle_offset_r
        vertex_dist = edge_vertices[0,0]**2 + edge_vertices[0,1]**2
        
        # Intersection of line with circle
        dx = edge_vertices[0,0]-edge_vertices[1,0]
        dy = edge_vertices[0,1]-edge_vertices[1,1]
        dr = math.sqrt(dx**2+dy**2)
        D = edge_vertices[1,0]*edge_vertices[0,1]-edge_vertices[0,0]*edge_vertices[1,1]       
            
        ''' Old code for only moving the points inwards onto circle_combined_r, not out
        #Reduce point distance, else do nothing
        if (vertex_dist > circle_combined_r**2):
            #sqrt_term = math.sqrt( (circle_combined_r**2)*(dr**2)-D**2 )
            x_second_term = numpy.sign(dy)*dx*sqrt_term
            y_second_term = abs(dy)*sqrt_term
            x1=(D*dy+x_second_term)/(dr**2)
            y1=(-D*dx+y_second_term)/(dr**2)
            x2=(D*dy-x_second_term)/(dr**2)
            y2=(-D*dx-y_second_term)/(dr**2)
    
            if (dy >= 0):
                #model.occ.addPoint(x1,y1,0.0,0.1,curr_pt)
                vertices[vertex_no,0]=x1
                vertices[vertex_no,1]=y1
            else:
                vertices[vertex_no,0]=x2
                vertices[vertex_no,1]=y2
        '''

        # Shift point onto r+offset radius        
        square_term = (circle_combined_r**2)*(dr**2)-D**2
        sqrt_term = numpy.sign(square_term)* \
            math.sqrt( abs(square_term) )
        x_second_term = numpy.sign(dy)*dx*sqrt_term
        y_second_term = abs(dy)*sqrt_term
        x1=(D*dy+x_second_term)/(dr**2)
        y1=(-D*dx+y_second_term)/(dr**2)
        x2=(D*dy-x_second_term)/(dr**2)
        y2=(-D*dx-y_second_term)/(dr**2)

        if (dy >= 0):
            #model.occ.addPoint(x1,y1,0.0,0.1,curr_pt)
            vertices[vertex_no,0]=x1
            vertices[vertex_no,1]=y1
        else:
            vertices[vertex_no,0]=x2
            vertices[vertex_no,1]=y2

        # Do same but lying on point for purposes of calculating centroid
        # Intersection of line with circle
        sqrt_term = math.sqrt( (circle_r**2)*(dr**2)-D**2 )
        x_second_term = numpy.sign(dy)*dx*sqrt_term
        y_second_term = abs(dy)*sqrt_term
        x1=(D*dy+x_second_term)/(dr**2)
        y1=(-D*dx+y_second_term)/(dr**2)
        x2=(D*dy-x_second_term)/(dr**2)
        y2=(-D*dx-y_second_term)/(dr**2)

        if (dy >= 0):
            #model.occ.addPoint(x1,y1,0.0,0.1,curr_pt)
            vertices_for_centroid[vertex_no,0]=x1
            vertices_for_centroid[vertex_no,1]=y1
        else:
            vertices_for_centroid[vertex_no,0]=x2
            vertices_for_centroid[vertex_no,1]=y2
                    
    return [vertices,edges,vertices_for_centroid]

def find_line_with_centre(model,centre):
    """
    @brief Goes through lines and compares OCC COM to known stored centre to match IDs
    @param model Gmsh model
    @param centre 2D/3D centre to match
    @return curve number found
    """
    local_tol = 1.0e-0 / char_length
    total_curves = 0
    
    curves = []
    
    for i in model.occ.getEntities(1):
        mass = model.occ.getMass(i[0],i[1])
        if (mass > tol):
            if (numpy.linalg.norm(model.occ.getCenterOfMass(i[0],i[1]) - centre) < local_tol):
                if (total_curves > 0):
                    if (mass < curve_mass):
                        curve_no = i[1]
                        curve_mass = mass
                else:
                    curve_no = i[1]
                    curve_mass = mass
                total_curves = total_curves + 1
                curves.append(i[1])
      
    if (total_curves == 0):
        print(f"ERROR: no curve found with centre {centre}")
        sys.exit(-1)
    elif (total_curves > 1):
        print(f"WARNING - total no. of curves found with centre {centre} is {total_curves}")
        print(f"Curves found are {curves}")
    
    print(f"Found curve: {curve_no}")
    return curve_no

def find_surf_with_centre(model,centre):
    """
    @brief As find_line_with_centre but for surfaces
    @param model Gmsh model
    @param centre Surface centre to match
    @return surface number found
    """
    local_tol = 1.0e-0 / char_length
    total_surfs = 0
    
    surfs = []
    
    for i in model.occ.getEntities(2):
        mass = model.occ.getMass(i[0],i[1])
        if (mass > tol):
            if (numpy.linalg.norm(model.occ.getCenterOfMass(i[0],i[1]) - centre) < local_tol):
                if (total_surfs > 0):
                    if (mass < surf_mass):
                        surf_no = i[1]
                        surf_mass = mass
                else:
                    surf_no = i[1]
                    surf_mass = mass
                total_surfs = total_surfs + 1
                surfs.append(i[1])
      
    if (total_surfs == 0):
        print(f"ERROR: no curve found with centre {centre}")
        sys.exit(-1)
    elif (total_surfs > 1):
        print(f"WARNING - total no. of surfs found with centre {centre} is {total_surfs}")
        print(f"Surfs found are {surfs}")
    
    print(f"Found surf: {surf_no}")
    return surf_no


def lloyds_algorithm(no_polygon_vertices,polygon_vertices,no_cells):
    """
    @brief Vertices define the (convex) polygon. Initial Voronoi cells created by placing {no_cells} points around the centroid
    @details This works for a convex polytope. Runs Lloyd's algorithm to generate centroidal Voronoi tessellation within polygon.
    @param no_polygon_vertices Number of polygon vertices
    @param polygon_vertices Polygon vertices array
    @param no_cells Number of Voronoi cells/points
    @return Voronoi object v
    """
    err_tol = 1.0e-2
    err = 1.0e16
    iter_no = 1
    
    polygon_centroid = cal_centroid(no_polygon_vertices,polygon_vertices)
    
    # Find the distance of closest point to centroid
    minimum_dist = 1.0e16
    for vertex_no in range(0,no_polygon_vertices):
        dist = numpy.linalg.norm(polygon_centroid[:]-polygon_vertices[vertex_no,:])
        if (dist < minimum_dist):
            minimum_dist = dist
    
    # Just need this to be small enough that ball around centroid is inside polygon
    min_dist = minimum_dist / numpy.random.randint(2,10)
    #voronoi_pts = uniform_points_on_circle(no_cells,min_dist,offset=2.0)
    centroid = cal_centroid(no_polygon_vertices,polygon_vertices)
    voronoi_pts = numpy.zeros([no_cells,2])
    for i in range(0,no_cells):
        #voronoi_pts[i,:] = centroid+min_dist*numpy.random.rand()
        voronoi_pts[i,:] = [centroid[0]+min_dist*math.cos(2.0*math.pi*numpy.random.rand()), \
                            centroid[1]+min_dist*math.sin(2.0*math.pi*numpy.random.rand())]
    # Previous Voronoi points - need this for printing at the end at this is the actual generating set
    prev_voronoi_pts = copy.deepcopy(voronoi_pts)
    
    while (err > 1.0e-0):
        logger.debug(f"iteration {iter_no}")

        prev_pts = numpy.copy(voronoi_pts)
        # Need to repoint pgon_foronoi each time, it seems to get updated with points after create_diagram
        # Note ConcavePolygon comes from code on Github issues, some shapes can have disappearing edges
        if (lobule_foronoi_type == 'standard'):
            pgon = foronoi.Polygon(polygon_vertices)
        elif (lobule_foronoi_type == 'concave'):
            pgon = ConcavePolygon(polygon_vertices)
        else:
            print(f"Error: lloyds")
            print(f"Unrecognised lobule_foronoi_type: {lobule_foronoi_type}")
            sys.exit(-1)
        
        v = foronoi.Voronoi(pgon)
        prev_voronoi_pts = copy.deepcopy(voronoi_pts)
        try:
            v.create_diagram(points=voronoi_pts)
            voronoi_success = True
        except Exception:
            voronoi_success = False
        
        if (voronoi_success):
            # Check for empty cells
            for site_no in range(0,no_cells):
                if (len(v.sites[site_no].vertices()) < 3 or len(v.sites[site_no].borders()) < 3):
                    voronoi_success = False
                    break
        
        if (voronoi_success):
            for site_no in range(0,no_cells):
                [no_vertices,no_edges,vertices,edges] = convert_foronoi_site(v.sites[site_no])
                # try due to /0 error in weird formations
                try:
                    voronoi_pts[site_no,:] = cal_centre_polygon(no_vertices,vertices)
                    err = numpy.linalg.norm(voronoi_pts-prev_pts)
                    logger.debug(f"Error = {err}")
                except:
                    voronoi_success = False
                    err = math.isnan()
            
        if (voronoi_success and not(math.isnan(err))):
            iter_no = iter_no+1
        else:
            err = 1.0e16
            min_dist = minimum_dist / numpy.random.randint(2,11)
            centroid = cal_centroid(no_polygon_vertices,polygon_vertices)
            for i in range(0,no_cells):
                voronoi_pts[i,:] = centroid+min_dist*numpy.random.rand()
            logger.debug(f"Iterations failed, retrying with different initial points: \n" + \
                         f"{voronoi_pts}")

        if (iter_no > 200):
            break

        

        
        #frni_placentone = frni.ForonoiPlacentone(v.sites[placentone_no],placenta_radius)
    
    # Indent in -> plot all iterations, outer -> just final

    # frni.visualise_voronoi(v)    
    
    print(f"Error = {err}")
    print(f"Generating points given by: ")
    for pt_no in range(0,no_cells):
        if (pt_no < no_cells - 1):
            print(f"({prev_voronoi_pts[pt_no,0]}, {prev_voronoi_pts[pt_no,1]}),")
        else:
            print(f"({prev_voronoi_pts[pt_no,0]}, {prev_voronoi_pts[pt_no,1]})")
        
            
    
    return v


def create_oriented_box(model,z_lower_face_centre,x_orientation,y_orientation,z_orientation,xy_length,z_length,tag=None):
    """
    @brief Create an oriented box (hexahedron-like) in OCC using given centre and orientation vectors.
    @param model Gmsh model
    @param z_lower_face_centre Centre of lower face (3-vector)
    @param x_orientation X-axis unit vector
    @param y_orientation Y-axis unit vector
    @param z_orientation Z-axis unit vector
    @param xy_length Length of square face
    @param z_length Height of box
    @param tag Optional tag for volume
    @return model with added geometry
    """
    curr_pt = model.occ.getMaxTag(0)+1
    curr_line = model.occ.getMaxTag(1)+1
    curr_lineloop = model.occ.getMaxTag(-1)+1
    curr_surf = model.occ.getMaxTag(2)+1
    curr_surfloop = model.occ.getMaxTag(-2)+1
    if (tag==None):
        curr_vol=model.occ.getMaxTag(3)+1
    else:
        curr_vol=tag
            
    model.occ.addPoint(*(z_lower_face_centre-(xy_length/2.0)*x_orientation-(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt)
    model.occ.addPoint(*(z_lower_face_centre+(xy_length/2.0)*x_orientation-(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+1)
    model.occ.addPoint(*(z_lower_face_centre+(xy_length/2.0)*x_orientation+(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+2)
    model.occ.addPoint(*(z_lower_face_centre-(xy_length/2.0)*x_orientation+(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+3)
    model.occ.addPoint(*(z_lower_face_centre+z_length*z_orientation \
        -(xy_length/2.0)*x_orientation-(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+4)
    model.occ.addPoint(*(z_lower_face_centre+z_length*z_orientation \
        +(xy_length/2.0)*x_orientation-(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+5)
    model.occ.addPoint(*(z_lower_face_centre+z_length*z_orientation \
        +(xy_length/2.0)*x_orientation+(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+6)
    model.occ.addPoint(*(z_lower_face_centre+z_length*z_orientation \
        -(xy_length/2.0)*x_orientation+(xy_length/2.0)*y_orientation),meshSize=1.0,tag=curr_pt+7)
    
    # z-
    model.occ.addLine(curr_pt,curr_pt+1,tag=curr_line)
    model.occ.addLine(curr_pt+1,curr_pt+2,tag=curr_line+1)
    model.occ.addLine(curr_pt+2,curr_pt+3,tag=curr_line+2)
    model.occ.addLine(curr_pt+3,curr_pt,tag=curr_line+3)
    model.occ.addCurveLoop([curr_line,curr_line+1,curr_line+2,curr_line+3],tag=curr_lineloop)
    # z+
    model.occ.addLine(curr_pt+4,curr_pt+5,tag=curr_line+4)
    model.occ.addLine(curr_pt+5,curr_pt+6,tag=curr_line+5)
    model.occ.addLine(curr_pt+6,curr_pt+7,tag=curr_line+6)
    model.occ.addLine(curr_pt+7,curr_pt+4,tag=curr_line+7)
    model.occ.addCurveLoop([curr_line+4,curr_line+5,curr_line+6,curr_line+7],tag=curr_lineloop+1)
    # y-
    model.occ.addLine(curr_pt,curr_pt+4,tag=curr_line+8)
    model.occ.addLine(curr_pt+1,curr_pt+5,tag=curr_line+9)
    model.occ.addCurveLoop([curr_line,curr_line+9,curr_line+4,curr_line+8],tag=curr_lineloop+2)
    # y+
    model.occ.addLine(curr_pt+2,curr_pt+6,tag=curr_line+10)
    model.occ.addLine(curr_pt+3,curr_pt+7,tag=curr_line+11)
    model.occ.addCurveLoop([curr_line+2,curr_line+10,curr_line+6,curr_line+11],tag=curr_lineloop+3)
    # x-
    model.occ.addCurveLoop([curr_line+3,curr_line+8,curr_line+7,curr_line+11],tag=curr_lineloop+4)
    # x+
    model.occ.addCurveLoop([curr_line+1,curr_line+9,curr_line+5,curr_line+10],tag=curr_lineloop+5)
    
    model.occ.addPlaneSurface([curr_lineloop],curr_surf)
    model.occ.addPlaneSurface([curr_lineloop+1],curr_surf+1)
    model.occ.addPlaneSurface([curr_lineloop+2],curr_surf+2)
    model.occ.addPlaneSurface([curr_lineloop+3],curr_surf+3)
    model.occ.addPlaneSurface([curr_lineloop+4],curr_surf+4)
    model.occ.addPlaneSurface([curr_lineloop+5],curr_surf+5)
    
    model.occ.addSurfaceLoop([curr_surf],curr_surfloop)
    model.occ.addSurfaceLoop([curr_surf+1],curr_surfloop+1)
    model.occ.addSurfaceLoop([curr_surf+2],curr_surfloop+2)
    model.occ.addSurfaceLoop([curr_surf+3],curr_surfloop+3)
    model.occ.addSurfaceLoop([curr_surf+4],curr_surfloop+4)
    model.occ.addSurfaceLoop([curr_surf+5],curr_surfloop+5)
    
    model.occ.addVolume([curr_surfloop,curr_surfloop+1,curr_surfloop+2,curr_surfloop+3,curr_surfloop+4,curr_surfloop+5],curr_vol)   
    
    return model

def find_dist_from_plane(plane_point,plane_normal,space_point):
    """
    @brief Find signed distance from point to plane
    @param plane_point A point on the plane
    @param plane_normal Plane normal vector (assumed normalized)
    @param space_point Point to measure distance from
    @return Signed distance (float)
    """
    dist = numpy.dot(plane_normal,space_point-plane_point)

    return dist

def find_line_no_with_COM_closest_to_dist(model,fixed_dist,fixed_point):
    """
    @brief Find line number with center-of-mass closest to given distance from fixed_point
    @param model Gmsh model
    @param fixed_dist Target distance
    @param fixed_point Reference point
    @return line number
    """
    line_no_to_find = 0
    dist_away = 1.0e9
    
    for line_pair in model.occ.getEntities(1):
        line_no = line_pair[1]
        line_COM = model.occ.getCenterOfMass(1,line_no)
        dist = numpy.linalg.norm(fixed_point-line_COM)
        dist_diff = abs(fixed_dist-dist)
        if (dist_diff < dist_away):
            dist_away = dist_diff
            line_no_to_find = line_no
        
    print(line_no_to_find)    
    return line_no_to_find

def find_surf_no_with_COM_on_plane(model,plane_point,plane_normal):
    """
    @brief Find a surface whose center-of-mass lies on given plane
    @param model Gmsh model
    @param plane_point A point on plane
    @param plane_normal Plane normal vector
    @return surface number
    """
    no_surfs = 0
    surf_no_to_find = 0
    
    for surf_pair in model.occ.getEntities(2):
        surf_no = surf_pair[1]
        surf_COM = model.occ.getCenterOfMass(2,surf_no)
        dist = find_dist_from_plane(plane_point,plane_normal,surf_COM)
        
        if (dist < geom_tol and dist > -geom_tol):
            no_surfs = no_surfs+1
            surf_no_to_find = surf_no
            
    if (no_surfs == 0 or no_surfs > 1):
        print(f"Err: find_surf_no_with_COM_on_plane")
        print(f"no_surfs = {no_surfs}")
        sys.exit(-1)
        
    return surf_no_to_find


def find_lobule_no(point,lobules):
    """
    @brief Find lobule index containing given point
    @param point Query point
    @param lobules List of lobule objects
    @return lobule index or -1 if not found
    """
    for lobule_no,lobule in enumerate(lobules):
        
        no_edges = lobule.no_edges
        vertices = lobule.vertices
        edges = lobule.edges
        edge_counter = 0
        
        for edge_no in range(0,no_edges):
            
            point_on_line = copy.deepcopy(vertices[edges[edge_no,0]][0:2])
            end_point_on_line = copy.deepcopy(vertices[edges[edge_no,1]][0:2])
            
            tangent = numpy.empty(2)
            normal = numpy.empty(2)
            
            tangent[0:2] = end_point_on_line - point_on_line
            normal[0:2] = [tangent[1],-tangent[0]]
            normal[0:2] = normalise_vector(2,normal)
            
            if (find_dist_from_plane(point_on_line,normal,lobule.centroid[0:2]) < 0):
                logger.warn(f"WARN: find_lobule_no")
                logger.warn(f"normal . centroid < 0, shouldn't happen")
                logger.warn(f"Reversing normal sign")
                normal = -normal
                
            point_on_line[0:2] = point_on_line[0:2] + \
                normal[0:2] * lobule_wall_thickness/2.0
                
            if (find_dist_from_plane(point_on_line,normal,point[0:2]) > 0):
                edge_counter = edge_counter+1
                
        if (edge_counter == no_edges):
            return lobule_no
    
    return -1

def find_cotyledon_no(point,cotyledons):
    """
    @brief Find cotyledon index containing given point
    @param point Query point
    @param cotyledons List of cotyledon objects
    @return cotyledon index or -1
    """
    for cotyledon_no,cotyledon in enumerate(cotyledons):
        
        no_edges = cotyledon.no_edges
        vertices = cotyledon.vertices
        edges = cotyledon.edges
        edge_counter = 0
        
        for edge_no in range(0,no_edges):
            
            point_on_line = copy.deepcopy(vertices[edges[edge_no,0]][0:2])
            end_point_on_line = copy.deepcopy(vertices[edges[edge_no,1]][0:2])
            
            tangent = numpy.empty(2)
            normal = numpy.empty(2)
            
            tangent[0:2] = end_point_on_line - point_on_line
            normal[0:2] = [tangent[1],-tangent[0]]
            normal[0:2] = normalise_vector(2,normal)
            
            if (find_dist_from_plane(point_on_line,normal,cotyledon.centroid[0:2]) < 0):
                logger.warn(f"WARN: find_cotyledon_no")
                logger.warn(f"normal . centroid < 0, shouldn't happen")
                logger.warn(f"Reversing normal sign")
                normal = -normal
                
            point_on_line[0:2] = point_on_line[0:2] + \
                normal[0:2] * placentone_wall_thickness/2.0
                
            if (find_dist_from_plane(point_on_line,normal,point[0:2]) > 0):
                edge_counter = edge_counter+1
                
        if (edge_counter == no_edges):
            return cotyledon_no
    
    return -1


def swap_2d_array_indices(size_of_first_axis,size_of_second_axis,array_in):
    """
    @brief Swap 2D array indices (transpose-like) with bounds checks
    @param size_of_first_axis Expected first axis size
    @param size_of_second_axis Expected second axis size
    @param array_in Input array
    @return array_out Swapped array
    """
    if (size_of_first_axis < 1 or size_of_second_axis < 1):
        print("ERROR: swap_2d_array_indicies")
        print("size_of_first_axis < 1 or size_of_second_axis < 1")
        sys.exit(-2)
        
    if (len(array_in[:,0]) != size_of_first_axis or len(array_in[0,:]) != size_of_second_axis):
        print("ERROR: swap_2d_array_indicies")
        print("Incorrect array bounds")
        print(f"size_of_first_axis = {size_of_first_axis} but len(..[:,0]) = {len(array_in[:,0])}")
        print(f"size_of_second_axis = {size_of_second_axis} but len(..[0,:]) = {len(array_in[0,:])}")
        sys.exit(-2)
    
    array_out = numpy.empty([size_of_second_axis,size_of_first_axis])
    
    for first_index in range(0,size_of_first_axis):
        
        array_out[:,first_index] = array_in[first_index,:]
        
    return array_out







def quadratic_formula(a,b,c):
    """
    @brief Solve quadratic equation ax^2 + bx + c = 0
    @param a Coefficient a
    @param b Coefficient b
    @param c Coefficient c
    @return [soln_p, soln_m]
    """
    discriminant = b**2 - 4.0*a*c
    
    if (discriminant < 0):
        print(f"Err: quadratic_formula")
        print(f"Complex solution")
        sys.exit(-1)
    else:
        soln_p = (-b + math.sqrt(discriminant))/(2.0*a)
        soln_m = (-b - math.sqrt(discriminant))/(2.0*a)
    
    return [soln_p,soln_m]


def get_face_in_bb(bounding_box):
    """
    @brief Helper to get face inside bounding box using Gmsh OCC getEntitiesInBoundingBox
    @details Only to be used for cylinder faces, hence the check len(entities) == 1
    @param bounding_box Bounding box with xmin,xmax,ymin,ymax,zmin,zmax
    @return face_label Face tag
    """
    face = gmsh.model.occ.getEntitiesInBoundingBox( \
        bounding_box.xmin, bounding_box.ymin, bounding_box.zmin, \
        bounding_box.xmax, bounding_box.ymax, bounding_box.zmax, dim=2)
    print("face: ",face)
    
    if len(face) != 1:
        print("Error: get_face_in_bb")
        print("Number of entities in bounding box != 1")
        print("Number of entities in bounding box = ",len(face))
        #sys.exit(-1)
    
    face_entity = face[0]
    face_label = face_entity[1]
    
    return face_label

def normalise_vector(dim,vec):
    """
    @brief Normalise vector of given dimension
    @param dim Dimension to normalise over
    @param vec Vector array-like
    @return Normalised vector (0 if norm below tol)
    """
    tol = 1.0e-14
    
    norm_vec = numpy.copy(vec[0:dim])
    
    sq_sum = 0
    for i in range(0,dim):
        sq_sum = sq_sum + norm_vec[i]**2
    sq_sum = math.sqrt(sq_sum)
    
    if (sq_sum < tol):
        return norm_vec[0:dim]*0
    else:
        return norm_vec/sq_sum

def check_unique_vol(model):
    """
    @brief Checks that currently only one volume exists in model
    @param model Gmsh model
    @raises ValueError if more than one volume exists
    """
    vol_no = model.occ.getMaxTag(3)
    if (vol_no != 1):
        print(f"ERROR: check_unique_vol")
        raise ValueError(f"Error: number of volumes = {vol_no} when there should only be 1")
    
def TEST_fuse(dim_1,tag_1,dim_2,tag_2):
    """
    @brief Simple wrapper to fuse two OCC entities (for testing)
    """
    gmsh.model.occ.fuse([(dim_1,tag_1)],[(dim_2,tag_2)])
    return None

def points_distribution_inner_random(circle_radius, \
                                     no_pts,no_pts_outer,no_pts_inner, \
                                     inner_radius_bound,outer_angle_offset=0.0, \
                                     outer_radius_offset=0.0,outer_angle_variance=0.0):
    """
    @brief Generate points with mixed outer-ring and inner-random distributions
    @param circle_radius Outer circle radius
    @param no_pts Total number of points
    @param no_pts_outer Number of outer points
    @param no_pts_inner Number of inner points
    @param inner_radius_bound Inner radius bound
    @param outer_angle_offset Angular offset for outer points
    @param outer_radius_offset Radial offset for outer points
    @param outer_angle_variance Angular variance for outer points
    @return points_array (no_pts x 2)
    """
    if (no_pts_outer+no_pts_inner != no_pts):
        print("ERROR: points_distribution_inner_random")
        print("outer+inner pts != no_pts")
        sys.exit(-1)
    
    if (outer_angle_variance > tol):
        theta_rand = 1.0e10
        rng = numpy.random.default_rng()     
    else:
        theta_rand = 0.0
        rng = None
    
    points_array = numpy.empty([no_pts,2])
    
    total_radius = circle_radius+outer_radius_offset
    for pt_no in range(0,no_pts_outer):
        theta = outer_angle_offset+pt_no*(2.0*math.pi/no_pts_outer)
        if (rng != None):
            theta_rand = 1.0e10
            while (abs(theta_rand)>outer_angle_variance):
                theta_rand = copy.deepcopy(rng.normal(loc=0.0,scale=outer_angle_variance/2.0, size=None))
        print(f"theta_rand = ",theta_rand)
        x_pt = total_radius*math.cos(theta+theta_rand)
        y_pt = total_radius*math.sin(theta+theta_rand)
        
        points_array[pt_no,:] = (x_pt,y_pt)
        
        if(pt_no>1):
            print(f"points_array[pt_no,:] = {points_array[pt_no,:]} \n \
                points_array[pt_no-1,:] = {points_array[pt_no-1,:]}")
        
    for pt_no in range(0,no_pts_inner):
        r = inner_radius_bound*math.sqrt(numpy.random.random())
        theta = numpy.random.random()*2.0*math.pi
        x_pt = r*math.cos(theta)
        y_pt = r*math.sin(theta)
        
        points_array[no_pts_outer+pt_no,:] = (x_pt,y_pt)
    
    return points_array

def determine_max_nodes_from_cotyledons(cotyledon_list):
    """
    @brief Determine maximum vertex count among cotyledons
    @param cotyledon_list List of cotyledon objects
    @return max_no_nodes
    """
    max_no_nodes = 0
    
    for cell in cotyledon_list:
        
        if (cell.no_vertices > max_no_nodes):
            
            max_no_nodes = cell.no_vertices
            
    return max_no_nodes

def determine_max_edges_from_cotyledons(cotyledon_list):
    """
    @brief Determine maximum edge count among cotyledons
    @param cotyledon_list List of cotyledon objects
    @return max_no_edges
    """
    max_no_edges = 0
    
    for cell in cotyledon_list:
        
        if (cell.no_edges > max_no_edges):
            
            max_no_edges = cell.no_edges
            
    return max_no_edges

def determine_largest_gmsh_vol(model):
    """
    @brief Determine the largest Gmsh volume by mass
    @param model Gmsh model
    @return volume tag (int)
    """
    volumes = model.occ.getEntities(dim=3)
    
    vols = numpy.array([model.occ.getMass(3,pair[1]) for pair in volumes])
    
    index_no = numpy.argmax(vols)
    
    vol_pair = volumes[index_no]
    
    vol_no = vol_pair[1]
    
    return vol_no

def calc_septal_vein_xy(wall_pt_1,wall_pt_2,edge_dir,ratio):
    """
    @brief Gets septal vein xy using two wall bounding points and ratio along edge
    @details Uses buffer equal to septal_vein_radius + 1.2 * funnel_radius
    @param wall_pt_1 First wall point (2D)
    @param wall_pt_2 Second wall point (2D)
    @param edge_dir Unit direction along edge from pt1->pt2
    @param ratio Fraction along the available segment [0,1]
    @return 2D point on wall
    """
    buffer_on_wall = septal_vein_radius+1.2*septal_vein_funnel_radius
    
    twod_start_pt = numpy.copy(wall_pt_1) + buffer_on_wall*edge_dir
    twod_end_pt = numpy.copy(wall_pt_2) - buffer_on_wall*edge_dir

    twod_pt = (1.0-ratio)*twod_start_pt + ratio*twod_end_pt

    return twod_pt

def calc_septal_vein_height(rel_hgt,xy_on_wall,xy_in_wall,ratio):
    """
    @brief Gets septal vein height using bounding sphere surfaces and buffers
    @param rel_hgt Relative height to consider
    @param xy_on_wall XY on wall
    @param xy_in_wall XY inside wall
    @param ratio Linear interpolation parameter [0,1]
    @return height value
    """
    sso = sphere_surface_eval(
            *xy_on_wall,initial_sphere_radius,0.0)
    ssi = sphere_surface_eval(
            *xy_in_wall,initial_sphere_radius,0.0)

    btm_hgt = max( \
        sso + wall_vein_buffer_on_wall, \
        ssi + wall_vein_buffer_in_wall)
    top_hgt = min( \
        rel_hgt + sso - wall_vein_buffer_on_wall, \
        rel_hgt + ssi - wall_vein_buffer_in_wall)
    
    if (btm_hgt > top_hgt):
        print(f"ERROR: calc_septal_vein_height")
        print("btm_hgt > top_hgt")
        print(f"btm_hgt = {btm_hgt}, top_hgt = {top_hgt}")
        gmsh.model.occ.addPoint(*xy_on_wall,btm_hgt,meshSize=1.0)
        gmsh.model.occ.addPoint(*xy_on_wall,top_hgt,meshSize=1.0)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        gmsh.finalize()
        sys.exit(-1)
        
    hgt = (1.0-ratio)*btm_hgt + ratio*top_hgt
    
    print(f"IN SEPTAL_VEIN_HEIGHT")
    print(f"btm_hgt: {btm_hgt}")
    print(f"top_hgt: {top_hgt}")
    print(f"rel_hgt: {rel_hgt}")
    print(f"hgt: {hgt}")
    print(f"ratio: {ratio}")
    print(f"wall_vein_buffer_on_wall: {wall_vein_buffer_on_wall}")
    print(f"OUT SEPTAL_VEIN_HEIGHT")

    return hgt

def calc_septal_vein_height_o(cotyledon_no,xy_on_wall,xy_in_wall,ratio):
    """
    @brief Variant of calc_septal_vein_height using cotyledon-specific removal sphere radius
    @param cotyledon_no Index of cotyledon
    @param xy_on_wall XY on wall
    @param xy_in_wall XY inside wall
    @param ratio Interpolation parameter
    @return height value
    """
    rsr_c = removal_sphere_radius(cotyledon_wall_heights[cotyledon_no])
    
    sse = sphere_surface_eval(
            *xy_on_wall,rsr_c,cotyledon_wall_heights[cotyledon_no])

    btm_hgt = max( \
        sphere_surface_eval(*xy_on_wall,initial_sphere_radius,0.0)+wall_vein_buffer_on_wall, \
        sphere_surface_eval(*xy_in_wall,initial_sphere_radius,0.0)+wall_vein_buffer_in_wall)
    top_hgt = min( \
        sphere_surface_eval(
            *xy_on_wall,rsr_c,cotyledon_wall_heights[cotyledon_no])-wall_vein_buffer_on_wall, \
        sphere_surface_eval(
            *xy_in_wall,rsr_c,cotyledon_wall_heights[cotyledon_no])-wall_vein_buffer_in_wall)
    
    gmsh.model.occ.addPoint(*xy_in_wall,btm_hgt,meshSize=1.0)
    gmsh.model.occ.addPoint(*xy_on_wall,top_hgt,meshSize=1.0)
    
    if (btm_hgt > top_hgt):
        print(f"ERROR: calc_septal_vein_height")
        print("btm_hgt > top_hgt")
        print(f"btm_hgt = {btm_hgt}, top_hgt = {top_hgt}")
        gmsh.model.occ.addPoint(*xy_on_wall,btm_hgt,meshSize=1.0)
        gmsh.model.occ.addPoint(*xy_on_wall,top_hgt,meshSize=1.0)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()
        gmsh.finalize()
        sys.exit(-1)
        
    hgt = (1.0-ratio)*btm_hgt + ratio*top_hgt
    
    print(f"IN SEPTAL_VEIN_HEIGHT")
    print(f"cotyledon_no: {cotyledon_no}")
    print(f"btm_hgt: {btm_hgt}")
    print(f"top_hgt: {top_hgt}")
    print(f"hgt: {hgt}")
    print(f"ratio: {ratio}")
    print(f"sse: {sse}")
    print(f"wall h: {cotyledon_wall_heights[cotyledon_no]}")
    print(f"wall_vein_buffer_on_wall: {wall_vein_buffer_on_wall}")
    print(f"OUT SEPTAL_VEIN_HEIGHT")

    return hgt

def readjust_vertices_for_wall_veins(cotyledon_no, \
        initial_vertex1,initial_vertex2,radius,edge_no,edge_set):
    """
    @brief Adjust vertices for wall veins so that outer vertex is placed just outside circle and height can hold vein
    @details Uses bisection to move outer one along line - assumes only one of the two vertices is outside circle.
    @param cotyledon_no Cotyledon index (unused)
    @param initial_vertex1 First vertex (2D)
    @param initial_vertex2 Second vertex (2D)
    @param radius Circle radius
    @param edge_no Edge number in edge_set
    @param edge_set Edge set object with methods edge_length and calc_rel_height_along_edge
    @return [vertex1, vertex2] adjusted vertices (2D)
    """
    initial_length = edge_set.edge_length[edge_no]
    
    # Limitations of calculating rel_height outside radius
    radius_w_buffer = 0.9*radius

    wall_height = lambda ratio : \
        edge_set.calc_rel_height_along_edge(edge_no,ratio)

    if (circ_eval(*initial_vertex1) < radius**2 and \
            circ_eval(*initial_vertex2) < radius**2):
        return [initial_vertex1,initial_vertex2]
    
    iter = 0
    iter_limit = 200
    bisec_tol = 1.0e-1
    
    # v1 moving
    if (circ_eval(*initial_vertex1) >= radius_w_buffer**2):
        start_v = copy.deepcopy(initial_vertex2)
        end_v = copy.deepcopy(initial_vertex1)
    # v2 moving
    else:
        start_v = copy.deepcopy(initial_vertex1)
        end_v = copy.deepcopy(initial_vertex2)
        
    # initialise final_vertex
    final_vertex = copy.deepcopy(start_v)
    
    while (iter < iter_limit):
        temp_v = copy.deepcopy(0.5*(start_v+end_v))
        if (circ_eval(*temp_v) - radius_w_buffer**2 > bisec_tol):
            end_v = copy.deepcopy(temp_v)
        elif (circ_eval(*temp_v) - radius_w_buffer**2 < -bisec_tol):
            start_v = copy.deepcopy(temp_v)
        else:
            final_vertex = copy.deepcopy(temp_v)
            break
        iter = iter+1
        if (iter == iter_limit):
            final_vertex = copy.deepcopy(temp_v)
            print("WARNING: readjust_vertices_for_wall_veins")
            print("Hit iteration limit in x-y point move")
            
    #####################################################
    ## Height root finding via (slow) bisection method ##
    #####################################################

    # v1 moving
    if (circ_eval(*initial_vertex1) >= radius_w_buffer**2):
        vertex1 = copy.deepcopy(final_vertex)
        vertex2 = copy.deepcopy(initial_vertex2)
        
        v_length = lambda v : \
            numpy.linalg.norm( \
                copy.deepcopy(v-initial_vertex2))
        ratio = lambda v : \
            1.0 - v_length(v)/initial_length
        temp_v = copy.deepcopy(vertex1)
    # v2 moving
    else:
        vertex1 = copy.deepcopy(initial_vertex1)
        vertex2 = copy.deepcopy(final_vertex)
        
        v_length = lambda v : \
            numpy.linalg.norm( \
                copy.deepcopy(v-initial_vertex1))
        ratio = lambda v : \
            v_length(v)/initial_length
        temp_v = copy.deepcopy(vertex2)
        
    iter = 0
    bisec_tol = 1.0e-3
    err = 1.0
    
    # v1 moving
    if (circ_eval(*initial_vertex1) >= radius_w_buffer**2):
        start_v = copy.deepcopy(initial_vertex2)
        end_v = copy.deepcopy(initial_vertex1)
    # v2 moving
    else:
        start_v = copy.deepcopy(initial_vertex1)
        end_v = copy.deepcopy(initial_vertex2)
        
    # This says that the end point of the wall is less than can hold vein on so want to find a point s.t. the endpoint is just tall enough to hold a vein
    if (wall_height(ratio(temp_v)) - wall_vein_buffer_on_wall < -bisec_tol):
        while (iter < iter_limit):
            temp_v = copy.deepcopy(0.5*(start_v+end_v))
            if (wall_height(ratio(temp_v)) - wall_vein_buffer_on_wall > bisec_tol):
                start_v = temp_v
            elif (wall_height(ratio(temp_v)) - wall_vein_buffer_on_wall < -bisec_tol):
               end_v = temp_v
            else:
                final_vertex = copy.deepcopy(temp_v)
                break
            iter = iter+1
            if (iter == iter_limit):
                print("WARNING: readjust_vertices_for_wall_veins")
                print("Hit iteration limit in z point move")
                final_vertex = copy.deepcopy(temp_v)

        """print(f"final v: {final_vertex}")
        gmsh.model.occ.addPoint(*final_vertex,wall_height(ratio(final_vertex)),meshSize=1.0)
        gmsh.model.occ.synchronize()
        gmsh.fltk.run()"""
            
    # v1 moving
    if (circ_eval(*initial_vertex1) >= radius_w_buffer**2):
        vertex1 = copy.deepcopy(vertex2 - 0.8*(vertex2-final_vertex))
    # v2 moving
    else:
        vertex2 = copy.deepcopy(vertex1 - 0.8*(vertex1-final_vertex))
    
    return [vertex1,vertex2]

def check_septal_vein_overlap_lobule_walls( \
    face_cyl_end_pt, lobule_node_set, \
        radius,fillet,thickness):
    """
    @brief Check if septal vein overlaps lobule walls
    @param face_cyl_end_pt Endpoint of cylindrical face (3D)
    @param lobule_node_set Node set for lobule
    @param radius Septal vein radius
    @param fillet Fillet radius
    @param thickness Lobule wall thickness
    @return True if overlap detected, else False
    """
    for i in range(0,lobule_node_set.no_nodes):
        if (abs(face_cyl_end_pt[2] - lobule_node_set.nodal_wall_height[i]) > \
                radius + fillet):
            continue
        elif (numpy.linalg.norm(lobule_node_set.node[i,0:2] - face_cyl_end_pt[0:2]) < \
                radius + fillet + thickness):
            print(f"OVERLAP: check_septal_overlap_lobule_walls")
            print(f"lobule wall overlaps with septal vein")
            print(f"lobule wall pt: {lobule_node_set.node[i,0:2]}")
            print(f"septal vein pt: {face_cyl_end_pt[0:2]}")
            return True
    return False

def debug_add_lines_from_edge_set(model,edge_no,edge_set,wall_thickness):
    """
    @brief Debug helper to add lines from edge set for visualization
    """
    curr_pt = model.occ.getMaxTag(0)
    
    vertex_set = edge_set.node_set
    v_nos = copy.deepcopy(edge_set.edge[edge_no,:])
    v0 = vertex_set.node[v_nos[0],:]
    v0_height = vertex_set.node_height[v_nos[0]]
    v1 = vertex_set.node[v_nos[1],:]
    v1_height = vertex_set.node_height[v_nos[1]]
    
    edge_dir = edge_set.edge_dir[edge_no,:]
    
    model.occ.addPoint(v0[0],v0[1],0.0,meshSize=DomSize)
    model.occ.addPoint(v1[0],v1[1],0.0,meshSize=DomSize)
    model.occ.addLine(curr_pt+1,curr_pt+2)
    
    return None

def debug_add_base_wall_from_edge(model,edge_no,edge_set,wall_thickness):
    """
    @brief Debug helper to add base wall from edge (creates rectangle in XY plane)
    """
    curr_pt = model.occ.getMaxTag(0)
    
    h_w_t = wall_thickness / 2.0
    vertex_set = edge_set.node_set
    v_nos = copy.deepcopy(edge_set.edge[edge_no,:])
    v0 = vertex_set.node[v_nos[0],:]
    v0_height = vertex_set.node_height[v_nos[0]]
    v1 = vertex_set.node[v_nos[1],:]
    v1_height = vertex_set.node_height[v_nos[1]]
    
    edge_dir = edge_set.edge_dir[edge_no,:]
    normal_dir = numpy.array([-edge_dir[1],edge_dir[0]])
    
    # Bottom face
    for i in range(0,2):
        model.occ.addPoint( \
            *(v0 + (-1)**i * h_w_t * normal_dir),0.0, \
                meshSize=DomSize,tag = curr_pt + 1 + 2 * i )
        model.occ.addPoint( \
            *(v1 + (-1)**i * h_w_t * normal_dir),0.0, \
                meshSize=DomSize,tag = curr_pt + 2 + 2 * i )
    # Tangent lines
    model.occ.addLine(curr_pt+1,curr_pt+2)
    model.occ.addLine(curr_pt+3,curr_pt+4)
    # Normal lines
    model.occ.addLine(curr_pt+1,curr_pt+3)
    model.occ.addLine(curr_pt+2,curr_pt+4)
    
    return None

def create_piecewise_linear_wall_from_pts(model,v0,v1,v0_h,v1_h, \
        normal_dir,wall_thickness):
    """
    @brief Part of setting up septal walls - create piecewise linear wall from two points and heights
    @param model Gmsh model
    @param v0 Start vertex (2D)
    @param v1 End vertex (2D)
    @param v0_h Height at v0
    @param v1_h Height at v1
    @param normal_dir Normal direction vector (2D)
    @param wall_thickness Thickness of wall
    """
    face_pt = numpy.empty(2,dtype = int) # Track z-, z+ face pts
    face_line = numpy.empty(4,dtype = int) # Track z-, z+ face pts
    face_ll = numpy.empty(6,dtype = int) # Track z-, z+ face pts
    
    h_w_t = wall_thickness / 2.0
    
    curr_surf = model.occ.getMaxTag(2)
    curr_sl = model.occ.getMaxTag(-2)
    curr_vol = model.occ.getMaxTag(3)
    
    face_pt[0] = model.occ.getMaxTag(0)
    face_line[0] = model.occ.getMaxTag(1)
    face_ll[0] = model.occ.getMaxTag(-1)
    ## Bottom face ##
    for i in range(0,2):
        model.occ.addPoint( \
            *(v0 + (-1)**i * h_w_t * normal_dir),0.0, \
                meshSize=DomSize,tag = face_pt[0] + 1 + 2 * i)
        model.occ.addPoint( \
            *(v1 + (-1)**i * h_w_t * normal_dir),0.0, \
                meshSize=DomSize,tag = face_pt[0] + 2 + 2 * i)
    # Tangent lines
    model.occ.addLine(face_pt[0] + 1,face_pt[0] + 2)
    model.occ.addLine(face_pt[0] + 3,face_pt[0] + 4)
    # Normal lines
    model.occ.addLine(face_pt[0] + 1,face_pt[0] + 3)
    model.occ.addLine(face_pt[0] + 2,face_pt[0] + 4)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop( \
        [face_line[0] + 1,face_line[0] + 4,face_line[0] + 2,face_line[0] + 3])
    # Surf
    model.occ.addPlaneSurface([face_ll[0] + 1])
    
    ## Top face ##
    face_pt[1] = model.occ.getMaxTag(0)
    face_line[1] = model.occ.getMaxTag(1)
    face_ll[1] = model.occ.getMaxTag(-1)
    ## Side width face 1, 1 ##
    for i in range(0,2):
        model.occ.addPoint( \
            *(v0 + (-1)**i * h_w_t * normal_dir),v0_h, \
                meshSize=DomSize,tag = face_pt[1] + 1 + 2 * i)
        model.occ.addPoint( \
            *(v1 + (-1)**i * h_w_t * normal_dir),v1_h, \
                meshSize=DomSize,tag = face_pt[1] + 2 + 2 * i)
    # Tangent lines
    model.occ.addLine(face_pt[1] + 1,face_pt[1] + 2)
    model.occ.addLine(face_pt[1] + 3,face_pt[1] + 4)
    # Normal lines
    model.occ.addLine(face_pt[1] + 1,face_pt[1] + 3)
    model.occ.addLine(face_pt[1] + 2,face_pt[1] + 4)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop( \
        [face_line[1] + 1,face_line[1] + 4,face_line[1] + 2,face_line[1] + 3])
    # Surf
    model.occ.addPlaneSurface([face_ll[1] + 1])
    
    ## Width face, 1 ##
    face_line[2] = model.occ.getMaxTag(1)
    face_ll[2] = model.occ.getMaxTag(-1)
    # v0 line
    model.occ.addLine(face_pt[0] + 1,face_pt[1] + 1)
    # v1 line
    model.occ.addLine(face_pt[0] + 2,face_pt[1] + 2)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[0] + 1,face_line[2] + 1,face_line[1] + 1,face_line[2] + 2])
    # Surf
    model.occ.addPlaneSurface([face_ll[2] + 1])
    
    ## Width face, 2 ##
    face_line[3] = model.occ.getMaxTag(1)
    face_ll[3] = model.occ.getMaxTag(-1)
    # v0 line
    model.occ.addLine(face_pt[0] + 3,face_pt[1] + 3)
    # v1 line
    model.occ.addLine(face_pt[0] + 4,face_pt[1] + 4)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[0] + 2,face_line[3] + 1,face_line[1] + 2,face_line[3] + 2])
    # Surf
    model.occ.addPlaneSurface([face_ll[3] + 1])
    
    ## Lengthwise face, v0: 1 -> -1 ##
    face_ll[4] = model.occ.getMaxTag(-1)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[2] + 1,face_line[0] + 3,face_line[3] + 1,face_line[1] + 3])
    # Surf
    model.occ.addPlaneSurface([face_ll[4] + 1])
    
    ## Lengthwise face, v1: 1 -> -1 ##
    face_ll[5] = model.occ.getMaxTag(-1)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[2] + 2,face_line[1] + 4,face_line[3] + 2,face_line[0] + 4])
    # Surf
    model.occ.addPlaneSurface([face_ll[5] + 1])
    
    surf_list = [i for i in range(curr_surf + 1,curr_surf + 1 + 6)]
    model.occ.addSurfaceLoop(surf_list)
    
    model.occ.addVolume([curr_sl + 1])
    
    return None

def add_wall_from_edge_with_joint(model,edge_no,edge_set,wall_thickness) -> None:
    """
    @brief Part of setting up septal walls with an additional structure at the cell vertices
    @param model Gmsh model
    @param edge_no Edge number
    @param edge_set Edge set object
    @param wall_thickness Wall thickness
    """
    vertex_set = edge_set.node_set
    v_nos = copy.deepcopy(edge_set.edge[edge_no,:])
    v0 = vertex_set.node[v_nos[0],:]
    v0_h = vertex_set.nodal_wall_height[v_nos[0]]
    v1 = vertex_set.node[v_nos[1],:]
    v1_h = vertex_set.nodal_wall_height[v_nos[1]]
    
    edge_dir = edge_set.edge_dir[edge_no,:]
    edge_length = edge_set.edge_length[edge_no]
    normal_dir = numpy.array([-edge_dir[1],edge_dir[0]])
    
    # Set the transition pts away from v0, v1 where will be flat wall
    v0_trans = copy.deepcopy(edge_set.transition_point[edge_no,0,:])
    v1_trans = copy.deepcopy(edge_set.transition_point[edge_no,1,:])
    
    # Create start wall
    create_piecewise_linear_wall_from_pts(model,v0,v0_trans,v0_h,v0_h, \
        normal_dir,wall_thickness)
    # Create main wall
    create_piecewise_linear_wall_from_pts(model,v0_trans,v1_trans,v0_h,v1_h, \
        normal_dir,wall_thickness)
    # Craete end wall
    create_piecewise_linear_wall_from_pts(model,v1_trans,v1,v1_h,v1_h, \
        normal_dir,wall_thickness)    
        
    return None

def add_wall_from_edge(model,edge_no,edge_set,wall_thickness) -> None:
    """
    @brief Part of setting up septal walls (no joint) - creates wall volume along an edge
    @param model Gmsh model
    @param edge_no Edge number
    @param edge_set Edge set object
    @param wall_thickness Wall thickness
    """
    face_pt = numpy.empty(2,dtype = int) # Track z-, z+ face pts
    face_line = numpy.empty(4,dtype = int) # Track z-, z+ face pts
    face_ll = numpy.empty(6,dtype = int) # Track z-, z+ face pts
    
    h_w_t = wall_thickness / 2.0
    vertex_set = edge_set.node_set
    v_nos = copy.deepcopy(edge_set.edge[edge_no,:])
    v0 = vertex_set.node[v_nos[0],:]
    v0_height = vertex_set.vertex_set.nodal_wall_height[v_nos[0]]
    v1 = vertex_set.node[v_nos[1],:]
    v1_height = vertex_set.vertex_set.nodal_wall_height[v_nos[1]]
    
    edge_dir = edge_set.edge_dir[edge_no,:]
    normal_dir = numpy.array([-edge_dir[1],edge_dir[0]])
    
    curr_surf = model.occ.getMaxTag(2)
    curr_sl = model.occ.getMaxTag(-2)
    curr_vol = model.occ.getMaxTag(3)
    
    face_pt[0] = model.occ.getMaxTag(0)
    face_line[0] = model.occ.getMaxTag(1)
    face_ll[0] = model.occ.getMaxTag(-1)
    ## Bottom face ##
    for i in range(0,2):
        model.occ.addPoint( \
            *(v0 + (-1)**i * h_w_t * normal_dir),0.0, \
                meshSize=DomSize,tag = face_pt[0] + 1 + 2 * i)
        model.occ.addPoint( \
            *(v1 + (-1)**i * h_w_t * normal_dir),0.0, \
                meshSize=DomSize,tag = face_pt[0] + 2 + 2 * i)
    # Tangent lines
    model.occ.addLine(face_pt[0] + 1,face_pt[0] + 2)
    model.occ.addLine(face_pt[0] + 3,face_pt[0] + 4)
    # Normal lines
    model.occ.addLine(face_pt[0] + 1,face_pt[0] + 3)
    model.occ.addLine(face_pt[0] + 2,face_pt[0] + 4)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop( \
        [face_line[0] + 1,face_line[0] + 4,face_line[0] + 2,face_line[0] + 3])
    # Surf
    model.occ.addPlaneSurface([face_ll[0] + 1])
    
    ## Top face ##
    face_pt[1] = model.occ.getMaxTag(0)
    face_line[1] = model.occ.getMaxTag(1)
    face_ll[1] = model.occ.getMaxTag(-1)
    ## Side width face 1, 1 ##
    for i in range(0,2):
        model.occ.addPoint( \
            *(v0 + (-1)**i * h_w_t * normal_dir),v0_height, \
                meshSize=DomSize,tag = face_pt[1] + 1 + 2 * i)
        model.occ.addPoint( \
            *(v1 + (-1)**i * h_w_t * normal_dir),v1_height, \
                meshSize=DomSize,tag = face_pt[1] + 2 + 2 * i)
    # Tangent lines
    model.occ.addLine(face_pt[1] + 1,face_pt[1] + 2)
    model.occ.addLine(face_pt[1] + 3,face_pt[1] + 4)
    # Normal lines
    model.occ.addLine(face_pt[1] + 1,face_pt[1] + 3)
    model.occ.addLine(face_pt[1] + 2,face_pt[1] + 4)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop( \
        [face_line[1] + 1,face_line[1] + 4,face_line[1] + 2,face_line[1] + 3])
    # Surf
    model.occ.addPlaneSurface([face_ll[1] + 1])
    
    ## Width face, 1 ##
    face_line[2] = model.occ.getMaxTag(1)
    face_ll[2] = model.occ.getMaxTag(-1)
    # v0 line
    model.occ.addLine(face_pt[0] + 1,face_pt[1] + 1)
    # v1 line
    model.occ.addLine(face_pt[0] + 2,face_pt[1] + 2)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[0] + 1,face_line[2] + 1,face_line[1] + 1,face_line[2] + 2])
    # Surf
    model.occ.addPlaneSurface([face_ll[2] + 1])
    
    ## Width face, 2 ##
    face_line[3] = model.occ.getMaxTag(1)
    face_ll[3] = model.occ.getMaxTag(-1)
    # v0 line
    model.occ.addLine(face_pt[0] + 3,face_pt[1] + 3)
    # v1 line
    model.occ.addLine(face_pt[0] + 4,face_pt[1] + 4)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[0] + 2,face_line[3] + 1,face_line[1] + 2,face_line[3] + 2])
    # Surf
    model.occ.addPlaneSurface([face_ll[3] + 1])
    
    ## Lengthwise face, v0: 1 -> -1 ##
    face_ll[4] = model.occ.getMaxTag(-1)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[2] + 1,face_line[0] + 3,face_line[3] + 1,face_line[1] + 3])
    # Surf
    model.occ.addPlaneSurface([face_ll[4] + 1])
    
    ## Lengthwise face, v1: 1 -> -1 ##
    face_ll[5] = model.occ.getMaxTag(-1)
    # Curve loop (clockwise from interior?)
    model.occ.addCurveLoop([face_line[2] + 2,face_line[1] + 4,face_line[3] + 2,face_line[0] + 4])
    # Surf
    model.occ.addPlaneSurface([face_ll[5] + 1])
    
    surf_list = [i for i in range(curr_surf + 1,curr_surf + 1 + 6)]
    model.occ.addSurfaceLoop(surf_list)
    
    model.occ.addVolume([curr_sl + 1])
    
    return None

def add_joint_from_node(model,node_no,node_set,wall_thickness) -> None:
    """
    @brief Only sets up the joint structure at the cell vertices
    @param model Gmsh model
    @param node_no Node number
    @param node_set Node set object
    @param wall_thickness Wall thickness
    """
    curr_vol = model.occ.getMaxTag(3)
    
    v_loc = copy.deepcopy(node_set.node[node_no,:])
    v_z = node_set.vertex_set.nodal_wall_height[node_no]
    
    # Find distance to bottom wall
    p_height = sphere_surface_eval(*v_loc[0:2],initial_sphere_radius,0.0)
    p_height = v_z - p_height
    
    cyl_dz = p_height/2.0
    
    model.occ.addCylinder(*v_loc,v_z,0.0,0.0,-v_z,wall_thickness)
    model.occ.addSphere(*initial_sphere_centre,initial_sphere_radius)
    model.occ.intersect([(3,curr_vol+1)],[(3,curr_vol+2)])
    
    return None
    
def points_near(pt1,pt2):
    """
    @brief Check if two points are near each other within num_err tolerance
    @param pt1 First point
    @param pt2 Second point
    @return True if distance <= num_err, else False
    """
    err = numpy.linalg.norm(pt1[:]-pt2[:])
    if (err <= num_err):
        return True
    else:
        return False
    
def calc_sphere_height_at_xy(xy_pt):
    """
    @brief Height of spherical cap bottom surface relative to z = 0
    @param xy_pt 2D point
    @return height value
    """
    return sphere_surface_eval(*xy_pt,initial_sphere_radius,0.0)

def calc_sphere_interior_height_at_xy(xy_pt):
    """
    @brief Height of interior z distance
    @param xy_pt 2D point
    @return interior height (placenta_height - sphere height)
    """
    return (placenta_height - calc_sphere_height_at_xy(xy_pt))

def calc_sphere_interior_height_frac_at_xy(xy_pt):
    """
    @brief Fractional interior height at xy relative to placenta_height
    @param xy_pt 2D point
    @return Fraction between 0 and 1
    """
    return (1.0 - calc_sphere_height_at_xy(xy_pt)/placenta_height)

def convert_float_to_gmsh_field_str(value: float) -> str:
    """
    @brief Convert float to string suitable for gmsh field expressions, parenthesise negative values
    @param value Float value
    @return String
    """
    if (value<0.0):
        value = f"({value})"
    else:
        value = f"{value}"
    return value

def replace_multiple_signs_in_str(old_str):
    """
    @brief Replace multiple signs in a string when passing to Gmsh field expressions
    @param old_str Input string
    @return Cleaned string
    """
    new_str = old_str.replace("++","+")
    new_str = new_str.replace("+-","-")
    new_str = new_str.replace("-+","-")
    new_str = new_str.replace("--","+")    
    new_str = new_str.replace("+ +","+")
    new_str = new_str.replace("+ -","-")
    new_str = new_str.replace("- +","-")
    new_str = new_str.replace("- -","+")
    
    return new_str