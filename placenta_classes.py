import sys
"""!
@file placenta_classes.py
@brief Data structures for placenta geometry generation using GMSH.
This module provides classes to represent 
placenta geometry objects, bounding boxes, faces, (ellipsoid-shaped) central 
cavities and global node/edge structures.
""" 
import math
import copy

import numpy

import placenta_fns as fns

from placenta_const import circ_eval

class Placentone:
    """! @brief Class for a single (convex) polygon (representing cotyledon/lobule) and its labels.
    Members:
    - int no_vertices: number of local polygon vertices.
    - int no_edges: number of local edges (typically equals no_vertices for closed polygons).
    - numpy.ndarray vertices: shape (no_vertices, 2) storing local XY coordinates.
    - numpy.ndarray edges: shape (no_edges, 2) storing local vertex index pairs that form edges.
    - numpy.ndarray centroid: 2-element array giving a reference centroid (not automatically updated).
    - numpy.ndarray global_vertex_nos: length no_vertices map local -> global (GMSH) vertex ids.
    - numpy.ndarray global_edge_nos: length no_edges map local -> global (GMSH) edge ids.
    - int lineloop_no: optional GMSH line loop id.
    - int surface_no: optional GMSH surface id.
    - int volume_no: optional GMSH volume id.
    - int boundary_cell: flag indicating boundary status (0 interior, non-zero boundary).
    Methods:
    - Placentone(no_vertices, no_edges)
        Constructor that allocates arrays for vertices, edges and mappings.
    - create_copy() -> Placentone
        Create and return a shallow copy of the Placentone with duplicated array data.
        The new instance contains copies of arrays so modifications to the copy will not
        change the original arrays.
    - shrink_placentone_fixed_dist(dist, model) -> None
        Move each vertex a fixed distance along the inward bisector at the vertex (local
        inward displacement). The centroid field is NOT updated by this operation; callers
        that require an accurate centroid must recompute it after shrinking.
    Parameters:
        - dist (float): signed distance to move each vertex (positive moves outward along
            the computed bisector direction, negative moves inward depending on convention).
        - model (object): kept for API compatibility with calling code; not used internally.
    Notes:
        - Uses external helper functions (e.g., normalise_vector) to compute unit directions.
        - Preserves local vertex indexing and edge topology; only the 2D coordinates are modified.
    - get_vertices_edges() -> [no_vertices, no_edges, vertices, edges]
        Return a compact tuple/list containing local counts and arrays needed by other routines.
    """
    def __init__(self,no_vertices,no_edges) -> None:
        self.no_vertices = no_vertices
        self.no_edges = no_edges
        
        # Edge vertex numbers are stored by local vertex numbering
        self.vertices = numpy.zeros((no_vertices,2))
        self.edges = numpy.zeros((no_edges,2),dtype=int)
        self.centroid = numpy.zeros(2)
        
        # Local to global (GMSH) label map
        self.global_vertex_nos = numpy.zeros((no_vertices),dtype=int)
        self.global_edge_nos = numpy.zeros((no_edges),dtype=int)
        
        # Store GMSH labels
        self.lineloop_no = 0
        self.surface_no = 0
        self.volume_no = 0
        
        # Store where it's an interior placentone or boundary
        self.boundary_cell = 0
        
    def create_copy(self):
        
        copy = Placentone(self.no_vertices,self.no_edges)
        copy.vertices = numpy.ndarray.copy(self.vertices)
        copy.edges = numpy.ndarray.copy(self.edges)
        copy.centroid = self.centroid
        
        copy.global_vertex_nos = numpy.ndarray.copy(self.global_vertex_nos)
        copy.global_edge_nos = numpy.ndarray.copy(self.global_edge_nos)

        copy.lineloop_no = self.lineloop_no
        copy.surface_no = self.surface_no
        copy.volume_no = self.volume_no
        
        return copy
        
    # Reduce vertices' distance from centroid by fixed value 
    # NOTE centroid does not give the actual centroid of this shrunk placentone anymore
    # To-do add function to recalculate on-demand
    def shrink_placentone_fixed_dist(self,dist,model) -> None:
        
        orig_vertices = numpy.copy(self.vertices)
        
        for vertex_no in range(0,self.no_vertices):
            
            if (vertex_no == 0):
                a = numpy.copy(orig_vertices[0,:]-orig_vertices[self.no_vertices-1,:])
                b = numpy.copy(orig_vertices[1,:]-orig_vertices[0,:])
            elif (vertex_no == self.no_vertices-1):
                a = numpy.copy(orig_vertices[self.no_vertices-1,:]-orig_vertices[self.no_vertices-2,:])
                b = numpy.copy(orig_vertices[0,:]-orig_vertices[self.no_vertices-1,:])
            else:
                a = numpy.copy(orig_vertices[vertex_no,:]-orig_vertices[vertex_no-1,:])
                b = numpy.copy(orig_vertices[vertex_no+1,:]-orig_vertices[vertex_no,:])
                
            a_unit = fns.normalise_vector(2,a)
            b_unit = fns.normalise_vector(2,b)
            angle = math.acos(numpy.dot(-a_unit,b_unit))
            
            cosA = numpy.cos(-angle/2.0)
            sinA = numpy.sin(-angle/2.0)
            c_x = cosA*b_unit[0]-sinA*b_unit[1] #\
                     #-cosA*orig_vertices[vertex_no,0]+sinA*orig_vertices[vertex_no,1]
            c_y = sinA*b_unit[0]+cosA*b_unit[1] #\
                     #-sinA*orig_vertices[vertex_no,0]+cosA*orig_vertices[vertex_no,1]
            c = numpy.array([c_x,c_y])
            c_unit = fns.normalise_vector(2,c)
            self.vertices[vertex_no,:] = orig_vertices[vertex_no,:]+dist*numpy.copy(c_unit)
            
        return None
            
    def get_vertices_edges(self):
        
        no_vertices = self.no_vertices
        no_edges = self.no_edges
        vertices = self.vertices
        edges = self.edges
        
        return [no_vertices,no_edges,vertices,edges]

        
class BoundingBox:
    """! @brief Axis-aligned bounding box with optional association to geometry entity identifiers.
    Members:
    - float xmin, ymin, zmin, xmax, ymax, zmax: extents of the box.
    - tuple bound_tuple: (xmin, ymin, zmin, xmax, ymax, zmax).
    - str entity_type: textual label for the contained entities (e.g., "surface", "volume").
    - numpy.ndarray entity_array: integer array of entity ids associated with this box.
    Methods:
    - BoundingBox()
    Constructor initialises bounds and placeholders.
    - update_bounds(bb_arr) -> None
    Set the bounding extents using a 6-element iterable (xmin, ymin, zmin, xmax, ymax, zmax).
    - update_entities(entity_type="none", entity_array=-1) -> None
    Store associated entity metadata. A copy of entity_array is stored.
    - print_bounds() -> None
    Print a human-readable representation of the extents (debug helper).
    - print_entities() -> None
    Print a human-readable representation of stored entity metadata (debug helper).
    """
    def __init__(self) -> None:
        self.xmin = 0.0
        self.ymin = 0.0
        self.zmin = 0.0
        self.xmax = 0.0
        self.ymax = 0.0
        self.zmax = 0.0
        self.bound_tuple = (0.0,0.0,0.0,0.0,0.0,0.0)
        
        self.entity_type = "none"
        self.entity_array = -1
        
    def update_bounds(self,bb_arr) -> None:
        self.xmin = bb_arr[0]
        self.ymin = bb_arr[1]
        self.zmin = bb_arr[2]
        self.xmax = bb_arr[3]
        self.ymax = bb_arr[4]
        self.zmax = bb_arr[5]
        self.bound_tuple = (self.xmin,self.ymin,self.zmin,self.xmax,self.ymax,self.zmax)
        
    def update_entities(self,entity_type="none",entity_array=-1) -> None:
        self.entity_type = entity_type
        self.entity_array = numpy.copy(entity_array)
        
    def print_bounds(self) -> None:
        print("xmin: ",self.xmin)
        print("ymin: ",self.ymin)
        print("zmin: ",self.zmin)
        print("xmax: ",self.xmax)
        print("ymax: ",self.ymax)
        print("zmax: ",self.zmax)
        
    def print_entities(self) -> None:
        print("entity_type: ",self.entity_type)
        print("entity_array: ",self.entity_array)
        
class Face:
    def __init__(self) -> None:
        self.apto_bdry_no = None
        self.face_no = None
        self.centre = None
        self.outward_unit_normal = None
        
        # These are used to store info about the original generating cylinders, e.g. marginal sinus veins which get chopped in half and have different centre
        self.cylinder_radius = None
        self.cylinder_centre = None
        self.cylinder_length = None
        self.cylinder_fillet = None
        
        self.vessel_type = None
        
    def update_face(self,face_no=None,centre=None,outward_unit_normal=None,vessel_type=None) -> None:
        if (face_no is not None):
            self.face_no = face_no
        if (centre is not None):
            self.centre = numpy.copy(centre)
        if (outward_unit_normal is not None):
            self.outward_unit_normal = numpy.copy(outward_unit_normal)
        if (vessel_type is not None):
            self.vessel_type = vessel_type
            
    def update_generating_cylinder_info(self,cylinder_radius=None,cylinder_centre=None,cylinder_length=None,cylinder_fillet=None) -> None:
        if (cylinder_radius is not None):
            self.cylinder_radius = cylinder_radius
        if (cylinder_centre is not None):
            self.cylinder_centre = cylinder_centre
        if (cylinder_length is not None):
            self.cylinder_length = numpy.copy(cylinder_length)
        if (cylinder_fillet is not None):
            self.cylinder_fillet = cylinder_fillet
        
    def update_apto_bdry_no(self,apto_bdry_no=None) -> None:
        self.apto_bdry_no = apto_bdry_no
        
        
''' 
  /|\ normal
   |
   | 
   | major axis
   |
  \|/
------- 
   ^centre
<-----> minor axis
'''
# Class to store central cavity (ellipsoid) info.
class Cavity:
    """! @brief Class for holding ellipsoid-shaped central cavity
    Members:
    - numpy.ndarray centre: 3D centre of the cavity.
    - numpy.ndarray orientation_normal: 3D unit vector indicating the major axis direction.
    - float minor_axis: minor axis length (or radius) perpendicular to major axis.
    - float major_axis: major axis length (along orientation_normal).
    - numpy.ndarray COM: centre-of-mass or auxiliary reference point.
    Methods:
    - Cavity()
        Constructor initialises members to None.
    - update_cavity(centre=None, orientation_normal=None, minor_axis=None, major_axis=None, COM=None) -> None
        Update stored parameters; only non-None arguments are applied and copied.
    - update_major_axis(z_plane_height, height_factor) -> None
        Set major_axis by computing distance from the cavity centre along orientation_normal
        to the horizontal plane z = z_plane_height and dividing by height_factor.
    Errors:
        - Exits if required members aren't initialised on construction
    """
    def __init__(self) -> None:
        self.centre = None
        self.orientation_normal = None
        self.minor_axis = None
        self.major_axis = None
        self.COM = None
        
    def update_cavity(self,centre=None,orientation_normal=None, \
                      minor_axis=None,major_axis=None,COM=None) -> None:
        if (centre is not None):
            self.centre = numpy.copy(centre)
        if (orientation_normal is not None):
            self.orientation_normal = numpy.copy(orientation_normal)
        if (minor_axis is not None):
            self.minor_axis = minor_axis
        if (major_axis is not None):
            self.major_axis = major_axis
        if (COM is not None):
            self.COM = numpy.copy(COM)
            
    def update_major_axis(self,z_plane_height,height_factor):
        # Make sure data initialised
        if (self.centre is None or self.orientation_normal is None):
            print(f"ERROR: calc_plate_intersection_pt")
            print(f"centre or orientation_normal have not been initialised")
            sys.exit(-1)
        
        dist = self.__calc_dist_to_plate(z_plane_height)
        self.update_cavity(major_axis = dist/height_factor)       
        
            
    # Find intersection point of normal with plate
    # Only called from functions which already check data initialisation      
    def __calc_dist_to_plate(self,z_plane_height):
            
        s = (z_plane_height - self.centre[2])/self.orientation_normal[2] # type: ignore
        # intersection_pt = self.centre + s*self.orientation_normal
        
        return s
    
class NodeSet:
    """! @brief Global unique node (vertex) class for representing collection of polygons.
    Purpose:
    - Deduplicate vertices across many placentone polygons and build a global node list.
    - Provide storage and conversion utilities for per-node wall heights:
            - abs_wall_height: a stochastic "absolute" height (prior to spherical-cap correction).
            - rel_wall_height: height relative to a local base/bottom surface (after interior fraction).
            - nodal_wall_height: actual z coordinate of wall = bottom_surf_z + rel_wall_height.
    - Provide mapping from each placentone's local vertex indices to global node indices.
    Members:
    - int dim: spatial dimension (usually 2 as Voronoi diagram set up on 2D plane).
    - str v_type: semantic container type (e.g., 'cotyledon' or 'lobule').
    - int no_nodes: populated number of unique nodes.
    - numpy.ndarray node: shape (no_nodes, dim) coordinates of each unique global node.
    - numpy.ndarray no_cell_nodes: integer counts of nodes per cell.
    - numpy.ndarray cell_nodes: mapping shape (no_cells, max_no_nodes) storing global node indices
        per cell (unused entries typically set to -1).
    - numpy.ndarray abs_wall_height: per-node absolute heights (stochastic source).
    - numpy.ndarray rel_wall_height: per-node relative heights (after spherical-cap fractioning).
    - numpy.ndarray nodal_wall_height: per-node actual z coordinate of the wall surface.
    Key methods:
    - NodeSet(dim, v_type)
        Constructor.
    - print_members() -> None
        Debug printing of each node and any populated height arrays.
    - set_node(node_no, node) -> None
        Set coordinates of a global node. Performs dimension checks.
    - set_all_random_heights(main, variance) -> None
        Populate abs_wall_height with random values around `main` using +/- `variance`.
    - set_all_random_heights_radius_dependent(inner_main, inner_variance,
                                                                                        outer_main, outer_variance, radius) -> None
        Per-node abs_wall_height is chosen based on whether node lies within a circular radius
        (uses circ_eval). Inner nodes sample around inner_main; outer nodes sample around outer_main,
        with outer values forced positive by design.
    - set_all_random_heights_negative(inner_main, inner_variance) -> None
        Populate abs_wall_height using only negative variance about the provided main value.
    - set_abs_wall_heights(inner_main, inner_variance, outer_main, outer_variance, outer_cutoff) -> None
        High-level initializer for abs_wall_height using radius-dependent randomization.
        Behavior depends on v_type (cotyledon vs lobule) and may call lobule-specific scaling.
    - rescale_lobule_abs_heights_with_cutoff(outer_cutoff) -> None
        Rescale lobule abs_wall_height values inside the cutoff using the spherical interior fraction
        to convert to a consistent relative internal height representation.
    - set_rel_and_nodal_wall_heights_cutoff(cutoff) -> None
        Convert stored abs_wall_height to rel_wall_height and nodal_wall_height by applying the
        spherical-cap interior fraction for nodes inside `cutoff`. Nodes outside the cutoff use
        abs_wall_height directly as the relative height. Requires abs_wall_height to be populated.
    - calc_rel_wall_height(node_no) -> float
        Compute relative wall height for a node as abs_wall_height * sphere_interior_fraction_at_xy.
    - calc_abs_wall_height(node_no) -> float
        Inverse operation: compute an absolute wall height given stored rel_wall_height and the
        spherical interior fraction at the node location.
    - calc_rel_wall_height_with_cutoff(node_no, cutoff_r) -> float
        Conditional relative height: if node lies outside cutoff_r returns abs_wall_height,
        else returns abs_wall_height scaled by the spherical interior fraction.
    - set_from_placentone_obj(placentone_obj) -> None
        Build the global node list by deduplicating vertices across an iterable of Placentone objects
        and then populate the per-cell global node mapping via an internal helper.
    - calc_cell_min_node_height_within_cutoff(cell_no, cutoff_r) -> float
        Return minimum rel_wall_height among nodes of the specified cell that are inside cutoff_r.
    - calc_cell_min_node_height_outside_cutoff(cell_no, cutoff_r) -> float
        Return minimum rel_wall_height among nodes of the specified cell that are outside cutoff_r.
    - calc_avg_rel_wall_height_within_cutoff(cutoff) -> float
        Average rel_wall_height across all nodes within the cutoff radius.
    Private helpers:
    - __create_abs_wall_height_storage(), __create_rel_wall_height_storage(), __create_nodal_wall_height_storage()
        Lazily allocate the respective arrays.
    - __set_cell_global_node_nos(placentone_obj) -> None
        Construct cell_nodes and no_cell_nodes arrays by matching placentone vertex coordinates to global nodes
        using points_near helper comparisons.
    Notes:
    - Several functions depend on external helpers and constants for tests
    - Many methods perform exits on errors
    - All functions assume set_from_placentone_obj already called
    - There are quite a few different node height functions, most were for testing during development
    """
    def __init__(self,dim,v_type) -> None:
        self.dim = dim
        
        self.v_type = v_type
        
        self.no_nodes = None
        self.node = None

        self.no_cell_nodes = None
        self.cell_nodes = None
        
        # Note this is the 'absolute' height, in actuality a scaling is done s.t.
        # pt_height = abs_wall_height*(1-actual_z_inside_placenta/placenta_height)
        self.abs_wall_height = None
        # This is the relative height, i.e. height above bottom surface
        self.rel_wall_height = None
        # This is the actual z value of the wall - buttom_surf_z + rel_wall_height
        self.nodal_wall_height = None
        
    # This should really be the init constructor, but instead I just call it after creating the object
    def set_from_placentone_obj(self,placentone_obj) -> None:
        
        no_global_vertices = 0
        global_vertices = []
        
        for cell in placentone_obj:
            
            [no_vertices,no_edges,vertices,edges] = \
                cell.get_vertices_edges()
                
            for vertex_no in range(0,no_vertices):
                
                already_added = False
                
                for global_v_no in range(0,no_global_vertices):
                    
                    if (fns.points_near( \
                            global_vertices[global_v_no],vertices[vertex_no,:])):
                        
                        already_added = True
                        break
                        
                if (not(already_added)):
                    
                    global_vertices.append(vertices[vertex_no,:])
                    no_global_vertices = no_global_vertices + 1
        
        if (no_global_vertices < 2):
            print("ERROR: set_from_foronoi_obj")
            print("Too few vertices")
            sys.exit(-1) 
            
        self.no_nodes = no_global_vertices
        self.node = numpy.empty((self.no_nodes,self.dim))
        for count,global_v in enumerate(global_vertices):
            self.set_node(count,global_v)
        
        self.__set_cell_global_node_nos(placentone_obj)
        
        return None
    
    def print_members(self) -> None:
        for i in range(0,self.no_nodes):
            print(f"Vertex {i} = \
                {numpy.array2string(self.node[i,:], separator=',')}")
            
            if (self.abs_wall_height is not None):
                print(f"Vertex_abs_wall_height{i} = \
                    {self.abs_wall_height[i]}")
            if (self.rel_wall_height is not None):
                print(f"Vertex_rel_wall_height{i} = \
                    {self.rel_wall_height[i]}")
            if (self.nodal_wall_height is not None):
                print(f"Vertex_nodal_wall_height{i} = \
                    {self.nodal_wall_height[i]}")
        
    def set_node(self,node_no,node) -> None:
        
        if (len(node) != self.dim):
            print("ERROR: set_node")
            print(f"Node length incorrect, {len(node)} and {self.dim}")
            sys.exit(-1)
            
        self.node[node_no,:] = copy.deepcopy(node)
            
        return None

    def __create_abs_wall_height_storage(self) -> None:

        if (self.abs_wall_height is None):
            self.abs_wall_height = numpy.empty(self.no_nodes)
        
        return None
    
    def __create_rel_wall_height_storage(self) -> None:

        if (self.rel_wall_height is None):
            self.rel_wall_height = numpy.empty(self.no_nodes)
        
        return None
    
    def __create_nodal_wall_height_storage(self) -> None:

        if (self.nodal_wall_height is None):
            self.nodal_wall_height = numpy.empty(self.no_nodes)
        
        return None
    
    def set_all_random_heights(self,main,variance) -> None:
        
        self.__create_abs_wall_height_storage()
        
        for i in range(0,self.no_nodes):
            rand_sign = numpy.random.choice([-1,1])
            rand_real = main + \
                rand_sign*numpy.random.rand()*variance
            self.abs_wall_height[i] = rand_real
        
        return None
    
    def set_all_random_heights_radius_dependent(self, \
            inner_main,inner_variance,outer_main,outer_variance,radius):
        
        self.__create_abs_wall_height_storage()
        
        for i in range(0,self.no_nodes):
            
            node_pt = copy.deepcopy(self.node[i,:])
            
            if (circ_eval(*node_pt) <= radius**2):
                
                self.__set_pt_random_height( \
                    i,inner_main,inner_variance)
                
            else:
                
                self.__set_pt_random_height_positive( \
                    i,outer_main,outer_variance)
        
        return None
    
    def set_all_random_heights_negative(self, \
            inner_main,inner_variance):
        
        self.__create_abs_wall_height_storage()
        
        for i in range(0,self.no_nodes):
            
            self.__set_pt_random_height_negative( \
                i,inner_main,inner_variance)
                
        return None
    
    def set_abs_wall_heights(self, \
            inner_main,inner_variance, \
            outer_main,outer_variance,outer_cutoff) -> None:
        
        self.__create_abs_wall_height_storage()
        
        if (outer_main is None or \
                outer_variance is None or \
                outer_cutoff is None):
            print("ERROR: set_abs_wall_heights")
            print(f"v_type == {self.v_type} but an argument is None")
            sys.exit(-1)
    
        self.set_all_random_heights_radius_dependent( \
            inner_main,inner_variance, \
            outer_main,outer_variance,outer_cutoff)
        
        # If cotyledon, set all random heights
        # If lobule, set abs_wall_height to be the same as rel_wall_height
        #   and then scale for consistency
        # No longer used as originally intended hence the passes
        if (self.v_type == 'cotyledon'):
            pass
        elif (self.v_type == 'lobule'):
            pass #self.rescale_lobule_abs_heights_with_cutoff(outer_cutoff)
        else:
            print("ERROR: set_abs_wall_heights")
            print(f"Unrecognised v_type {self.v_type}")
            sys.exit(-1)
            
        return None
    
    def rescale_lobule_abs_heights_with_cutoff(self,outer_cutoff) -> None:
        
        for i in range(0,self.no_nodes):
            
            xy_pt = copy.deepcopy(self.node[i,:])
            
            if (circ_eval(*xy_pt) <= outer_cutoff**2):
                self.abs_wall_height[i] = self.abs_wall_height[i]/ \
                    fns.calc_sphere_interior_height_frac_at_xy(xy_pt)
                        
    
    def set_rel_and_nodal_wall_heights_cutoff(self,cutoff) -> None:
        
        self.__create_rel_wall_height_storage()
        self.__create_nodal_wall_height_storage()
        
        if (self.abs_wall_height is None):
            print("ERROR: set_rel_wall_heights_cutoff")
            print("abs_wall_height not initialised")
            sys.exit(-1)
            
        if (self.v_type == 'cotyledon' or self.v_type == 'lobule'):
        
            for i in range(0,self.no_nodes):
                
                xy_pt = copy.deepcopy(self.node[i,:])
                
                if (circ_eval(*xy_pt) <= cutoff**2):
                    self.rel_wall_height[i] = self.calc_rel_wall_height(i)
                    self.nodal_wall_height[i] = \
                        fns.calc_sphere_height_at_xy(xy_pt) + \
                            self.rel_wall_height[i]
                else:
                    self.rel_wall_height[i] = self.abs_wall_height[i]
                    self.nodal_wall_height[i] = self.rel_wall_height[i]
            
        #elif (self.v_type == 'lobule'):
        #    
        #    for i in range(0,self.no_nodes):
        #        
        #        self.rel_wall_height[i] = self.abs_wall_height[i]
        #        self.nodal_wall_height[i] = self.rel_wall_height[i]
            
        return None
    
    def __set_pt_random_height(self,node_no,main,variance) -> None:
        
        rand_sign = numpy.random.choice([-1,1])
        rand_real = main + \
            rand_sign*numpy.random.rand()*variance
        self.abs_wall_height[node_no] = rand_real
        
        return None
    
    # As above but variance can only be negative
    def __set_pt_random_height_negative(self,node_no,main,variance) -> None:
        
        rand_real = main + \
            -numpy.random.rand()*variance
        self.abs_wall_height[node_no] = rand_real
        
        return None
    
    # As above but variance can only be negative
    def __set_pt_random_height_positive(self,node_no,main,variance) -> None:
        
        rand_real = main + \
            numpy.random.rand()*variance
        self.abs_wall_height[node_no] = rand_real
        
        return None
    
    # Relative in sense that considers height of spherical cap bottom plate
    def calc_rel_wall_height(self,node_no) -> float:
        
        xy_pt = copy.deepcopy(self.node[node_no,:])
        
        rel_z = self.abs_wall_height[node_no]* \
            fns.calc_sphere_interior_height_frac_at_xy(xy_pt)
        
        return rel_z
    
    # Relative in sense that considers height of spherical cap bottom plate
    def calc_abs_wall_height(self,node_no) -> float:
        
        xy_pt = copy.deepcopy(self.node[node_no,:])
        
        rel_z = self.rel_wall_height[node_no]/ \
            fns.calc_sphere_interior_height_frac_at_xy(xy_pt)
        
        return rel_z
        
    # Relative in sense that considers height of spherical cap bottom plate
    def calc_rel_wall_height_with_cutoff(self,node_no,cutoff_r) -> float:
        
        xy_pt = copy.deepcopy(self.node[node_no,:])
            
        if (circ_eval(*xy_pt) >= cutoff_r**2):
            rel_z = self.abs_wall_height[node_no]
        else:
            rel_z = self.abs_wall_height[node_no]* \
                fns.calc_sphere_interior_height_frac_at_xy(xy_pt)
        
        return rel_z
    
    def __set_cell_global_node_nos(self,placentone_obj) -> None:
        
        # Work out the maximum number of nodes per cotyledon
        max_no_nodes = fns.determine_max_nodes_from_cotyledons(placentone_obj)
        
        # Add member which holds global vertex numbers in each cell
        self.no_cell_nodes = numpy.empty(len(placentone_obj),dtype = int)
        self.cell_nodes = numpy.empty((len(placentone_obj),max_no_nodes),dtype = int)
        self.cell_nodes[:,:] = -1
        
        for cell_no,cell in enumerate(placentone_obj):
            
            [no_vertices,no_edges,vertices,edges] = \
                cell.get_vertices_edges()
                
            self.no_cell_nodes[cell_no] = no_vertices
            
            for cell_v_no in range(0,cell.no_vertices):
                
                cell_pt = copy.deepcopy(cell.vertices[cell_v_no,:])
                
                for global_v_no in range(0,self.no_nodes):
                
                    set_pt = copy.deepcopy(self.node[global_v_no,:])
                    
                    if (fns.points_near( \
                            cell_pt,set_pt)):
                        
                        self.cell_nodes[cell_no,cell_v_no] = global_v_no
                        break
                    
    def calc_cell_min_node_height_within_cutoff(self,cell_no,cutoff_r) -> float:
        
        if (self.no_cell_nodes is None or self.cell_nodes is None):
            print("ERROR: calc_cell_min_node_height_within_cutoff")
            print("no_cell_nodes or cell_nodes not initialised")
            sys.exit(-1)
            
        min_height = 1.0e8
        
        for i in range(0,self.no_cell_nodes[cell_no]):
            
            glob_v_no = self.cell_nodes[cell_no,i]
            xy_pt = self.node[glob_v_no,:]
            
            if (circ_eval(*xy_pt) <= cutoff_r**2):
                if (self.rel_wall_height[glob_v_no] < min_height):
                    min_height = self.rel_wall_height[glob_v_no]
                
        return min_height
    
    def calc_cell_min_node_height_outside_cutoff(self,cell_no,cutoff_r) -> float:
        
        if (self.no_cell_nodes is None or self.cell_nodes is None):
            print("ERROR: calc_cell_min_node_height_outside_cutoff")
            print("no_cell_nodes or cell_nodes not initialised")
            sys.exit(-1)
            
        min_height = 1.0e8
        
        for i in range(0,self.no_cell_nodes[cell_no]):
            
            glob_v_no = self.cell_nodes[cell_no,i]
            xy_pt = self.node[glob_v_no,:]
            
            if (circ_eval(*xy_pt) >= cutoff_r**2):
                if (self.rel_wall_height[glob_v_no] < min_height):
                    min_height = self.rel_wall_height[glob_v_no]
                
        return min_height
    
    def calc_avg_rel_wall_height_within_cutoff(self,cutoff) -> None:
        
        if (self.rel_wall_height is None):
            print("ERROR: store_avg_rel_wall_height_within_cutoff")
            print("rel_wall_height not initialised")
            sys.exit(-1)
        
        counter = 0
        avg_height = 0.0
        
        for node_no in range(0,self.no_nodes):
            
            xy_pt = copy.deepcopy(self.node[node_no,:])
                
            if (circ_eval(*xy_pt) < cutoff**2):
                counter = counter + 1
                avg_height = avg_height + self.rel_wall_height[node_no]
                
        if (counter > 0):
            avg_height = avg_height / counter
        
        return avg_height
            
class EdgeSet:
    
    def __init__(self,node_set) -> None:
        self.node_set = copy.deepcopy(node_set)
        self.dim = self.node_set.dim
        
        self.no_edges = None
        self.edge = None
        
        self.edge_length = None
        self.edge_dir = None
        
        self.transition_percent = None
        self.transition_point = None
        
        self.no_cell_edges = None
        self.cell_edges = None
        
    def print_members(self):
        print(f"No edges: {self.no_edges}")
        for i in range(0,self.no_edges):
            print(f"Edge {i}: {self.edge[i,:]}")
            print(f"Edge direction {i}: {self.edge_dir[i]}")
            print(f"Edge length {i}: {self.edge_length[i]}")
        
    def set_all_edges(self,vertex_nos):
        
        arr_shape = vertex_nos.shape
        
        if (len(arr_shape) != 2):
            print("ERROR: set_all_shape")
            print("vertex_nos is not a 2D array")
            sys.exit(-1)
        
        no_vertex_pairs = arr_shape[0]
        
        if (no_vertex_pairs != self.no_edges):
            print("ERROR: set_all_shape")
            print("no_vertex_pairs != no_edges")
            sys.exit(-1)
            
        for edge_no in range(0,self.no_edges):
            self.edge[edge_no,:] = copy.deepcopy(vertex_nos[edge_no,:])
    
    def set_edge(self,edge_no,vertex_no) -> None:
        
        if (len(vertex_no) != 2):
            print("ERROR: set_edge")
            print(f"vertex_no is not length 2: len = {len(vertex_no)}")
            sys.exit(-1)
            
        self.edge[edge_no,:] = copy.deepcopy(vertex_no)
        
    def update_edge_properties(self,edge_no):
        v_no = copy.deepcopy(self.edge[edge_no,:])
        
        self.edge_dir[edge_no,:] = \
            self.node_set.node[v_no[1],:]-self.node_set.node[v_no[0],:]
        self.edge_length[edge_no] = numpy.linalg.norm(self.edge_dir[edge_no,:])
        self.edge_dir[edge_no,:] = self.edge_dir[edge_no,:] / self.edge_length[edge_no]
    
    def set_transition_points_percent(self,per) -> None:
        
        self.transition_percent = numpy.empty(self.no_edges)
        self.transition_point = numpy.empty((self.no_edges,2,self.dim))
        
        n_set = self.node_set
        
        for edge_no in range(0,self.no_edges):
            
            # This is a hack, but it's a quick fix for small edges
            if (self.edge_length[edge_no] < 1.0):
                self.transition_percent[edge_no] = 0.4
            else:
                self.transition_percent[edge_no] = per
            
            v0 = copy.deepcopy(n_set.node[self.edge[edge_no,0],:])
            v1 = copy.deepcopy(n_set.node[self.edge[edge_no,1],:])
            
            edge_dir = copy.deepcopy(self.edge_dir[edge_no,:])
            
            v0_trans = v0 + \
                self.transition_percent[edge_no] * self.edge_length[edge_no] * edge_dir
            v1_trans = v1 - \
                self.transition_percent[edge_no] * self.edge_length[edge_no] * edge_dir
                
            self.transition_point[edge_no,0,:] = v0_trans
            self.transition_point[edge_no,1,:] = v1_trans
            
        return None
        
    def set_edge_ele_adjacencies(self):
        
        return None
    
    def calc_pt_along_edge(self,edge_no,ratio) -> float:
        
        start_node = copy.deepcopy(self.node_set.node[self.edge[edge_no,0],:])
            
        xy_pt = start_node + \
            self.edge_length[edge_no] * self.edge_dir[edge_no,:] * ratio
        
        return xy_pt
    
    def calc_rel_height_along_edge(self,edge_no,ratio) -> None:
        
        n_set = self.node_set
        
        if (n_set.nodal_wall_height is None):
            print("ERROR: calc_rel_height_along_edge")
            print(f"Node set's nodal_wall_height is None")
            sys.exit(-1)
        elif (ratio < 0.0 or ratio > 1.0):
            print("ERROR: calc_rel_height_along_edge")
            print(f"0 <= ratio <= 1")
            sys.exit(-1)
            
        node_height = copy.deepcopy(n_set.nodal_wall_height[self.edge[edge_no,:]])
        
        xy_pt = self.calc_pt_along_edge(edge_no,ratio)
        
        if (self.transition_percent is not None):
            
            if (ratio < self.transition_percent[edge_no]):  
                
                return (node_height[0] - fns.calc_sphere_height_at_xy(xy_pt))
            
            elif (ratio > 1.0 - self.transition_percent[edge_no]):
                
                return (node_height[1] - fns.calc_sphere_height_at_xy(xy_pt))
            
            else:
                
                interim_ratio = ratio - self.transition_percent[edge_no]
                scaled_ratio = interim_ratio/(1.0 - 2.0 * self.transition_percent[edge_no])
                interim_node_height = \
                    (1.0 - scaled_ratio) * node_height[0] + scaled_ratio * node_height[1]
                return (interim_node_height - fns.calc_sphere_height_at_xy(xy_pt))
            
        else:
        
            interim_node_height = \
                ratio * node_height[0] + (1.0 - ratio) * node_height[1]
            return (interim_node_height - fns.calc_sphere_height_at_xy(xy_pt))
        
    def __set_cell_global_edge_nos(self,placentone_obj) -> None:
        
        # Work out the maximum number of edges per cotyledon
        max_no_edge = fns.determine_max_edges_from_cotyledons(placentone_obj)
        
        # Add member which holds global vertex numbers in each cell
        self.no_cell_edges = numpy.empty(len(placentone_obj),dtype = int)
        self.cell_edges = numpy.empty((len(placentone_obj),max_no_edge),dtype = int)
        self.cell_edges[:,:] = -1
        
        for cell_no,cell in enumerate(placentone_obj):
            
            [no_vertices,no_edges,vertices,edges] = \
                cell.get_vertices_edges()
                
            self.no_cell_edges[cell_no] = no_edges
            
            for cell_e_no in range(0,cell.no_edges):
                
                cell_edge_pt = numpy.empty((2,self.dim))
                cell_v_no = copy.deepcopy(cell.edges[cell_e_no,:])
                for i in range(0,2):
                    cell_edge_pt[i,:] = copy.deepcopy(cell.vertices[cell_v_no[i],:])
                
                for glob_e_no in range(0,self.no_edges):
                
                    glob_edge_pt = numpy.empty((2,self.dim))
                    glob_v_no = copy.deepcopy(self.edge[glob_e_no,:])
                    for i in range(0,2):
                        glob_edge_pt[i,:] = copy.deepcopy(self.node_set.node[glob_v_no[i],:])
                    
                    if (fns.points_near( \
                            cell_edge_pt[0,:],glob_edge_pt[0,:]) and \
                            fns.points_near( \
                            cell_edge_pt[1,:],glob_edge_pt[1,:]) or \
                        fns.points_near( \
                            cell_edge_pt[0,:],glob_edge_pt[1,:]) and \
                            fns.points_near( \
                            cell_edge_pt[1,:],glob_edge_pt[0,:])):
                        
                        self.cell_edges[cell_no,cell_e_no] = glob_e_no
                        break
        
        
    def set_from_placentone_obj(self,placentone_obj):
        
        num_err = 1.0e-12
        
        no_global_edges = 0
        global_edges = []
        
        no_global_points = self.node_set.no_nodes
        
        for cell in placentone_obj:
            
            [no_vertices,no_edges,vertices,edges] = \
                cell.get_vertices_edges()
            
            # Loop over local edges
            for edge_no in range(0,no_edges):
                
                # Store the vertex coords of edge
                local_e_v0 = vertices[edges[edge_no,0],:]
                local_e_v1 = vertices[edges[edge_no,1],:]
                
                already_added = False
                
                # Check if this edge is already in global set of edges
                for global_e_no in range(0,no_global_edges):
                    
                    global_e = global_edges[global_e_no]
                    global_e_v0 = copy.deepcopy( \
                        self.node_set.node[global_e[0],:])
                    global_e_v1 = copy.deepcopy( \
                        self.node_set.node[global_e[1],:])                    
                    
                    if ( \
                            (fns.points_near(local_e_v0,global_e_v0) and \
                                fns.points_near(local_e_v1,global_e_v1)) or \
                            (fns.points_near(local_e_v0,global_e_v1) and \
                                fns.points_near(local_e_v1,global_e_v0))):
                        
                        already_added = True
                        break
                
                if (not(already_added)):
                    
                    global_e_v0_no = -1
                    global_e_v1_no = -1
                    
                    for global_v_no in range(0,no_global_points):
                        
                        # Check 1st point
                        if (fns.points_near( \
                                self.node_set.node[global_v_no,:],local_e_v0)):
                            global_e_v0_no = global_v_no
                        # Check 2nd point
                        if (fns.points_near( \
                                self.node_set.node[global_v_no,:],local_e_v1)):
                            global_e_v1_no = global_v_no
                    
                    # Check found points
                    if (global_e_v0_no == -1 or global_e_v1_no == -1):
                        print(f"ERROR: edge_set - set_from_placentone_obj")
                        print(f"global_e_v0_no = {global_e_v0_no}, global_e_v1_no = {global_e_v1_no}")
                        sys.exit(-1)
                    
                    # Add array representing this edge with vertex numbers to list
                    global_edges.append( \
                        numpy.sort( \
                            numpy.array([global_e_v0_no,global_e_v1_no],dtype = int)))
                    no_global_edges = no_global_edges + 1
        
        # Store edge data
        self.no_edges = no_global_edges
        self.edge = numpy.empty((self.no_edges,2),dtype = int)
        self.edge_dir = numpy.empty((self.no_edges,2))
        self.edge_length = numpy.empty(self.no_edges)
        for count,global_e in enumerate(global_edges):
            self.set_edge(count,global_e)
            self.update_edge_properties(count)
            
        self.__set_cell_global_edge_nos(placentone_obj) 
        
    def get_vertices_from_cell_edge(self,cell_no,cell_e_no):
        
        if (self.no_cell_edges is None or self.cell_edges is None):
            print("ERROR: get_vertices_from_cell_edge")
            print("no_cell_edges or cell_edges not initialised")
            sys.exit(-1)
        
        glob_e_no = copy.deepcopy(self.cell_edges[cell_no,cell_e_no])
        
        v_no = copy.deepcopy(self.edge[glob_e_no,:])
        vertex_1 = copy.deepcopy(self.node_set.node[v_no[0],:])
        vertex_2 = copy.deepcopy(self.node_set.node[v_no[1],:])
        
        return [vertex_1,vertex_2]
    
    def calc_edge_vertex_circle_intersection(self,edge_no,shift) -> None:
        
        self.edge[edge_no,:] = self.edge[edge_no,:] + shift
        
        return None
    