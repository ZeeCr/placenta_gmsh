import os
import sys
import importlib

import math
import scipy
import numpy
import copy

import gmsh
import matplotlib
import foronoi
from foronoi.contrib import ConcavePolygon

import placenta_const
import placenta_plots as plots
import placenta_fns as fns
import placenta_classes as clses
import placenta_foronoi as frni

importlib.reload(placenta_const)
importlib.reload(plots)
importlib.reload(fns)
importlib.reload(clses)
importlib.reload(frni)

from placenta_const import *




def setup_voronoi_pts():
    
    if (not(fixed_cotyledon_pts)):
        #points = numpy.random.rand(no_placentones,2)  
        #points = (placenta_radius-10)*(2.0*points-1.0)
        points = fns.points_distribution_inner_random(placenta_radius, \
                                         no_placentones,no_placentones-no_inner_placentones,no_inner_placentones, \
                                         inner_sub_radius, \
                                         outer_radius_offset=placenta_voronoi_outer_radius_offset,outer_angle_variance=outer_angle_variance(no_placentones))
    elif (fixed_cotyledon_pts):
        no_placentones = no_stored_cotyledon
        points = cotyledon_pts
        if (len(points) != no_placentones):
            print(f"ERROR setup_voronoi_pts: len(points) != no_placentones")
            logger.debug(f"len(points) = {len(points)}, no_placentones = {no_placentones}")
            # sys.exit(-10)
            raise Exception("Error in setup_voronoi_pts")
    else:
        print(f"ERROR setup_voronoi_pts: fixed_cotyledon_pts not specified")
        # sys.exit(-1)
        raise Exception("Error in setup_voronoi_pts")

    print(f"Voronoi generating points : \n {points}")
    
    return points

def setup_voronoi_diagram(points):
    
    foronoi_polygon_bound = placenta_radius+placenta_voronoi_outer_radius_offset
    
    polygon = foronoi.Polygon([
            (-foronoi_polygon_bound,-foronoi_polygon_bound),(foronoi_polygon_bound,-foronoi_polygon_bound), \
            (foronoi_polygon_bound,foronoi_polygon_bound),(-foronoi_polygon_bound,foronoi_polygon_bound)
        ])
    
    #polygon = foronoi.Polygon([
    #    (-placenta_radius,-placenta_radius),(placenta_radius,-placenta_radius), \
    #    (placenta_radius,placenta_radius),(-placenta_radius,placenta_radius)
    #])
    
    """
    polygon = foronoi.Polygon([
        (0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0, 1.0)
    ])"""

    voronoi_diagram = foronoi.Voronoi(polygon)
    voronoi_diagram.create_diagram(points=points)
    # frni.visualise_voronoi(voronoi_diagram)
    
    return voronoi_diagram


def create_cotyledons(model,points,no_cells,voronoi_diagram):
    
    v_type = 'cotyledon'
    wall_thickness = placentone_wall_thickness
    wall_height = placentone_removal_height
    wall_height_variability = wall_height/2.0
    outer_wall_height = placenta_height
    outer_wall_height_variability = outer_wall_height/4.0
    outer_wall_cutoff = placenta_radius
    joint_transition = 0.05
    
    [node_set,edge_set,cotyledon_list] = create_c_l_cut( \
        model,no_cells,voronoi_diagram, \
        v_type,wall_thickness,joint_transition, \
        wall_height,wall_height_variability, \
        outer_wall_height,outer_wall_height_variability,outer_wall_cutoff)
    
    # plots.plot_placentone_list(points,cotyledon_list)
    
    return [node_set,edge_set,cotyledon_list]


def create_lobules(model,no_cotyledon,cotyledon_list,c_node_set,c_edge_set):
    
    no_lobules = 0
    lobule_list = []
    
    lobule_node_set_per_cotyledon = numpy.empty(no_cotyledon, dtype=object)
    lobule_edge_set_per_cotyledon = numpy.empty(no_cotyledon, dtype=object)
    
    for cotyledon_no in range(0,no_cotyledon):
        
        v_type = 'lobule'
        wall_thickness = lobule_wall_thickness
        wall_height = c_node_set.calc_cell_min_node_height_within_cutoff( \
            cotyledon_no,placenta_radius)/1.5
        wall_height_variability = 0.0#wall_height/4.0
        outer_wall_height = c_node_set.calc_cell_min_node_height_outside_cutoff( \
            cotyledon_no,placenta_radius)
        outer_wall_height_variability = 0.0#outer_wall_height/4.0
        outer_wall_cutoff = placenta_radius
        joint_transition = 0.05 # Note that for lobules, this has a hack for small edges in set_transition_points_percent
        
        logger.info("---------------------\n" + \
                     f"Setting up sub placentones for {cotyledon_no}")
        print("---------------------\n" + \
                     f"Setting up sub placentones for {cotyledon_no}")    
        cotyledon_no_vertices = cotyledon_list[cotyledon_no].no_vertices
        cotyledon_vertices = numpy.copy(cotyledon_list[cotyledon_no].vertices)

        if (not(fixed_lobules)):
            if (cotyledon_list[cotyledon_no].boundary_cell):
                sub_voronoi = fns.lloyds_algorithm(cotyledon_no_vertices,cotyledon_vertices,no_lobules_outer)
                no_lobules_to_add=no_lobules_outer
            else:
                sub_voronoi = fns.lloyds_algorithm(cotyledon_no_vertices,cotyledon_vertices,no_lobules_inner)
                no_lobules_to_add=no_lobules_inner
            no_lobules=no_lobules+no_lobules_to_add
        else:
            
            if (lobule_foronoi_type == 'standard'):
                pgon = foronoi.Polygon(cotyledon_vertices)
            elif (lobule_foronoi_type == 'concave'):
                pgon = ConcavePolygon(cotyledon_vertices)
            else:
                print(f"Error: create_lobules")
                print(f"Unrecognised lobule_foronoi_type: {lobule_foronoi_type}")
                raise Exception("Unrecognised lobule_foronoi_type")
                # sys.exit(-1)
                
            print("aASDSADSA\n\n\n\n\n")
            sub_voronoi = foronoi.Voronoi(pgon)
            sub_voronoi.create_diagram(points = \
                    f_lobule_pts[0:f_no_lobule_pts[cotyledon_no]+1,:,cotyledon_no])
            
            no_lobules_to_add = f_no_lobule_pts[cotyledon_no]
            no_lobules = no_lobules + f_no_lobule_pts[cotyledon_no]
            
            # frni.visualise_voronoi(sub_voronoi)
            
        [node_set,edge_set,sub_lobule_list] = create_c_l_cut( \
            model,no_lobules_to_add,sub_voronoi, \
            v_type,wall_thickness,joint_transition, \
            wall_height,wall_height_variability, \
            outer_wall_height,outer_wall_height_variability,outer_wall_cutoff)
        
        #lobule_list.append(sub_lobule_list)
        for i in sub_lobule_list:
            lobule_list.append(i)
            
        lobule_node_set_per_cotyledon[cotyledon_no] = node_set
        lobule_edge_set_per_cotyledon[cotyledon_no] = edge_set
        
        model.occ.synchronize()
    
    return [node_set,edge_set,no_lobules,lobule_list, \
        lobule_node_set_per_cotyledon,lobule_edge_set_per_cotyledon]

def create_c_l_cut(model,no_cells,voronoi_diagram, \
        v_type,wall_thickness,joint_transition, \
        wall_height,wall_height_variability, \
        outer_wall_height = None,outer_wall_height_variability = None,outer_wall_cutoff = None):
    
    struct_list = []
    
    for cell_no in range(0,no_cells):
        
        logger.info("---------------------\n" + \
                     f"Setting up structure {cell_no} from foronoi")
        
        frni_placentone = frni.ForonoiPlacentone( \
                voronoi_diagram.sites[cell_no],placenta_radius)
        
        struct_list.append(frni_placentone)
        
    node_set = clses.NodeSet(2,v_type)
    node_set.set_from_placentone_obj(struct_list)
    node_set.set_abs_wall_heights( \
        wall_height,wall_height_variability, \
        outer_wall_height,outer_wall_height_variability,outer_wall_cutoff)
    node_set.set_rel_and_nodal_wall_heights_cutoff(outer_wall_cutoff)
    edge_set = clses.EdgeSet(node_set) 
    edge_set.set_from_placentone_obj(struct_list)
    edge_set.set_transition_points_percent(joint_transition)
        
    cut_from_voronoi(model,node_set,edge_set,wall_thickness)
        
        
        
    
    
    return [node_set,edge_set,struct_list]

def cut_from_voronoi(model, \
        node_set,edge_set,wall_thickness):
        
    existing_vols = model.occ.getEntities(3)
    no_existing_vols = len(existing_vols)
    
    domain_vol = fns.determine_largest_gmsh_vol(model)
    
    '''
    for edge_no in range(0,edge_set.no_edges):
        fns.add_wall_from_edge(model,edge_no,edge_set,placentone_wall_thickness)

    for node_no in range(0,node_set.no_nodes):
        fns.add_joint_from_node(model,node_no,node_set,placentone_wall_thickness)
    '''
    
    for edge_no in range(0,edge_set.no_edges):
        fns.add_wall_from_edge_with_joint(model,edge_no,edge_set,wall_thickness)
    '''
    wall_list = [i for i in model.occ.getEntities(3) if i != (3,domain_vol)]
    model.occ.cut([(3,domain_vol)],wall_list)
    '''  
    debug_counter = 0
    while (len(model.occ.getEntities(3)) - no_existing_vols > 1):
        debug_counter = debug_counter + 1
        wall_list = [i for i in model.occ.getEntities(3) if i not in existing_vols]
        wall_list2 = [i for i in wall_list[1:]]
        model.occ.fuse([wall_list[0]],wall_list2)
        model.occ.synchronize()
        
        if (debug_counter > 100):
            print("ERROR: cut_from_voronoi")
            print("Not fusing properly")
            # sys.exit(-1)
            raise Exception("Not fusing properly")

    # Cut walls
    model.occ.cut([(3,domain_vol)],[(3,2)])
        
    return [node_set,edge_set]




def cut_from_voronoi_o(model,points,no_cells,voronoi_diagram):
    
    cotyledons = []
    
    existing_vols = model.occ.getEntities(3)
    no_existing_vols = len(existing_vols)
    
    domain_vol = fns.determine_largest_gmsh_vol(model)
    
    for cell_no in range(0,no_cells):

        curr_pt = model.occ.getMaxTag(0)+1
        curr_line = model.occ.getMaxTag(1)+1
        logger.info("---------------------\n" + \
                     f"Setting up placentone {cell_no} from foronoi \n" + \
                     f"curr_pt: {curr_pt} \n" + \
                     f"curr_line: {curr_line}")
        
        frni_placentone = frni.ForonoiPlacentone( \
                voronoi_diagram.sites[cell_no],placenta_radius)
        
        cotyledons.append(frni_placentone)
    
    node_set = clses.NodeSet(2)
    node_set.set_from_placentone_obj(cotyledons)
    node_set.set_all_random_heights_radius_dependent( \
        placentone_removal_height,placentone_removal_height/2.0, \
        placenta_height,placenta_height/10.0,placenta_radius)
    #node_set.print_members()
    edge_set = clses.EdgeSet(node_set) 
    edge_set.set_from_placentone_obj(cotyledons)
    edge_set.set_transition_points_percent(0.04)
    #edge_set.print_members()
    
    '''
    for edge_no in range(0,edge_set.no_edges):
        fns.add_wall_from_edge(model,edge_no,edge_set,placentone_wall_thickness)

    for node_no in range(0,node_set.no_nodes):
        fns.add_joint_from_node(model,node_no,node_set,placentone_wall_thickness)
    '''
    
    for edge_no in range(0,edge_set.no_edges):
        fns.add_wall_from_edge_with_joint(model,edge_no,edge_set,placentone_wall_thickness)
    '''
    wall_list = [i for i in model.occ.getEntities(3) if i != (3,domain_vol)]
    model.occ.cut([(3,domain_vol)],wall_list)
    '''  
    debug_counter = 0
    while (len(model.occ.getEntities(3)) - no_existing_vols > 1):
        debug_counter = debug_counter + 1
        wall_list = [i for i in model.occ.getEntities(3) if i not in existing_vols]
        wall_list2 = [i for i in wall_list[1:]]
        model.occ.fuse([wall_list[0]],wall_list2)
        model.occ.synchronize()
        
        if (debug_counter > 100):
            print("ERROR: cut_from_voronoi")
            print("Not fusing properly")
            # sys.exit(-1)
            raise Exception("Not fusing properly")

    # Cut walls
    model.occ.cut([(3,domain_vol)],[(3,2)])

    # plots.plot_placentone_list(points,cotyledons)
        
    return [node_set,edge_set,cotyledons]




def create_lobules_o(model,cotyledons):
    
    no_lobules = 0
    lobules = []
    
    for placentone_no in range(0,no_placentones):

        logger.info("---------------------\n" + \
                     f"Setting up sub placentones for {placentone_no}")
        print("---------------------\n" + \
                     f"Setting up sub placentones for {placentone_no}")    
        placentone_no_vertices = cotyledons[placentone_no].no_vertices
        placentone_no_edges = cotyledons[placentone_no].no_edges
        placentone_vertices = numpy.copy(cotyledons[placentone_no].vertices)
        placentone_edges = numpy.copy(cotyledons[placentone_no].edges)

        if (not(fixed_lobules)):
            if (cotyledons[placentone_no].boundary_cell):
                sub_voronoi = fns.lloyds_algorithm(placentone_no_vertices,placentone_vertices,no_lobules_outer)
                no_lobules_to_add=no_lobules_outer
            else:
                sub_voronoi = fns.lloyds_algorithm(placentone_no_vertices,placentone_vertices,no_lobules_inner)
                no_lobules_to_add=no_lobules_inner
            no_lobules=no_lobules+no_lobules_to_add
        else:
            
            if (lobule_foronoi_type == 'standard'):
                pgon = foronoi.Polygon(placentone_vertices)
            elif (lobule_foronoi_type == 'concave'):
                pgon = ConcavePolygon(placentone_vertices)
            else:
                print(f"Error: create_lobules")
                print(f"Unrecognised lobule_foronoi_type: {lobule_foronoi_type}")
                # sys.exit(-1)
                raise Exception(f"Unrecognised lobule_foronoi_type: {lobule_foronoi_type}")

            sub_voronoi = foronoi.Voronoi(pgon)
            sub_voronoi.create_diagram(points = \
                    f_lobule_pts[0:f_no_lobule_pts[placentone_no]+1,:,placentone_no])
            no_lobules_to_add = f_no_lobule_pts[placentone_no]
            no_lobules = no_lobules + f_no_lobule_pts[placentone_no]
            
            # frni.visualise_voronoi(sub_voronoi)

        cut_from_voronoi(model, \
            f_lobule_pts[0:f_no_lobule_pts[placentone_no]+1,:,placentone_no], \
            no_lobules_to_add,sub_voronoi)
            
        
            

    return [no_lobules,lobules]





def create_cotyledon_o(model,points,voronoi_diagram):
    
    cotyledons = []
    cotyledons_shrunk = []
    
    for placentone_no in range(0,no_placentones):
        
        cotyledon_rsr = removal_sphere_radius( \
            cotyledon_wall_heights[placentone_no])

        curr_pt = model.occ.getMaxTag(0)+1
        curr_line = model.occ.getMaxTag(1)+1
        curr_lineloop = model.occ.getMaxTag(-1)+1
        curr_surface = model.occ.getMaxTag(2)+1
        curr_volume = model.occ.getMaxTag(3)+1
        logger.info("---------------------\n" + \
                     f"Setting up placentone {placentone_no} from foronoi \n" + \
                     f"curr_pt: {curr_pt} \n" + \
                     f"curr_line: {curr_line}")
        
        frni_placentone = frni.ForonoiPlacentone( \
                voronoi_diagram.sites[placentone_no],placenta_radius)

        logger.debug(f"Before: frni_placentone.edges = {frni_placentone.edges}")
        for vertex_no in range(0,frni_placentone.no_vertices):
            v_pt = frni_placentone.vertices[vertex_no,:]
            model.occ.addPoint(v_pt[0],v_pt[1],0.0,DomSize,curr_pt)

            frni_placentone.global_vertex_nos[vertex_no] = curr_pt
            curr_pt = curr_pt+1
        for edge_no in range(0,frni_placentone.no_edges):
            edge_point_nos = numpy.ndarray.copy(frni_placentone.edges[edge_no,:])

            logger.debug(f"Adding edge no {edge_no} with tag {curr_line} and \n" + \
                         f"old edge_point_nos (local foronoi vertex labels) {edge_point_nos[0]} and {edge_point_nos[1]} with \n" + \
                         "new edge_point_nos (global vertex labels) " \
                             f"{frni_placentone.global_vertex_nos[edge_point_nos[0]]} and " \
                             f"{frni_placentone.global_vertex_nos[edge_point_nos[1]]}")
            logger.debug(f"Vertex {edge_point_nos[0]}: {frni_placentone.vertices[edge_point_nos[0],:]} \n" + \
                         f"Vertex {edge_point_nos[1]}: {frni_placentone.vertices[edge_point_nos[1],:]} \n")     

            edge_point_nos[0] = frni_placentone.global_vertex_nos[edge_point_nos[0]]
            edge_point_nos[1] = frni_placentone.global_vertex_nos[edge_point_nos[1]]
            model.occ.addLine(edge_point_nos[0],edge_point_nos[1],curr_line)

            frni_placentone.global_edge_nos[edge_no] = curr_line
            curr_line = curr_line+1

        frni_placentone_shrunk = frni_placentone.create_copy()
        frni_placentone_shrunk.shrink_placentone_fixed_dist(placentone_wall_thickness,model)
        #curr_pt = model.occ.getMaxTag(0)+1
        logger.debug(f"After: frni_placentone.edges = {frni_placentone.edges}")
        # Do same for shrunk placentone
        for vertex_no in range(0,frni_placentone_shrunk.no_vertices):   
            logger.debug(f"in main, shrunk placentone - vertex_no = {vertex_no} with curr_pt = {curr_pt}")
            v_pt = numpy.copy(frni_placentone_shrunk.vertices[vertex_no,:])
            model.occ.addPoint(v_pt[0],v_pt[1],0.0,DomSize,curr_pt)


            frni_placentone_shrunk.global_vertex_nos[vertex_no] = curr_pt
            curr_pt = curr_pt+1




        for edge_no in range(0,frni_placentone_shrunk.no_edges):
            edge_point_nos = numpy.ndarray.copy(frni_placentone_shrunk.edges[edge_no,:])

            logger.debug(f"Adding edge no {edge_no} with tag {curr_line} and \n" + \
                         f"old edge_point_nos (local foronoi vertex labels) {edge_point_nos[0]} and {edge_point_nos[1]} with \n" + \
                         "new edge_point_nos (global vertex labels) " \
                             f"{frni_placentone_shrunk.global_vertex_nos[edge_point_nos[0]]} and {frni_placentone_shrunk.global_vertex_nos[edge_point_nos[1]]}")
            logger.debug(f"Vertex {edge_point_nos[0]}: {frni_placentone_shrunk.vertices[edge_point_nos[0],:]} \n" + \
                         f"Vertex {edge_point_nos[1]}: {frni_placentone_shrunk.vertices[edge_point_nos[1],:]} \n")

            edge_point_nos[0] = frni_placentone_shrunk.global_vertex_nos[edge_point_nos[0]]
            edge_point_nos[1] = frni_placentone_shrunk.global_vertex_nos[edge_point_nos[1]]
            model.occ.addLine(edge_point_nos[0],edge_point_nos[1],curr_line)

            frni_placentone_shrunk.global_edge_nos[edge_no] = curr_line
            curr_line = curr_line+1



        # Create surfaces
        #for edge_no in range(0,frni_placentone.no_edges):
        model.occ.addCurveLoop(frni_placentone.global_edge_nos[:],curr_lineloop)
        model.occ.addCurveLoop(frni_placentone_shrunk.global_edge_nos[:],curr_lineloop+1)

        model.occ.addPlaneSurface([curr_lineloop,curr_lineloop+1],curr_surface)

        model.occ.extrude([(2,curr_surface)],0.0,0.0,placentone_height)
        
        curr_volume_sph = model.occ.addSphere( \
            0.0,0.0,cotyledon_rsr+cotyledon_wall_heights[placentone_no],cotyledon_rsr)

        model.occ.cut([(3,curr_volume)],[(3,curr_volume_sph)])
        model.occ.cut([(3,1)],[(3,curr_volume)])
        model.occ.synchronize()

        frni_placentone.lineloop_no = curr_lineloop
        frni_placentone.surface_no = curr_surface
        frni_placentone.volume_no = curr_volume
        frni_placentone_shrunk.lineloop_no = curr_lineloop+1
        frni_placentone_shrunk.surface_no = curr_surface #Shares a surface as they are not separately made
        frni_placentone_shrunk.volume_no = curr_volume #"" volume ""

        # Store placentones in list
        cotyledons.append(frni_placentone)
        cotyledons_shrunk.append(frni_placentone_shrunk)

    # plots.plot_placentone_list(points,cotyledons)
    
    return [cotyledons,cotyledons_shrunk]






def create_lobules2(model,cotyledons):
    
    no_lobules = 0
    lobules = []
    lobules_shrunk = []
    
    for placentone_no in range(0,no_placentones):
    #for placentone_no in [2]:
    
        lobule_rsr = removal_sphere_radius( \
            lobule_wall_heights[placentone_no])

        logger.info("---------------------\n" + \
                     f"Setting up sub placentones for {placentone_no}")
        print("---------------------\n" + \
                     f"Setting up sub placentones for {placentone_no}")    
        placentone_no_vertices = cotyledons[placentone_no].no_vertices
        placentone_no_edges = cotyledons[placentone_no].no_edges
        placentone_vertices = numpy.copy(cotyledons[placentone_no].vertices)
        placentone_edges = numpy.copy(cotyledons[placentone_no].edges)

        if (not(fixed_lobules)):
            if (cotyledons[placentone_no].boundary_cell):
                sub_voronoi = fns.lloyds_algorithm(placentone_no_vertices,placentone_vertices,no_lobules_outer)
                no_lobules_to_add=no_lobules_outer
            else:
                sub_voronoi = fns.lloyds_algorithm(placentone_no_vertices,placentone_vertices,no_lobules_inner)
                no_lobules_to_add=no_lobules_inner
            no_lobules=no_lobules+no_lobules_to_add
        else:
            if (lobule_foronoi_type == 'standard'):
                pgon = foronoi.Polygon(placentone_vertices)
            elif (lobule_foronoi_type == 'concave'):
                pgon = ConcavePolygon(placentone_vertices)
            else:
                print(f"Error: create_lobules")
                print(f"Unrecognised lobule_foronoi_type: {lobule_foronoi_type}")
                # sys.exit(-1)
                raise Exception(f"Unrecognised lobule_foronoi_type: {lobule_foronoi_type}")
                
            sub_voronoi = foronoi.Voronoi(pgon)
            sub_voronoi.create_diagram(points = \
                    f_lobule_pts[0:f_no_lobule_pts[placentone_no]+1,:,placentone_no])
            no_lobules_to_add = f_no_lobule_pts[placentone_no]
            no_lobules = no_lobules + f_no_lobule_pts[placentone_no]

            # frni.visualise_voronoi(sub_voronoi)

        for site_no in range(0,no_lobules_to_add):

            curr_pt = model.occ.getMaxTag(0)+1
            curr_line = model.occ.getMaxTag(1)+1
            curr_lineloop = model.occ.getMaxTag(-1)+1
            curr_surface = model.occ.getMaxTag(2)+1
            curr_volume = model.occ.getMaxTag(3)+1

            frni_lobule = frni.ForonoiPlacentone(sub_voronoi.sites[site_no],placenta_radius)

            for vertex_no in range(0,frni_lobule.no_vertices):
                v_pt = frni_lobule.vertices[vertex_no,:]
                model.occ.addPoint(v_pt[0],v_pt[1],0.0,DomSize,curr_pt)

                frni_lobule.global_vertex_nos[vertex_no] = curr_pt
                curr_pt = curr_pt+1
            for edge_no in range(0,frni_lobule.no_edges):
                edge_point_nos = numpy.ndarray.copy(frni_lobule.edges[edge_no,:])

                logger.debug(f"Adding edge no {edge_no} with tag {curr_line} and \n" + \
                             f"old edge_point_nos (local foronoi vertex labels) {edge_point_nos[0]} and {edge_point_nos[1]} with \n" + \
                             "new edge_point_nos (global vertex labels) " \
                                 f"{frni_lobule.global_vertex_nos[edge_point_nos[0]]} and {frni_lobule.global_vertex_nos[edge_point_nos[1]]}")
                logger.debug(f"Vertex {edge_point_nos[0]}: {frni_lobule.vertices[edge_point_nos[0],:]} \n" + \
                             f"Vertex {edge_point_nos[1]}: {frni_lobule.vertices[edge_point_nos[1],:]} \n")     

                edge_point_nos[0] = frni_lobule.global_vertex_nos[edge_point_nos[0]]
                edge_point_nos[1] = frni_lobule.global_vertex_nos[edge_point_nos[1]]
                model.occ.addLine(edge_point_nos[0],edge_point_nos[1],curr_line)

                frni_lobule.global_edge_nos[edge_no] = curr_line
                curr_line = curr_line+1

            frni_lobule_shrunk = frni_lobule.create_copy()
            frni_lobule_shrunk.shrink_placentone_fixed_dist(lobule_wall_thickness,model)
            curr_pt = model.occ.getMaxTag(0)+1
            # Do same for shrunk placentone
            for vertex_no in range(0,frni_lobule_shrunk.no_vertices):

                v_pt = frni_lobule_shrunk.vertices[vertex_no,:]
                model.occ.addPoint(v_pt[0],v_pt[1],0.0,DomSize,curr_pt)

                frni_lobule_shrunk.global_vertex_nos[vertex_no] = curr_pt
                curr_pt = curr_pt+1
            for edge_no in range(0,frni_lobule_shrunk.no_edges):
                edge_point_nos = numpy.ndarray.copy(frni_lobule_shrunk.edges[edge_no,:])

                logger.debug(f"Adding edge no {edge_no} with tag {curr_line} and \n" + \
                             f"old edge_point_nos (local foronoi vertex labels) {edge_point_nos[0]} and {edge_point_nos[1]} with \n" + \
                             "new edge_point_nos (global vertex labels) " \
                                 f"{frni_lobule_shrunk.global_vertex_nos[edge_point_nos[0]]} and {frni_lobule_shrunk.global_vertex_nos[edge_point_nos[1]]}")
                logger.debug(f"Vertex {edge_point_nos[0]}: {frni_lobule_shrunk.vertices[edge_point_nos[0],:]} \n" + \
                             f"Vertex {edge_point_nos[1]}: {frni_lobule_shrunk.vertices[edge_point_nos[1],:]} \n")

                edge_point_nos[0] = frni_lobule_shrunk.global_vertex_nos[edge_point_nos[0]]
                edge_point_nos[1] = frni_lobule_shrunk.global_vertex_nos[edge_point_nos[1]]
                model.occ.addLine(edge_point_nos[0],edge_point_nos[1],curr_line)

                frni_lobule_shrunk.global_edge_nos[edge_no] = curr_line
                curr_line = curr_line+1


            # Create surfaces
            model.occ.addCurveLoop(frni_lobule.global_edge_nos[:],curr_lineloop)
            model.occ.addCurveLoop(frni_lobule_shrunk.global_edge_nos[:],curr_lineloop+1)

            model.occ.addPlaneSurface([curr_lineloop,curr_lineloop+1],curr_surface)

            # Can't do curr_volume = extrude as it returns both boundary and surface list
            model.occ.extrude([(2,curr_surface)],0.0,0.0,placentone_height)
            
            curr_volume_sph = model.occ.addSphere( \
                0.0,0.0,lobule_rsr+lobule_wall_heights[placentone_no],lobule_rsr)

            model.occ.cut([(3,curr_volume)],[(3,curr_volume_sph)])
                
            model.occ.cut([(3,1)],[(3,curr_volume)])
            model.occ.synchronize()
            
            # Check that the cut has worked properly - there should only be 2 volumes here
            # But sometimes the cut just separates the wall in two
            model.occ.synchronize()
            while (model.occ.getMaxTag(3) > 1):
                print(f"WARNING: Lobule wall cut has not worked properly")
                print(f"Trying to fix it by removing highest-index volume")
                model.occ.remove([(3,model.occ.getMaxTag(3))],recursive=True)
                
            # Store sub placentones in list
            lobules.append(frni_lobule)
            lobules_shrunk.append(frni_lobule_shrunk)
            

    return [no_lobules,lobules,lobules_shrunk]