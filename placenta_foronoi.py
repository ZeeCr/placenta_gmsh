import importlib
import math

import numpy
import copy

import foronoi

import placenta_classes as clses
import placenta_fns as fns
import placenta_const as const

importlib.reload(clses)
importlib.reload(const)

def visualise_voronoi(voronoi):
    
    vis = foronoi.Visualizer(voronoi, canvas_offset=0)
    vis.plot_sites(show_labels=True)
    vis.plot_edges(show_labels=True)
    vis.plot_vertices()
    vis.plot_border_to_site()
    vis.show()
    
    return None

class ForonoiPlacentone(clses.Placentone):

    def __init__(self,foronoi_site,circle_r=0.0):
        self.foronoi_site = foronoi_site
        
        [no_vertices,no_edges,vertices,edges] = self.convert_foronoi_struct()
        
        # Horrible implementation - if this isn't here, then cells with no intersections don't declare this array
        vertices_for_centroid = numpy.copy(vertices)
        
        intersection = False
        # I only want to call this when I set up the initial lobule structure, not during copy etc.
        if (circle_r > const.tol):
            [intersection,intersection_edges] = fns.does_cell_intersect_circle(no_vertices,no_edges,vertices,edges,circle_r)
            if (intersection):
                [vertices,edges] = fns.reorder_vertices_first_edge_intersects(no_vertices,no_edges,vertices,edges, \
                    circle_r)
                
                # Recalculate intersection edge numbers
                [placeholder,intersection_edges] = fns.does_cell_intersect_circle(no_vertices,no_edges,vertices,edges,circle_r)
                
                # In the event that there is only 1 edge connecting the two intersection vertices, add a vertex and edge between
                if (intersection_edges[1] == intersection_edges[0] + 2):
                    pt_to_add = copy.deepcopy(0.5*(vertices[intersection_edges[0] + 1,:] + vertices[intersection_edges[1],:]))
                    [no_vertices,no_edges,vertices,edges] = fns.add_vertex_after_idx_to_cell(intersection_edges[0] + 1,no_vertices,no_edges, \
                        vertices,edges,pt_to_add)
                    # Recalculate intersection edge numbers
                    [placeholder,intersection_edges] = fns.does_cell_intersect_circle(no_vertices,no_edges,vertices,edges,circle_r)
                # Move point to outside adjacent points' midpoint for greatest 'coverage' of circle periphery
                elif (intersection_edges[1] == intersection_edges[0] + 3):
                    pt_to_add = copy.deepcopy(0.5*(vertices[intersection_edges[0] + 1,:] + vertices[intersection_edges[1],:]))
                    vertices[intersection_edges[0] + 2,:] = pt_to_add
                
                #[no_vertices,no_edges,vertices,edges] = fns.del_vertices_between_intersection_vertices(no_vertices,no_edges,vertices,edges,circle_r)
                [vertices,edges,vertices_for_centroid] = fns.shrink_outside_vertices(no_vertices,no_edges,vertices,edges,circle_r,intersection_edges)
                
        super().__init__(no_vertices,no_edges)

        self.vertices = numpy.ndarray.copy(vertices)
        self.edges = numpy.ndarray.copy(edges)
        #self.centroid = fns.cal_centroid(no_vertices,vertices_for_centroid)
        self.centroid = fns.cal_centre_polygon(no_vertices,vertices_for_centroid)
        
        if (intersection):
            self.boundary_cell = 1
        
    def convert_foronoi_struct(self):
        
        [no_vertices,no_edges,vertices,edges] = fns.convert_foronoi_site(self.foronoi_site)

        self.no_vertices = no_vertices
        self.no_edges = no_edges
        
        return [no_vertices,no_edges,vertices,edges]        
    
    def create_copy(self):
        
        obj_copy = copy.deepcopy(self)
        '''
        copy = ForonoiPlacentone(self.foronoi_site,copy=True)
        
        super().__init__(self.no_vertices,self.no_edges)
        
        copy.vertices = numpy.ndarray.copy(self.vertices)
        copy.edges = numpy.ndarray.copy(self.edges)
        copy.centroid = numpy.ndarray.copy(self.centroid)
        
        copy.global_vertex_nos = numpy.ndarray.copy(self.global_vertex_nos)
        copy.global_edge_nos = numpy.ndarray.copy(self.global_edge_nos)

        copy.lineloop_no = self.lineloop_no
        copy.surface_no = self.surface_no
        copy.volume_no = self.volume_no
        '''
        return obj_copy
        
    # Python was a mistake - creating virtual function which exists only to call base class function
    def shrink_placentone_fixed_dist(self,dist,model) -> None:
        
        super().shrink_placentone_fixed_dist(dist,model)
        
    def get_vertices_edges(self):
        
        return super().get_vertices_edges()
    