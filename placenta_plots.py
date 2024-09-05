import math
import importlib
import copy

import numpy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects

import placenta_const

from placenta_const import *

importlib.reload(placenta_const)

royalblue = [0, 20/256, 82/256]

def plot_placentone_list(points,placentone_list):

    shift = placenta_voronoi_outer_radius_offset
    fig = plt.figure()
    ax = fig.add_axes([1,1,2,2], aspect=1)
    ax.set_xlim(-placenta_radius-shift,placenta_radius+shift)
    ax.set_ylim(-placenta_radius-shift,placenta_radius+shift)
    
    #ax.plot(0.0, 0.0, c='C0', lw=10, label="Blue signal", zorder=10)
    #ax.plot(0.0, 1.0, c='C1', lw=10, label="Orange signal")
    #ax.plot([-placenta_radius,placenta_radius,placenta_radius,-placenta_radius,-placenta_radius], \
    #        [-placenta_radius,-placenta_radius,placenta_radius,placenta_radius,-placenta_radius], zorder=2, color='black', linewidth=1, label="Orange signal")
    c = matplotlib.patches.Circle((0.0, 0.0), radius=placenta_radius, clip_on=False, zorder=1, linewidth=1.0,
               edgecolor='black', facecolor='none',
               path_effects=[matplotlib.patheffects.withStroke(linewidth=5, foreground='white')])
    ax.add_artist(c)    
    
    for placentone_no,placentone in enumerate(placentone_list):
        placentone_no_vertices = placentone.no_vertices
        placentone_no_edges = placentone.no_edges
        placentone_vertices = numpy.copy(placentone.vertices)
        placentone_edges = numpy.copy(placentone.edges)
        placentone_centroid = numpy.copy(placentone.centroid)
        
        ax.text(*placentone_centroid,str(placentone_no), zorder=10,
            ha='center', va='top', weight='bold', color='red',
            fontfamily='Courier New',fontsize=24)
        
        ax.scatter(placentone_vertices[:,0],placentone_vertices[:,1], zorder=8, color='blue')
        ax.scatter(*placentone_centroid,zorder=8, color='red')
        
        for edge_no in range(0,placentone_no_edges):
            ax.text(*(placentone_vertices[placentone_edges[edge_no,0],:]+numpy.array([0.0,2.0])),str(0), zorder=10,
                ha='center', va='top', weight='bold', color='green',
                fontfamily='Courier New',fontsize=12)            
            ax.text(*(placentone_vertices[placentone_edges[edge_no,1],:]+numpy.array([0.0,-2.0])),str(1), zorder=10,
                ha='center', va='top', weight='bold', color='green',
                fontfamily='Courier New',fontsize=12)
            X = numpy.array([placentone_vertices[placentone_edges[edge_no,0],0],placentone_vertices[placentone_edges[edge_no,1],0]])
            Y = numpy.array([placentone_vertices[placentone_edges[edge_no,0],1],placentone_vertices[placentone_edges[edge_no,1],1]])
            edge_c = numpy.array([(X[1]+X[0])/2.0,(Y[1]+Y[0])/2.0])
            ax.plot(X,Y,c='g',zorder=9)
            
            edge_c_to_centroid = numpy.copy(placentone_centroid-edge_c)
            edge_c = edge_c + 0.25*edge_c_to_centroid
            ax.text(edge_c[0],edge_c[1],str(edge_no), zorder=10,
                ha='center', va='top', weight='bold', color='black',
                fontfamily='Courier New',fontsize=24)
            
            
    
    for point_no,point in enumerate(points):
        ax.text(*point,str(point_no), zorder=10,
        ha='center', va='top', weight='bold', color='pink',
        fontfamily='Courier New',fontsize=24)
    
        ax.scatter(*point, zorder=8, color='pink')
    
    
    
    
    plt.show()
    
    
    
    
    return None

def plot_edge_set(edge_set,placentone_list = None):

    shift = placenta_voronoi_outer_radius_offset
    fig = plt.figure()
    ax = fig.add_axes([1,1,2,2], aspect=1)
    ax.set_xlim(-placenta_radius-shift,placenta_radius+shift)
    ax.set_ylim(-placenta_radius-shift,placenta_radius+shift)
    
    #ax.plot(0.0, 0.0, c='C0', lw=10, label="Blue signal", zorder=10)
    #ax.plot(0.0, 1.0, c='C1', lw=10, label="Orange signal")
    ax.plot([-placenta_radius,placenta_radius,placenta_radius,-placenta_radius,-placenta_radius], \
            [-placenta_radius,-placenta_radius,placenta_radius,placenta_radius,-placenta_radius], zorder=2, color='black', linewidth=1, label="Orange signal")
    c = matplotlib.patches.Circle((0.0, 0.0), radius=placenta_radius, clip_on=False, zorder=1, linewidth=1.0,
               edgecolor='black', facecolor='none',
               path_effects=[matplotlib.patheffects.withStroke(linewidth=5, foreground='white')])
    ax.add_artist(c)    
    
    for edge_no in range(0,edge_set.no_edges):
        
        v_no = copy.deepcopy(edge_set.edge[edge_no,:])
        vertices = numpy.empty((2,2))
        vertices[0,:] = edge_set.node_set.node[v_no[0],:]
        vertices[1,:] = edge_set.node_set.node[v_no[1],:]
        
        midp = 0.5*(vertices[1,:]+vertices[0,:])
        
        ax.text(*midp,str(edge_no), zorder=10,
            ha='center', va='top', weight='bold', color='g',
            fontfamily='Courier New',fontsize=24)
    
        # Edge line
        X = numpy.array([vertices[0,0],vertices[1,0]])
        Y = numpy.array([vertices[0,1],vertices[1,1]])
        ax.plot(X,Y,c='g',zorder=9)
        
        # Vector showing edge dir
        vec_line = copy.deepcopy(vertices[0,:] + \
            0.1*edge_set.edge_length[edge_no]*edge_set.edge_dir[edge_no,:])
        X = numpy.array([vertices[0,0],vec_line[0]])
        Y = numpy.array([vertices[0,1],vec_line[1]])
        ax.plot(X,Y,c='r',zorder=9)
        ax.text(*vec_line,str(edge_no), zorder=10,
            ha='center', va='top', weight='bold', color='r',
            fontfamily='Courier New',fontsize=24)
        
    if (placentone_list is not None):
    
        for cell_no in range(0,len(placentone_list)):
            
            for loc_e_no in range(0,edge_set.no_cell_edges[cell_no]):
                
                glob_e_no = edge_set.cell_edges[cell_no,loc_e_no]
                [v0,v1] = edge_set.get_vertices_from_cell_edge(cell_no,loc_e_no)
                midp = v0[:] + 0.33*(v1[:]-v0[:])
                midp_c = copy.deepcopy(placentone_list[cell_no].centroid - midp)
                midp_c = copy.deepcopy(midp + 0.1*midp_c)
                
                # Vector showing cell-wise labels
                X = numpy.array([midp[0],midp_c[0]])
                Y = numpy.array([midp[1],midp_c[1]])
                ax.plot(X,Y,c='orange',zorder=9)
                ax.text(*midp_c,str(loc_e_no)+", "+str(glob_e_no), zorder=10,
                    ha='center', va='top', weight='bold', color='orange',
                    fontfamily='Courier New',fontsize=16)
            
    plt.show()
    
    return None

def plot_cotyeldon_lobule_list(points, cotyeldon_list, lobule_list, \
        cotyledon_artery_number = None, cotyledon_vein_number = None):

    shift = placenta_voronoi_outer_radius_offset
    fig = plt.figure()
    ax = fig.add_axes([1,1,2,2], aspect=1)
    ax.set_xlim(-placenta_radius-shift,placenta_radius+shift)
    ax.set_ylim(-placenta_radius-shift,placenta_radius+shift)
    
    #ax.plot(0.0, 0.0, c='C0', lw=10, label="Blue signal", zorder=10)
    #ax.plot(0.0, 1.0, c='C1', lw=10, label="Orange signal")
    c = matplotlib.patches.Circle((0.0, 0.0), radius=placenta_radius, clip_on=False, zorder=1, linewidth=1.0,
               edgecolor='black', facecolor='none',
               path_effects=[matplotlib.patheffects.withStroke(linewidth=5, foreground='white')])
    ax.add_artist(c)    
    
    # Cotyledon
    for placentone_no,placentone in enumerate(cotyeldon_list):
        placentone_no_vertices = placentone.no_vertices
        placentone_no_edges = placentone.no_edges
        placentone_vertices = numpy.copy(placentone.vertices)
        placentone_edges = numpy.copy(placentone.edges)
        placentone_centroid = numpy.copy(placentone.centroid)
        
        ax.text(placentone_centroid[0] + 0.1*abs(placentone_centroid[0]), placentone_centroid[1], \
            str(placentone_no + 1), zorder=10,
            ha='center', va='top', weight='bold', color='red',
            fontfamily='Courier New',fontsize=24)
        
        ax.scatter(*placentone_centroid, \
            zorder=8, color='red')
        
        for edge_no in range(0,placentone_no_edges):
            X = numpy.array([placentone_vertices[placentone_edges[edge_no,0],0],placentone_vertices[placentone_edges[edge_no,1],0]])
            Y = numpy.array([placentone_vertices[placentone_edges[edge_no,0],1],placentone_vertices[placentone_edges[edge_no,1],1]])
            edge_c = numpy.array([(X[1]+X[0])/2.0,(Y[1]+Y[0])/2.0])
            ax.plot(X,Y,c='g',zorder=9) 
            
        if (cotyledon_artery_number is not None):
            ax.text(placentone_centroid[0] + 0.1*placenta_radius, \
                placentone_centroid[1] + 0.1*placenta_radius, \
                f"Arteries: {str(cotyledon_artery_number[placentone_no])}", zorder=10,
                ha='center', va='top', weight='bold', color='blue',
                fontfamily='Courier New',fontsize=18)
        if (cotyledon_vein_number is not None):
            ax.text(placentone_centroid[0] + 0.1*placenta_radius, \
                placentone_centroid[1] - 0.1*placenta_radius, \
                f"Veins: {str(cotyledon_vein_number[placentone_no])}", zorder=10,
                ha='center', va='top', weight='bold', color='blue',
                fontfamily='Courier New',fontsize=18)
    
    # Lobules
    for lobule_no,lobule in enumerate(lobule_list):
        lobule_no_vertices = lobule.no_vertices
        lobule_no_edges = lobule.no_edges
        lobule_vertices = numpy.copy(lobule.vertices)
        lobule_edges = numpy.copy(lobule.edges)
        lobule_centroid = numpy.copy(lobule.centroid)
        
        for edge_no in range(0,lobule_no_edges):
            X = numpy.array([lobule_vertices[lobule_edges[edge_no,0],0],lobule_vertices[lobule_edges[edge_no,1],0]])
            Y = numpy.array([lobule_vertices[lobule_edges[edge_no,0],1],lobule_vertices[lobule_edges[edge_no,1],1]])
            edge_c = numpy.array([(X[1]+X[0])/2.0,(Y[1]+Y[0])/2.0])
            ax.plot(X,Y,c='g',zorder=19, alpha = 0.1)    
    
    
    plt.show()
    
    return None