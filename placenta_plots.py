import math
import importlib

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
    ax.plot([-placenta_radius,placenta_radius,placenta_radius,-placenta_radius,-placenta_radius], \
            [-placenta_radius,-placenta_radius,placenta_radius,placenta_radius,-placenta_radius], zorder=2, color='black', linewidth=1, label="Orange signal")
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