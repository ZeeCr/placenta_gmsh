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




def create_ellipsoid_cavities(model,cavities):
    
    model.occ.synchronize()        
        
    for cavity_no, cavity in enumerate(cavities):
        
        removal_vol = model.occ.addSphere(0.0,0.0,initial_sphere_radius,initial_sphere_radius)       

        print(f"Adding cavity {cavity_no}")

        cavity.update_major_axis(placenta_height,cavity_height_divisor)
        cavity.update_cavity(minor_axis = cavity_minor_axis)

        cavity_vol = model.occ.addSphere(0.0,0.0,0.0,radius=1.0) 
        
        model.occ.remove([(3,cavity_vol)], recursive=False)     

        curr_surf = model.occ.getMaxTag(2)
        
        # Stretch
        model.occ.dilate([(2,curr_surf)],0.0,0.0,0.0,\
                         cavity.minor_axis,cavity.minor_axis,cavity.major_axis)       
        
        norm = cavity.orientation_normal
        centre = cavity.centre
        
        denom = math.sqrt(1.0+(norm[0]/norm[2])**2)
        entry11 = 1.0/denom
        entry12 = -norm[1]*norm[0]/(norm[2]*denom)
        entry13 = norm[0]
        entry21 = 0.0
        entry22 = norm[2]*denom
        entry23 = norm[1]
        entry31 = -norm[0]/(norm[2]*denom)
        entry32 = -norm[1]/denom
        entry33 = norm[2]
        
        # Rotate + translate
        transformationMatrix = numpy.array([[entry11,entry12,entry13,centre[0]],
                                         [entry21,entry22,entry23,centre[1]],
                                         [entry31,entry32,entry33,centre[2]],
                                         [0, 0, 0, 1]])
        gmsh.model.occ.affineTransform([(2,curr_surf)], transformationMatrix.flatten())

        # This creates a hemiellipsoid surface only inside the placenta
        model.occ.intersect([(2,curr_surf)], [(3,removal_vol)], removeObject = True, removeTool = True)

        placenta_vol_no = fns.determine_largest_gmsh_vol(model)
        
        # Not really sure what this does, I guess chops out the shape from the placenta but leaves the hemiellipsoid
        model.occ.intersect([(3, placenta_vol_no)], [(2,curr_surf)], tag=-1, removeObject=False, removeTool=True)     

        # Creates two volumes with a conformal surface
        model.occ.fragment([(3, placenta_vol_no)], [(2,curr_surf)], removeObject=True, removeTool=True)

        model.occ.removeAllDuplicates()

        COM = model.occ.getCenterOfMass(2,curr_surf)
        cavity.update_cavity(COM=COM)

    model.occ.synchronize() 
    
    return None





def create_septal_fragment(model,outlet_faces,surf_COMs_to_ignore):
    
    model.occ.synchronize()        
        
    for outlet_no, outlet in enumerate(outlet_faces):

        curr_surf = model.occ.getMaxTag(2) + 1
        curr_vol = model.occ.getMaxTag(3) + 1
        
        centre = outlet.centre
        normal = outlet.outward_unit_normal
        length = outlet.cylinder_length
    
        inside_face_centre = centre-length*normal
        
        placenta_vol_no = fns.determine_largest_gmsh_vol(model)
        
        line_no = fns.find_line_with_centre(model,inside_face_centre)
        
        try:
            debug_int = 1
            model.occ.addCurveLoop([line_no])
            debug_int = 2
            model.occ.addPlaneSurface([model.occ.getMaxTag(-1)])
        except:
            print(f"ERROR: create_septal_fragment")
            if (debug_int == 1):
                print(f"Failed to create curve loop")
            elif (debug_int == 2):
                print(f"Failed to create plane surface")
            model.occ.synchronize()
            # gmsh.fltk.run()
            gmsh.finalize()
            # sys.exit(-1)
            raise Exception("Error in create_septal_fragment")
            
        COM = model.occ.getCenterOfMass(2,model.occ.getMaxTag(2))
        model.occ.fragment([(3,placenta_vol_no)],[(2,model.occ.getMaxTag(2))])
        
        model.occ.removeAllDuplicates()

        surf_COMs_to_ignore.append(COM)

    model.occ.synchronize()
            
    return model,surf_COMs_to_ignore
        
def create_septal_fragment2(model,outlet_faces,surf_COMs_to_ignore):
    
    model.occ.synchronize()        
        
    for outlet_no, outlet in enumerate(outlet_faces):

            curr_surf = model.occ.getMaxTag(2) + 1
            curr_vol = model.occ.getMaxTag(3) + 1
            
            centre = outlet.centre
            normal = outlet.outward_unit_normal
            length = outlet.cylinder_length
            radius = outlet.cylinder_radius+outlet.cylinder_fillet
            pt_on_placenta_wall = copy.deepcopy(centre-length*normal)
            
            diff = copy.deepcopy(centre-pt_on_placenta_wall)
            
            model.occ.addCylinder(*pt_on_placenta_wall, \
                                    *diff,radius,curr_vol)
            
            model.occ.remove([(3,curr_vol)],recursive=False)
            
            cyl_surfs = [(2,curr_surf+i) for i in range(0,3)]
            
            placenta_vol_no = fns.determine_largest_gmsh_vol(model)
            # Not really sure what this does, I guess chops out the shape from the placenta but leaves the cylinder
            model.occ.intersect([(3, placenta_vol_no)], cyl_surfs, tag=-1, removeObject=False, removeTool=True)
            
            # Creates two volumes with a conformal surface
            model.occ.fragment([(3, placenta_vol_no)], [(2,curr_surf+2)], removeObject=True, removeTool=True)

            model.occ.removeAllDuplicates()
            
            COM = model.occ.getCenterOfMass(2,curr_surf+2)
            surf_COMs_to_ignore.append(COM)
    
            model.occ.synchronize()
            
    return model,surf_COMs_to_ignore