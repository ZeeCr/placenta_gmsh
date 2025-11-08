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


def add_spiral_arteries_and_lobule_veins(model,no_lobules, \
        cotyledons,lobules,no_inlets,no_lobule_veins,no_outlets,outlet_faces):
    
    no_cavities = 0
    bounding_box = clses.BoundingBox()
    bounding_boxes = []
    face = clses.Face()
    inlet_faces = []
    cavity = clses.Cavity()
    cavities = []
    
    # Track artery info across placentones
    to_store_artery_added = numpy.empty(no_lobules,dtype=int)
    # Track basal plate vein info across placentones
    to_store_stop_at_basal_veins_added = numpy.empty(0,dtype=int)
    to_store_no_vein_to_add = numpy.empty(0,dtype=int)
    to_store_vein_power = numpy.empty(0,dtype=int)
    
    #for placentone_no in [0,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18]:
    #for placentone_no in [0]:
    #for placentone_no in range(0,no_placentones):
    for placentone_no in range(0,no_lobules):
        
        # The basal vein part needs info from here, so can't just comment it all out
        artery_prob = numpy.random.randint(1,11)
        if (artery_prob <= artery_bias):
            add_artery_to_lobule = 1
        else:
            add_artery_to_lobule = 0
        to_store_artery_added[placentone_no] = add_artery_to_lobule
        
        print(f"Adding sub placentone {placentone_no} / {no_lobules-1}")

        # Add pipes
        curr_pt = model.occ.getMaxTag(0)+1
        curr_line = model.occ.getMaxTag(1)+1
        curr_lineloop = model.occ.getMaxTag(-1)+1
        curr_surface = model.occ.getMaxTag(2)+1
        curr_volume = model.occ.getMaxTag(3)+1

        placentone_pt_on_circle = numpy.zeros(3)
        placentone_pt_on_circle[0:2] = lobules[placentone_no].centroid
        placentone_pt_on_circle[2] = sphere_surface_eval( \
            *placentone_pt_on_circle[0:2],initial_sphere_radius,0.0)
        # Circle centre is the original huge sphere used to create the placenta, i.e.
        # this is the outward normal vector to the placenta's surface
        placenta_normal_vector = numpy.array([placentone_pt_on_circle[0]-0.0, \
                                          placentone_pt_on_circle[1]-0.0, \
                                          placentone_pt_on_circle[2]-initial_sphere_radius])
        placenta_normal_vector = fns.normalise_vector(3,placenta_normal_vector)

        # Radius vector from the centre of the placenta plane at z=placenta_height to the centroid
        placenta_plane_radius_vector_2d = numpy.zeros(2)
        placenta_plane_radius_vector_2d = placentone_pt_on_circle[0:2]
        placenta_plane_radius_vector_2d = fns.normalise_vector(2,placenta_plane_radius_vector_2d)
        placenta_tangent = numpy.array([-placenta_plane_radius_vector_2d[1],placenta_plane_radius_vector_2d[0]])

        ##################### ARTERY SETUP ########################
        artery_pt_out_placenta = numpy.array([placentone_pt_on_circle[0]+septal_vessel_length*placenta_normal_vector[0], \
                             placentone_pt_on_circle[1]+septal_vessel_length*placenta_normal_vector[1], \
                             placentone_pt_on_circle[2]+septal_vessel_length*placenta_normal_vector[2]])

        # Start is outside the sphere, end is inside
        centre_to_cyl_start_dist = numpy.linalg.norm(artery_pt_out_placenta-initial_sphere_centre)
        centre_to_cyl_end_dist = math.sqrt(initial_sphere_radius**2 - septal_artery_radius**2)
        cyl_dist = centre_to_cyl_start_dist-centre_to_cyl_end_dist
        artery_pt_on_placenta = copy.deepcopy(artery_pt_out_placenta - cyl_dist*placenta_normal_vector)
        # Replace the dx,dy,dz by something smarter - 
        # i.e. halfway between placenta height and placenta surface point to make sure it doesn't go above placenta z plane
        '''
        model.occ.addCylinder(artery_pt_out_placenta[0],artery_pt_out_placenta[1],artery_pt_out_placenta[2], \
                                   -(12.0/10.0)*septal_vessel_length*placenta_normal_vector[0], \
                                   -(12.0/10.0)*septal_vessel_length*placenta_normal_vector[1], \
                                   -(12.0/10.0)*septal_vessel_length*placenta_normal_vector[2], \
                                   septal_artery_radius,curr_volume)
        '''
        
        if (add_artery_to_lobule == 1):
            
            # Update cavity with info for ellipsoid, reverse normal_vector sign for inward
            cavity.update_cavity(centre = placentone_pt_on_circle,orientation_normal = -placenta_normal_vector)
            cavities.append(copy.deepcopy(cavity))
            
            model.occ.addCylinder( \
                artery_pt_out_placenta[0],artery_pt_out_placenta[1],artery_pt_out_placenta[2], \
                -cyl_dist*placenta_normal_vector[0], \
                -cyl_dist*placenta_normal_vector[1], \
                -cyl_dist*placenta_normal_vector[2], \
                septal_artery_radius,curr_volume)
            cyl_face = model.occ.getMaxTag(2)
            bb = model.occ.getBoundingBox(2,cyl_face)

            bounding_box.update_bounds(bb)

            face_no = fns.get_face_in_bb(bounding_box)
            bounding_boxes.append(copy.deepcopy(bounding_box))

            model.occ.fuse([(3,1)],[(3,curr_volume)])

            try:

                # An occasional error means that the centroid is outside the placental domain
                fns.check_unique_vol(model)
                
                max_vol_tag = model.occ.getMaxTag(3)
                gmsh.model.occ.fillet([max_vol_tag], \
                    [fns.find_line_with_centre(model,artery_pt_on_placenta)], \
                    [septal_artery_funnel_radius])

                # Incrememnt inlet
                no_inlets = no_inlets+1
                
                # Increment cavities
                no_cavities = no_cavities + 1

                face.update_face(cyl_face,artery_pt_out_placenta,placenta_normal_vector)
                inlet_faces.append(copy.deepcopy(face))    

            except Exception:

                print(f"Error setting up artery in sub placentone")
                model.occ.remove([(3,curr_volume)], recursive=True)
                
                # Pop cavities as it's appended earlier assuming that the artery will be placed
                cavities.pop()





        # BASSAL PLATE VEINS #

        no_veins_added = 0
        no_vein_to_add = 0
        for i in range(1,3):
            if (numpy.random.randint(1,11) <= basal_vein_bias):
                no_vein_to_add = no_vein_to_add + 1
        #no_vein_to_add = numpy.random.randint(0,3)
        stop_at_veins_added = 2
        if (no_vein_to_add == 1 or stop_at_veins_added == 1):
            vein_power = numpy.random.randint(1,3)
        else:
            vein_power = 0
        
        if (stored_basal_veins):
            print(f"stored_basal_veins == {stored_basal_veins}")
            stop_at_veins_added = stored_stop_at_basal_veins_added[placentone_no]
            no_vein_to_add = stored_no_basal_veins_to_add[placentone_no]
            vein_power = stored_basal_vein_powers[placentone_no]

        print(f"Placentone_no: {placentone_no}")
        print(f"stop_at_veins_added: {stop_at_veins_added}")
        print(f"no_vein_to_add: {no_vein_to_add}")
        print(f"vein_power: {vein_power}")
        
        # Store then print out at the end
        to_store_stop_at_basal_veins_added = \
            numpy.append(to_store_stop_at_basal_veins_added,stop_at_veins_added)
        to_store_no_vein_to_add = \
            numpy.append(to_store_no_vein_to_add,no_vein_to_add)
        to_store_vein_power = numpy.append(to_store_vein_power,vein_power)
        
        #####################
        ##### ADD VEINS #####    
        #####################

        for vein_no in range(1,no_vein_to_add+1,1):
            vein_side = (-1)**(vein_no+vein_power)
            vein_pt_on_placenta = copy.deepcopy(artery_pt_on_placenta)
            vein_pt_on_placenta[0:2] = vein_pt_on_placenta[0:2] + \
                vein_side*septal_vessel_separation*placenta_tangent[0:2]

            # Circle centre is the original huge sphere used to create the placenta, i.e.
            # this is the outward normal vector to the placenta's surface
            vein_normal_vector = numpy.array([vein_pt_on_placenta[0]-0.0, \
                                              vein_pt_on_placenta[1]-0.0, \
                                              vein_pt_on_placenta[2]-initial_sphere_radius])
            vein_normal_vector = fns.normalise_vector(3,vein_normal_vector)

            shift_angle = math.acos(numpy.dot(placenta_normal_vector,vein_normal_vector))
            adjacent_dist = initial_sphere_radius*math.cos(shift_angle)
            shift_dist = initial_sphere_radius - adjacent_dist
            shift_radius = shift_dist/math.cos(shift_angle)
            shift_coeff = 1.1
            
            vein_surface_pt = copy.deepcopy(vein_pt_on_placenta-shift_radius*vein_normal_vector)
            vein_outside_pt = vein_surface_pt+septal_vessel_length*vein_normal_vector
            
            centre_to_vein_outside_pt = numpy.linalg.norm(vein_outside_pt-initial_sphere_centre)
            centre_to_vein_inside_pt = math.sqrt(initial_sphere_radius**2 - septal_vein_radius**2)
            lobule_vein_length = centre_to_vein_outside_pt-centre_to_vein_inside_pt
            vein_inside_pt = copy.deepcopy(vein_outside_pt-lobule_vein_length*vein_normal_vector)
            
            # Crude estimate at edges of the fillet circle
            placenta_tangent_3d = numpy.array([*placenta_tangent,0.0])
            placenta_normal_3d = numpy.array([*placenta_plane_radius_vector_2d,0.0])
            cotyledon_nos = numpy.empty(4,dtype=int)
            lobule_nos = numpy.empty(4,dtype=int)
            front_est = vein_inside_pt+(septal_vein_radius+septal_vein_funnel_radius)*placenta_tangent_3d
            left_est = vein_inside_pt+(septal_vein_radius+septal_vein_funnel_radius)*placenta_normal_3d
            right_est = vein_inside_pt-(septal_vein_radius+septal_vein_funnel_radius)*placenta_normal_3d
            lobule_nos[0] = fns.find_lobule_no(front_est,lobules)
            lobule_nos[1] = fns.find_lobule_no(vein_inside_pt,lobules)
            lobule_nos[2] = fns.find_lobule_no(left_est,lobules)
            lobule_nos[3] = fns.find_lobule_no(right_est,lobules)
            cotyledon_nos[0] = fns.find_cotyledon_no(front_est,cotyledons)
            cotyledon_nos[1] = fns.find_cotyledon_no(vein_inside_pt,cotyledons)
            cotyledon_nos[2] = fns.find_cotyledon_no(left_est,cotyledons)
            cotyledon_nos[3] = fns.find_cotyledon_no(right_est,cotyledons)

            # All points in same lobule so can create cyl
            if (lobule_nos[0] != -1 and cotyledon_nos[0] != -1 \
                    and numpy.all(lobule_nos == lobule_nos[0]) and numpy.all(cotyledon_nos == cotyledon_nos[0])):
                    
                model.occ.addCylinder(*vein_outside_pt, \
                                        *(-lobule_vein_length*vein_normal_vector), \
                                        septal_vein_radius,curr_volume)
                try:
                    
                    model.occ.fuse([(3,1)],[(3,curr_volume)])         
                    
                    # An occasional error means that the centroid is outside the placental domain
                    fns.check_unique_vol(model)

                    try:
                        model.occ.fillet([1],[fns.find_line_with_centre(model,vein_inside_pt)],[septal_vein_funnel_radius])
                        model.occ.synchronize()
                        # INCORRECT, SEEMS TO ACTUALLY - As surface is curved, fillet radius is not exactly what asked for - recalculate this
                        line_no = fns.find_line_with_centre(model,vein_inside_pt)
                        pt_on_line = model.getValue(1,line_no,[0.0])
                        actual_radius = numpy.linalg.norm(copy.deepcopy(pt_on_line-vein_inside_pt))
                        actual_fillet_radius = actual_radius-septal_vein_radius

                        cyl_face_no = fns.find_surf_with_centre(model,vein_outside_pt)
                        bb = model.occ.getBoundingBox(2,cyl_face_no)
                        #bounding_box.update_bounds(bb)
                        #face_no = fns.get_face_in_bb(bounding_box)
                        #bounding_boxes.append(copy.deepcopy(bounding_box))
                        face.update_face(cyl_face_no,vein_outside_pt,vein_normal_vector,'septal_vein')
                        face.update_generating_cylinder_info(cylinder_radius=septal_vein_radius, \
                                                                cylinder_centre=vein_outside_pt, \
                                                                cylinder_length=lobule_vein_length, \
                                                                cylinder_fillet=actual_fillet_radius)
                        outlet_faces.append(copy.deepcopy(face))
                        no_outlets = no_outlets+1
                        no_veins_added = no_veins_added+1
                        no_lobule_veins = no_lobule_veins+1
                    except:
                        print(f"Error in filleting basal vein {vein_no}")
                        model.occ.synchronize()
                        # gmsh.fltk.run()
                        gmsh.finalize()
                        # sys.exit(-1)
                        raise Exception(f"Error in filleting basal vein {vein_no}")

                except Exception:

                    print(f"Error setting up basal vein in sub placentone")   
                    model.occ.remove([(3,curr_volume)], recursive=True)

            else:
                print(f"Basal plate vein {vein_no} not all in same lobule")
                print(f"Ignoring")
            
            if (no_veins_added == stop_at_veins_added):
                break
            
    # Print out the structure across placentones
    print(f"artery added array: {numpy.array2string(to_store_artery_added,separator = ',')}")
    print(f"basal veins added array: {numpy.array2string(to_store_stop_at_basal_veins_added,separator = ',')}")
    print(f"no veins added array: {numpy.array2string(to_store_no_vein_to_add,separator = ',')}")
    print(f"vein power array: {numpy.array2string(to_store_vein_power,separator = ',')}")
                
    return [no_inlets,no_cavities,bounding_boxes,inlet_faces,cavities,no_lobule_veins,no_outlets,outlet_faces]



##################################################################
############## MARGINAL SINUS VEINS (BORDER VEINS) ###############
##################################################################  
def create_marginal_veins(model,cotyledons,lobules,no_outlets,outlet_faces):
    
    face = clses.Face()
    marginal_sinus_vein_offset_shifted = marginal_sinus_vein_offset
    vein_no = 0
    no_marginal_sinus_veins_added = 0
    debug_counter = 0
    
    model.occ.synchronize()
    
    if (no_marginal_sinus_veins > 0):
        possible_veins = [i for i in range(0,no_marginal_sinus_veins)]
        # biased random choice of how many marginal veins to add
        # bias ∈ [0,10]: 0 → always 0, 10 → skewed to full range
        alpha = 1 + marginal_sinus_vein_bias
        beta  = 1 + (10 - marginal_sinus_vein_bias)
        r     = numpy.random.beta(alpha, beta)                 # r ∈ [0,1]
        # round to nearest integer in [0, no_marginal_sinus_veins]
        no_veins_to_add = int(round(r * no_marginal_sinus_veins))
        # ensure bounds
        no_veins_to_add = max(0, min(no_veins_to_add, no_marginal_sinus_veins))
        veins_to_add = numpy.random.choice( \
            possible_veins,size=no_veins_to_add,replace=False)
    else:
        no_veins_to_add = 0
        veins_to_add = []

    print(f"Number of peripheral veins to add: {no_veins_to_add}")
    print(f"Peripheral veins numbers: {veins_to_add}")

    while no_marginal_sinus_veins_added < no_veins_to_add and debug_counter < 1e5:
        
        vein_no = veins_to_add[no_marginal_sinus_veins_added]

        debug_counter = debug_counter + 1
        print(f"Creating marginal vein {vein_no}")
        
        curr_surf = model.occ.getMaxTag(2)+1
        curr_vol = model.occ.getMaxTag(3)+1
        
        vein_surface_pt = numpy.empty(3)
        vein_surface_pt[2] = marginal_vein_height
        
        cross_section_r = math.sqrt(initial_sphere_radius**2 - (marginal_vein_height-initial_sphere_radius)**2)
        
        theta_mult = (2.0*math.pi/no_marginal_sinus_veins)
        theta_rand = (1.0 - 2.0*marginal_sinus_vein_offset_buffer) * numpy.random.random() - 0.5 # shifts bound from [0, 1] of rand to e.g. [-0.4, 0.4]
        theta = marginal_sinus_vein_offset_shifted+(vein_no+theta_rand)*theta_mult
        
        vein_surface_pt[0] = cross_section_r*math.cos(theta)
        vein_surface_pt[1] = cross_section_r*math.sin(theta)
        
        vein_surface_outward_norm = numpy.empty(3)
        vein_surface_outward_norm[0:2] = vein_surface_pt[0:2]
        vein_surface_outward_norm[2] = vein_surface_pt[2] - initial_sphere_radius
        vein_surface_outward_norm = fns.normalise_vector(3,vein_surface_outward_norm)
        vein_surface_outward_norm_2d_proj = numpy.empty(3)
        vein_surface_outward_norm_2d_proj[0:2] = fns.normalise_vector(2,vein_surface_outward_norm[0:2])
        vein_surface_outward_norm_2d_proj[2] = 0.0
        
        # Counterclockwise (looking down) tangent
        vein_tangent = numpy.empty(3)
        vein_tangent[0:3] = [-math.sin(theta),math.cos(theta),0.0]
        
        # 'upward' pointing vector
        upward_cyl_face_tangent = numpy.cross(vein_tangent,-vein_surface_outward_norm)
        
        dzdx = numpy.array([math.sqrt(initial_sphere_radius**2 - (vein_surface_pt[0]**2 + vein_surface_pt[1]**2)),0.0,vein_surface_pt[0]])
        dzdx = dzdx/math.sqrt(initial_sphere_radius**2 - vein_surface_pt[1]**2)
        
        marginal_border_point = vein_surface_pt + marginal_sinus_vein_radius*dzdx
        marginal_border_vector = marginal_border_point-initial_sphere_centre
        marginal_border_radius = numpy.linalg.norm(marginal_border_vector)
        shift_angle = math.acos(numpy.dot(marginal_border_vector,vein_surface_outward_norm)/marginal_border_radius)
        shift_radius = marginal_border_radius-initial_sphere_radius
        shift_dist = shift_radius*math.cos(shift_angle)
        
        # Need this as a small offset to push cylinder into semi-circle more, otherwise tol too small and boolean intersection errors
        shift_dist_coeff = 1.1
        
        cylinder_inside_pt = vein_surface_pt-shift_dist*vein_surface_outward_norm#- 2.0*vein_surface_outward_norm
        cylinder_outside_pt = cylinder_inside_pt+(shift_dist+septal_vessel_length)*vein_surface_outward_norm
        
        # Crude estimate at edges of the fillet circle
        cotyledon_nos = numpy.empty(3,dtype=int)
        lobule_nos = numpy.empty(3,dtype=int)
        left_est = cylinder_inside_pt-1.2*(marginal_sinus_vein_radius+marginal_fillet_radius)*vein_tangent
        right_est = cylinder_inside_pt+1.2*(marginal_sinus_vein_radius+marginal_fillet_radius)*vein_tangent
        lobule_nos[0] = fns.find_lobule_no(left_est,lobules)
        lobule_nos[1] = fns.find_lobule_no(cylinder_inside_pt,lobules)
        lobule_nos[2] = fns.find_lobule_no(right_est,lobules)
        cotyledon_nos[0] = fns.find_cotyledon_no(left_est,cotyledons)
        cotyledon_nos[1] = fns.find_cotyledon_no(cylinder_inside_pt,cotyledons)
        cotyledon_nos[2] = fns.find_cotyledon_no(right_est,cotyledons)
        
        near_artery_or_vein = False
        artery_cavity_clearance = marginal_sinus_vein_radius+septal_artery_radius+marginal_fillet_radius+cavity_minor_axis
        vein_clearance = marginal_sinus_vein_radius+marginal_fillet_radius+septal_vein_radius+septal_vein_funnel_radius
        for face_obj in outlet_faces:
            if (face_obj.vessel_type == 'septal_vein'):
                face_inside_pt = copy.deepcopy(face_obj.cylinder_centre-face_obj.cylinder_length*face_obj.outward_unit_normal)
                if (numpy.linalg.norm(cylinder_inside_pt[0:2]-face_inside_pt[0:2]) < vein_clearance):
                    near_artery_or_vein = True
        # Check cylinders won't overlap
        if (numpy.linalg.norm(cylinder_inside_pt[0:2]-lobules[lobule_nos[0]].centroid[0:2]) < \
                artery_cavity_clearance):
            near_artery_or_vein = True
        
        # All points in same lobule so can create cyl
        if (lobule_nos[0] != -1 and cotyledon_nos[0] != -1 and not(near_artery_or_vein) \
                and numpy.all(lobule_nos == lobule_nos[0]) and numpy.all(cotyledon_nos == cotyledon_nos[0])):
                
            model.occ.addCylinder(*cylinder_outside_pt,*(-((shift_dist_coeff*shift_dist)+septal_vessel_length)*vein_surface_outward_norm), \
                marginal_sinus_vein_radius,curr_vol)

            model.occ.synchronize()
            
            model.occ.fuse([(3,1)],[(3,curr_vol)])

            try:
                model.occ.fillet([1],[fns.find_line_with_centre(model,cylinder_inside_pt)],[marginal_fillet_radius])
                model.occ.synchronize()
                # INCORRECT, SEEMS TO ACTUALLY - As surface is curved, fillet radius is not exactly what asked for - recalculate this
                line_no = fns.find_line_with_centre(model,cylinder_inside_pt)
                pt_on_line = model.getValue(1,line_no,[0.0])
                actual_radius = numpy.linalg.norm(copy.deepcopy(pt_on_line-cylinder_inside_pt))
                actual_fillet_radius = actual_radius-marginal_sinus_vein_radius

                cyl_face_no = fns.find_surf_with_centre(model,cylinder_outside_pt)
                bb = model.occ.getBoundingBox(2,cyl_face_no)
                #bounding_box.update_bounds(bb)
                #face_no = fns.get_face_in_bb(bounding_box)
                #bounding_boxes.append(copy.deepcopy(bounding_box))
                face.update_face(cyl_face_no,cylinder_outside_pt,vein_surface_outward_norm)
                face.update_generating_cylinder_info(cylinder_radius=marginal_sinus_vein_radius, \
                                                        cylinder_centre=cylinder_outside_pt, \
                                                        cylinder_length=shift_dist+septal_vessel_length, \
                                                        cylinder_fillet=actual_fillet_radius)
                outlet_faces.append(copy.deepcopy(face))
                no_marginal_sinus_veins_added = no_marginal_sinus_veins_added + 1
                no_outlets = no_outlets+1
                vein_no = vein_no+1
                marginal_sinus_vein_offset_shifted = marginal_sinus_vein_offset
                
                # model.occ.synchronize()
                # gmsh.fltk.run()
                
                    
                print(f"Added peripheral vein {vein_no}")
                print(f"face no: {outlet_faces[-1].face_no}")
                print(f"pt out placenta: [{outlet_faces[-1].centre[0]},{outlet_faces[-1].centre[1]},{outlet_faces[-1].centre[2]}]")
                print(f"norm out cell: [{outlet_faces[-1].outward_unit_normal[0]},{outlet_faces[-1].outward_unit_normal[1]},{outlet_faces[-1].outward_unit_normal[2]}]")
                
            except:
                print(f"Error in filleting marginal vein {vein_no}")
                model.occ.synchronize()
                # gmsh.fltk.run()
                gmsh.finalize()
                # sys.exit(-1)
                raise Exception(f"Error in filleting marginal vein {vein_no}")
                model.occ.remove([(3,curr_vol)],recursive=True)

                marginal_sinus_vein_offset_shifted = \
                    marginal_sinus_vein_offset_shifted+math.pi/20.0
                
                debug_counter = debug_counter+1
                if (debug_counter==2):
                    model.occ.synchronize()
                    gmsh.fltk.run()
                    gmsh.finalize()
                    sys.exit(-1)
        else:
            print(f"Marginal vein {vein_no} pts not all in same lobule")
            print(f"Shifting angle of placement")
            
            print(f"cotyledon_nos: {cotyledon_nos}")
            print(f"lobule_nos: {lobule_nos}")
            print(f"near_artery_or_vein: {near_artery_or_vein}")
            print(f"marginal_sinus_vein_offset_shifted: {marginal_sinus_vein_offset_shifted}")
            
            marginal_sinus_vein_offset_shifted = \
                marginal_sinus_vein_offset_shifted+math.pi/100.0
            
            debug_counter = debug_counter+1
    
    return [model,no_marginal_sinus_veins_added,no_outlets,outlet_faces]

def create_septal_veins(model,no_cotyledon,cotyledons, \
        c_edge_set,no_outlets,outlet_faces,lobule_node_set_per_cotyledon):
        
    face = clses.Face()
    
    no_septal_wall_veins = 0
    no_septal_wall_veins_per_cotyledon = numpy.zeros(no_cotyledon,dtype=int)
    
    vein_length_buffer = placentone_wall_thickness/100.0 # Ensure two opposite placed veins don't overlap
    length = placentone_wall_thickness/2.0 - vein_length_buffer
    
    if (not(fixed_septal_wall_veins)):

        for cotyledon_no in range(0,no_cotyledon):

            print("Adding septal wall vein to placentone ",cotyledon_no)
            
            lobule_node_set = lobule_node_set_per_cotyledon[cotyledon_no]
            curr_success_added_veins = []
            
            # add_vein_to_c = numpy.random.randint(0,2)
            
            if septal_vein_bias <= 0:
                no_septal_vein_to_add = 0
            elif septal_vein_bias >= 10:
                no_septal_vein_to_add = 6
            else:
                # continuous random choice between 0–6 skewed by septal_vein_bias
                # use a Beta distribution: higher septal_vein_bias → more weight at higher values
                alpha = 1 + septal_vein_bias
                beta  = 1 + (10 - septal_vein_bias)
                r     = numpy.random.beta(alpha, beta)   # r ∈ [0,1]
                no_septal_vein_to_add = int(round(r * 6))  # map to 0–6
            
            for i in range(0,no_septal_vein_to_add):
                
                retry_cntr = 0
                pt_success = False
                while (retry_cntr < 10 and not(pt_success)):
                    
                    curr_vol = model.occ.getMaxTag(3)+1

                    placentone_to_do = cotyledons[cotyledon_no]

                    min_edge_length = 4.0*septal_vein_radius
                    min_edge_length = 1.5*min_edge_length

                    # Count how many edges lie in placenta
                    no_edges_in_placenta = 0
                    edges_in_placenta = []
                    for loc_edge_no in range(0,c_edge_set.no_cell_edges[cotyledon_no]):
                        
                        glob_edge_no = c_edge_set.cell_edges[cotyledon_no,loc_edge_no]
                        
                        [vertex1,vertex2] = \
                            c_edge_set.get_vertices_from_cell_edge(cotyledon_no,loc_edge_no)
                        
                        vertex1_dist = circ_eval(*vertex1[0:2])
                        vertex2_dist = circ_eval(*vertex2[0:2])
                        
                        # Check if both points aren't in due to limitations of readjust_v function
                        if (not(vertex1_dist >= placenta_radius**2 and vertex2_dist >= placenta_radius**2)):
                        
                            [vertex1,vertex2] = fns.readjust_vertices_for_wall_veins( \
                                cotyledon_no,vertex1,vertex2,placenta_radius, \
                                glob_edge_no,c_edge_set)
                            
                            edge_dir = numpy.copy(vertex2-vertex1)
                            edge_length = numpy.linalg.norm(edge_dir)

                            if (edge_length > min_edge_length):
                                
                                no_edges_in_placenta = no_edges_in_placenta+1
                                edges_in_placenta.append(loc_edge_no)
                            
                    if (no_edges_in_placenta > 0):
                        
                        edges_in_placenta = numpy.array(edges_in_placenta)
                        
                        # end_pt is in wall
                        face_cyl_pt = numpy.array([0.0,0.0,0.0])
                        face_cyl_end_pt = numpy.array([0.0,0.0,0.0])
                        
                        # Randomly pick which wall to place the vein on, how far along and height
                        loc_edge_no = numpy.random.randint(0,no_edges_in_placenta)
                        loc_edge_no = edges_in_placenta[loc_edge_no]
                        glob_edge_no = c_edge_set.cell_edges[cotyledon_no,loc_edge_no]
                        
                        ratio_retry = 0
                        ratio_overlap = True
                        while (ratio_overlap == True and ratio_retry < 5):
                        
                            # Either fix height ratio or choose randomly
                            if (adjust_stored_septal_vein_height):
                                height_ratio = adjust_septal_height_ratio
                            else:
                                height_ratio = numpy.random.random()
                                
                            # Hack to stop point being too close to edge - not needed anymore as using node set
                            edge_ratio = numpy.random.random()
                            while (edge_ratio < 0.15 or edge_ratio > 0.85):
                                edge_ratio = numpy.random.random()

                            [vertex1,vertex2] = \
                                c_edge_set.get_vertices_from_cell_edge(cotyledon_no,loc_edge_no)
                            
                            edge_dir = numpy.copy(vertex2-vertex1)
                            edge_length = numpy.linalg.norm(numpy.copy(vertex2-vertex1))
                            
                            vertex1_dist = circ_eval(*vertex1[0:2])
                            vertex2_dist = circ_eval(*vertex2[0:2])
                            
                            [shift_vertex1,shift_vertex2] = fns.readjust_vertices_for_wall_veins( \
                                cotyledon_no,vertex1,vertex2,placenta_radius, \
                                glob_edge_no,c_edge_set)

                            shift_edge_length = numpy.linalg.norm(numpy.copy(shift_vertex2-shift_vertex1))
                            
                            # Edge ratio acts upon the readjusted vertices, hence this converts to original edge
                            if (vertex1_dist >= placenta_radius**2):
                                shift_edge_ratio = 1.0 - (shift_edge_length/edge_length)*(1.0-edge_ratio)
                            elif (vertex2_dist >= placenta_radius**2):
                                shift_edge_ratio = edge_ratio*(shift_edge_length/edge_length)
                            else:
                                shift_edge_ratio = edge_ratio
                            
                            # Create vein centroid
                            face_cyl_end_pt[0:2] = c_edge_set.calc_pt_along_edge( \
                                glob_edge_no,shift_edge_ratio)
                            
                            rel_hgt = c_edge_set.calc_rel_height_along_edge( \
                                glob_edge_no,shift_edge_ratio)
                            
                            # Oriented out of Voronoi cell
                            norm_dir = numpy.array([-edge_dir[1],edge_dir[0]])
                            # Flip normal if centroid is in direction of norm_dir
                            if (norm_dir[0]*(placentone_to_do.centroid[0] - face_cyl_end_pt[0]) \
                                    + norm_dir[1]*(placentone_to_do.centroid[1] - face_cyl_end_pt[1]) > 0):
                                norm_dir = -norm_dir
                            norm_dir = fns.normalise_vector(2,norm_dir)
                            
                            # Shift point slightly outward to avoid overlap on adjacent vein
                            face_cyl_end_pt[0:2] = face_cyl_end_pt[0:2] - vein_length_buffer*norm_dir[0:2]
                            
                            face_cyl_pt[0:2] = copy.deepcopy(face_cyl_end_pt[0:2]) - length*norm_dir[0:2]

                            Z_pt = fns.calc_septal_vein_height( \
                                rel_hgt,face_cyl_pt[0:2], \
                                face_cyl_end_pt[0:2],height_ratio)
                            
                            face_cyl_pt[2] = Z_pt
                            face_cyl_end_pt[2] = Z_pt
                            
                            # First check lobule points
                            ratio_overlap = fns.check_septal_vein_overlap_lobule_walls( \
                                face_cyl_end_pt, lobule_node_set, \
                                    septal_vein_radius, fillet_radius, lobule_wall_thickness)
                            # Then current successful points
                            for success_pt in curr_success_added_veins:
                                if (abs(Z_pt - success_pt[2]) > \
                                        2.0*(septal_vein_radius + fillet_radius)):
                                    continue
                                if (numpy.linalg.norm(face_cyl_end_pt[0:2]-success_pt[0:2]) < \
                                        2.0*(septal_vein_radius + fillet_radius)):
                                    ratio_overlap = True
                                    break
                            # Increment
                            ratio_retry = ratio_retry + 1
                            
                        # Reset in case of failed overlaps
                        if (ratio_overlap == True):
                            pt_success = False
                            retry_cntr = retry_cntr + 1
                            continue

                        if (face_cyl_pt[2]+septal_vein_radius < placenta_height):
                            
                            cyl_radius = septal_artery_radius
                            
                            model.occ.addCylinder(*face_cyl_pt, \
                                                *(face_cyl_end_pt-face_cyl_pt), \
                                                cyl_radius,curr_vol)
                            
                            cyl_face = model.occ.getMaxTag(2)

                            model.occ.fuse([(3,1)],[(3,curr_vol)])       

                            try:
                                # An occasional error means that the centroid is outside the placental domain
                                fns.check_unique_vol(model)
                                max_vol_tag = model.occ.getMaxTag(3)
                                try:
                                    model.occ.fillet([max_vol_tag], [fns.find_line_with_centre(model,face_cyl_pt)], [fillet_radius])
                                except:
                                    print(f"Error in filleting septal wall vein")
                                    model.occ.synchronize()
                                    # gmsh.fltk.run()
                                    gmsh.finalize()
                                    # sys.exit(-1)
                                    raise Exception(f"Error in filleting septal wall vein")

                                # Incrememnt outlet
                                no_outlets = no_outlets+1
                                no_septal_wall_veins = no_septal_wall_veins + 1

                                face.update_face(cyl_face,face_cyl_end_pt,numpy.array([norm_dir[0],norm_dir[1],0.0]))
                                face.update_generating_cylinder_info(cylinder_radius=cyl_radius, \
                                                                        cylinder_centre=face_cyl_end_pt, \
                                                                        cylinder_length=length, \
                                                                        cylinder_fillet=fillet_radius)
                                
                                outlet_faces.append(copy.deepcopy(face))
                                
                                pt_success = True
                                curr_success_added_veins.append(copy.deepcopy(face_cyl_end_pt[0:3]))
                                
                                # model.occ.synchronize()
                                # gmsh.fltk.run()
                                
                                print(f"Added septal wall")
                                print(f"face no: {outlet_faces[-1].face_no}")
                                print(f"pt in wall: [{outlet_faces[-1].centre[0]},{outlet_faces[-1].centre[1]},{outlet_faces[-1].centre[2]}]")
                                print(f"norm out cell: [{outlet_faces[-1].outward_unit_normal[0]},{outlet_faces[-1].outward_unit_normal[1]},{outlet_faces[-1].outward_unit_normal[2]}]")

                            except Exception:

                                print(f"Error setting up septal wall vessel")
                                model.occ.remove([(3,curr_vol)], recursive=True)


        print(f"No. septall wall veins: {no_septal_wall_veins}")

        print(f"face_nos")
        for i in range(no_outlets-no_septal_wall_veins,no_outlets):
            print(f"{outlet_faces[i].face_no},")

        print(f"pts,")
        for i in range(no_outlets-no_septal_wall_veins,no_outlets):
            face_cyl_pt = numpy.copy(outlet_faces[i].centre)
            print(f"[{face_cyl_pt[0]},{face_cyl_pt[1]},{face_cyl_pt[2]}],")

        print(f"norms,")
        for i in range(no_outlets-no_septal_wall_veins,no_outlets):
            face_norm = numpy.copy(outlet_faces[i].outward_unit_normal)
            print(f"[{face_norm[0]},{face_norm[1]},{face_norm[2]}],")    
            
    else:

        for v_no in range(0,no_stored_septal_wall_veins):
            
            print(f"Retrieving stored septal wall vein {v_no}")
            
            curr_vol = model.occ.getMaxTag(3)+1
            
            cyl_face = septal_vein_face_nos[v_no]
            face_cyl_end_pt = septal_veins_pts[v_no]
            norm_dir = septal_veins_norms[v_no]
            length = vein_length
            
            face_pt_on_placenta = copy.deepcopy(face_cyl_end_pt)
            face_pt_on_placenta[0:2] = face_cyl_end_pt[0:2] - length*norm_dir[0:2]
            
            # Needs to be fixed
            if (adjust_stored_septal_vein_height):
                
                print("ERROR: calc_septal_vein_height args need to be fixed")
                sys.exit(-1)
                
                face_pt_on_placenta[2] = fns.calc_septal_vein_height( \
                    face_pt_on_placenta[0:2],face_cyl_end_pt[0:2],adjust_septal_height_ratio)
                face_cyl_end_pt[2] = face_pt_on_placenta[2]
            
            cyl_radius = septal_vein_radius
            
            face.update_face(cyl_face,face_cyl_end_pt,numpy.array([norm_dir[0],norm_dir[1],0.0]))
            face.update_generating_cylinder_info(cylinder_radius=cyl_radius, \
                                                    cylinder_centre=face_cyl_end_pt, \
                                                    cylinder_length=length, \
                                                    cylinder_fillet=septal_vein_funnel_radius)

            outlet_faces.append(copy.deepcopy(face))
            
            try:
                
                model.occ.addCylinder(*face_cyl_end_pt, \
                                        *(-(1.0+cyl_offset_length)*length*norm_dir), \
                                        cyl_radius,curr_vol)

                model.occ.fuse([(3,1)],[(3,curr_vol)])

                max_vol_tag = model.occ.getMaxTag(3)
                model.occ.fillet([max_vol_tag], [fns.find_line_with_centre(model,face_pt_on_placenta)], [septal_vein_funnel_radius])

                no_outlets = no_outlets + 1
                no_septal_wall_veins = no_septal_wall_veins + 1
            
            except:
                print(f"Hit failsafe when loading septal wall veins")
                model.occ.synchronize()
                # gmsh.fltk.run()
                gmsh.finalize()
                # sys.exit(-1)
                raise Exception(f"Hit failsafe when loading septal wall veins")

    
    
    return [no_septal_wall_veins,no_outlets,outlet_faces]