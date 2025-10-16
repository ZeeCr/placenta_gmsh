import sys
import math
import importlib

import numpy
import copy

import gmsh
import foronoi

import placenta_fns as fns
import placenta_plots as plots
import placenta_const

importlib.reload(fns)
importlib.reload(plots)
importlib.reload(placenta_const)

from placenta_const import *



def refine_inside_around_cavity(model,no_cavities,cavities,field_list):
    
    for cavity_no in range(0,no_cavities):
        
        field_no = 401+2*cavity_no
        
        centre = numpy.copy(cavities[cavity_no].centre)
        normal = numpy.copy(cavities[cavity_no].orientation_normal)
        peak_pt = centre+cavities[cavity_no].major_axis*normal
        minor_axis = cavities[cavity_no].minor_axis
        major_axis = cavities[cavity_no].major_axis
        
        nx = normal[0]
        ny = normal[1]
        nz = normal[2]
        x0 = centre[0]
        y0 = centre[1]
        z0 = centre[2]
        
        nx = fns.convert_float_to_gmsh_field_str(nx)
        ny = fns.convert_float_to_gmsh_field_str(ny)
        nz = fns.convert_float_to_gmsh_field_str(nz)
        x0 = fns.convert_float_to_gmsh_field_str(x0)
        y0 = fns.convert_float_to_gmsh_field_str(y0)
        z0 = fns.convert_float_to_gmsh_field_str(z0)

        # if (nx<0.0):
        #     nx = f"({nx})"
        # else:
        #     nx = f"{nx}"
        # if (ny<0.0):
        #     ny = f"({ny})"
        # else:
        #     ny = f"{ny}"
        # if (nz<0.0):
        #     nz = f"({nz})"
        # else:
        #     nz = f"{nz}"
        # if (x0<0.0):
        #     x0 = f"({x0})"
        # else:
        #     x0 = f"{x0}"
        # if (y0<0.0):
        #     y0 = f"({y0})"
        # else:
        #     y0 = f"{y0}"
        # if (z0<0.0):
        #     z0 = f"({z0})"
        # else:
        #     z0 = f"{z0}"
        
        denom = f"((1.0+(({nx}^2)/({nz}^2)))^0.5)"
        x_ref = f"((x-{x0})/{denom})-{nx}*(z-{z0})/({nz}*{denom})"
        y_ref = f"-({ny}*{nx}/({nz}*{denom}))*(x-{x0})+(y-{y0})*{nz}*{denom}-({ny}*(z-{z0})/{denom})"
        z_ref = f"{nx}*(x-{x0})+{ny}*(y-{y0})+{nz}*(z-{z0})"
        x_ref = fns.replace_multiple_signs_in_str(x_ref)
        y_ref = fns.replace_multiple_signs_in_str(y_ref)
        z_ref = fns.replace_multiple_signs_in_str(z_ref)
        
        ellipsoid_str = f"(({x_ref})^2+({y_ref})^2)/(({minor_axis})^2)+(({z_ref})^2)/(({major_axis})^2)"
        ellipsoid_str = fns.replace_multiple_signs_in_str(ellipsoid_str)
        
        model.mesh.field.add("MathEval",field_no)
        model.mesh.field.setString(field_no,"F",ellipsoid_str)
        
        model.mesh.field.add("Threshold", field_no+1)
        model.mesh.field.setNumber(field_no+1,"InField",field_no)
        model.mesh.field.setNumber(field_no+1,"SizeMin",CavityMeshSize)
        model.mesh.field.setNumber(field_no+1,"SizeMax",OuterCavityMeshSize)
        model.mesh.field.setNumber(field_no+1,"DistMin",1.0)
        model.mesh.field.setNumber(field_no+1,"DistMax",(1.0+cavity_mesh_thickness)**2)
        model.mesh.field.setNumber(field_no+1,"Sigmoid",1)
        model.mesh.field.setNumber(field_no+1,"StopAtDistMax",1)
    
        field_list = field_list + [field_no+1]

    
    return [model,field_list]



def refine_apex_cavity(model,no_cavities,cavities,field_list):
    
    
    for cavity_no in range(0,no_cavities):
        
        field_no = 501+cavity_no
        
        centre = numpy.copy(cavities[cavity_no].centre)
        normal = numpy.copy(cavities[cavity_no].orientation_normal)
        peak_pt = centre+cavities[cavity_no].major_axis*normal
        
        field_radius = cavities[cavity_no].major_axis/4.0
        field_thickness = field_radius/2.0
        
        model.mesh.field.add("Ball",field_no)
        model.mesh.field.setNumber(field_no,"VIn",CavityApexSize)
        model.mesh.field.setNumber(field_no,"VOut",DomSize)
        model.mesh.field.setNumber(field_no,"Radius",field_radius)
        model.mesh.field.setNumber(field_no,"Thickness",field_thickness)
        model.mesh.field.setNumber(field_no,"XCenter",peak_pt[0])
        model.mesh.field.setNumber(field_no,"YCenter",peak_pt[1])
        model.mesh.field.setNumber(field_no,"ZCenter",peak_pt[2])
    
        field_list = field_list + [field_no]

    
    return [model,field_list]

def refine_cot_walls(model,edge_set,field_list, \
        inner_mesh_size = InnerWallDomSize, outer_mesh_size = DomSize, \
        buffer_multiplier = 1.0, field_start = 601):
    
    for edge_no in range(0,edge_set.no_edges):

        xy_buffer = 1.0*placentone_wall_thickness*buffer_multiplier
        z_buffer = 0.2*buffer_multiplier
        shifted_buffer = 0.1*buffer_multiplier

        v1_no = edge_set.edge[edge_no,0]
        v2_no = edge_set.edge[edge_no,1]
        
        v1 = edge_set.node_set.node[v1_no,:]
        v2 = edge_set.node_set.node[v2_no,:]
        
        # Wall outside domain so don't bother
        if (circ_eval(*v1) > placenta_radius**2 and circ_eval(*v2) > placenta_radius**2):
            continue

        v1_z = edge_set.node_set.abs_wall_height[v1_no]
        v2_z = edge_set.node_set.abs_wall_height[v2_no]

        # print(f"v1_height: {fns.calc_sphere_height_at_xy(v1)}")
        # print(f"v2_height: {fns.calc_sphere_height_at_xy(v2)}")
        # print(f"v1_z: {v1_z}, v2_z: {v2_z}")
        
        min_z = min(fns.calc_sphere_height_at_xy(v1), fns.calc_sphere_height_at_xy(v2)) * (1.0 - z_buffer)
        max_z = max(v1_z, v2_z) * (1.0 + z_buffer)

        field_no = field_start + 2*edge_no
        
        line_dir = v2 - v1
        line_norm = numpy.array([-line_dir[1], line_dir[0]])  # Perp
        
        line_length = numpy.linalg.norm(line_dir)

        line_unit_dir = line_dir / line_length if line_length > 0 else line_dir
        
        line_unit_norm = line_norm / line_length if line_length > 0 else line_dir
        
        # Set up 3d z plane
        shifted_v1_full = numpy.array([v1[0], v1[1], v1_z * (1.0 + z_buffer)])
        shifted_v2_full = numpy.array([v2[0], v2[1], v2_z * (1.0 + z_buffer)])
        plane_tang1 = shifted_v2_full - shifted_v1_full
        plane_tang1 = plane_tang1 / numpy.linalg.norm(plane_tang1) if numpy.linalg.norm(plane_tang1) > 0 else plane_tang1
        plane_norm = numpy.cross(plane_tang1, numpy.array([line_unit_norm[0], line_unit_norm[1], 0.0]))
        if plane_norm[2] > 0.0:
            plane_norm = -plane_norm

        # -------------V2-----------
        #              ^ line_dir
        #              |
        #              |------------------> line_unit_norm
        #              |
        #              |
        # -------------V1-----------


        shifted_neg_norm_v1 = v1 - xy_buffer * line_unit_norm
        shifted_pos_norm_v1 = v1 + xy_buffer * line_unit_norm
        
        shifted_line_v1 = v1 - shifted_buffer * line_dir
        shifted_line_v2 = v2 + shifted_buffer * line_dir
        shifted_line_length = numpy.linalg.norm(shifted_line_v2 - shifted_line_v1)
        
        ludx = fns.convert_float_to_gmsh_field_str(line_unit_dir[0])
        ludy = fns.convert_float_to_gmsh_field_str(line_unit_dir[1])
        lunx = fns.convert_float_to_gmsh_field_str(line_unit_norm[0])
        luny = fns.convert_float_to_gmsh_field_str(line_unit_norm[1])
        pnx = fns.convert_float_to_gmsh_field_str(plane_norm[0])
        pny = fns.convert_float_to_gmsh_field_str(plane_norm[1])
        pnz = fns.convert_float_to_gmsh_field_str(plane_norm[2])

        min_z = fns.convert_float_to_gmsh_field_str(min_z)
        max_z = fns.convert_float_to_gmsh_field_str(max_z)
        snn1x = fns.convert_float_to_gmsh_field_str(shifted_neg_norm_v1[0])
        snn1y = fns.convert_float_to_gmsh_field_str(shifted_neg_norm_v1[1])
        spn1x = fns.convert_float_to_gmsh_field_str(shifted_pos_norm_v1[0])
        spn1y = fns.convert_float_to_gmsh_field_str(shifted_pos_norm_v1[1])
        s1x = fns.convert_float_to_gmsh_field_str(shifted_v1_full[0])
        s1y = fns.convert_float_to_gmsh_field_str(shifted_v1_full[1])
        s1z = fns.convert_float_to_gmsh_field_str(shifted_v1_full[2])

        slv1x = fns.convert_float_to_gmsh_field_str(shifted_line_v1[0])
        slv1y = fns.convert_float_to_gmsh_field_str(shifted_line_v1[1])
        slv2x = fns.convert_float_to_gmsh_field_str(shifted_line_v2[0])
        slv2y = fns.convert_float_to_gmsh_field_str(shifted_line_v2[1])

        # Dist = | (X - v1).n | = | (x-v1x)*nx + (y-v1y)*ny |
        xy_neg_dist = f"(x-{snn1x})*({lunx}) + (y-{snn1y})*({luny})"
        xy_pos_dist = f"(x-{spn1x})*(-{lunx}) + (y-{spn1y})*(-{luny})"
        # z_neg_dist = f"z-{min_z}"
        # z_pos_dist = f"{max_z}-z"
        z_pos_dist = f"(x - {s1x})*({pnx}) + (y - {s1y})*({pny}) + (z - {s1z})*({pnz})"

        # Not actually plane, line but measure dist from to determine if 'behind' v1
        # Want it s.t. points between v1, v2 are 0 <= dist <= shifted_line_length, so +ve distance towards the other point so v2 is -ve dir vec
        v1_plane_dist = f"(x-{slv1x})*({ludx}) + (y-{slv1y})*({ludy})"
        v2_plane_dist = f"(x-{slv2x})*(-{ludx}) + (y-{slv2y})*(-{ludy})"

        line_centre = ( v1 + v2 ) / 2.0

        # print( (line_centre[0]-shifted_neg_norm_v1[0])*line_unit_norm[0] + (line_centre[1]-shifted_neg_norm_v1[1])*line_unit_norm[1] )
        # print( (line_centre[0]-shifted_pos_norm_v1[0])*(-line_unit_norm[0]) + (line_centre[1]-shifted_pos_norm_v1[1])*(-line_unit_norm[1]) )
        
        # print( (fns.calc_sphere_height_at_xy(v1)+fns.calc_sphere_height_at_xy(v2))/2.0 - min_z )
        # print( max_z - (fns.calc_sphere_height_at_xy(v1)+fns.calc_sphere_height_at_xy(v2))/2.0 )

        # print( (line_centre[0]-shifted_line_v1[0])*line_unit_dir[0] + (line_centre[1]-shifted_line_v1[1])*line_unit_dir[1] )
        # print( (line_centre[0]-shifted_line_v2[0])*(-line_unit_dir[0]) + (line_centre[1]-shifted_line_v2[1])*(-line_unit_dir[1]) )
        # print("~~~")
        # print(fns.calc_sphere_height_at_xy(v1))
        # print(fns.calc_sphere_height_at_xy(v2))
        # print("-------")
        
        # In theory, these planes define box s.t. point in box if no val is < 0
        # dist_str = f"min( {xy_neg_dist}, {xy_pos_dist}, {z_neg_dist}, {z_pos_dist}, {v1_plane_dist}, {v2_plane_dist} )"
        dist_str = f"min( {xy_neg_dist}, {xy_pos_dist}, {z_pos_dist}, {v1_plane_dist}, {v2_plane_dist} )"
        dist_str = fns.replace_multiple_signs_in_str(dist_str)
        # dist_str = f"min( min( min( min( min( {xy_neg_dist}, {xy_pos_dist} ), {z_neg_dist}), {z_pos_dist}), {v1_plane_dist}), {v2_plane_dist} )"
        # dist_str = f"min( {xy_neg_dist}, 4.0 )"
        # print(xy_neg_dist)

        model.mesh.field.add("MathEval",field_no)
        model.mesh.field.setString(field_no,"F",dist_str)
        
        # DistMax shouldn't matter as long as what I've done works as intended
        model.mesh.field.add("Threshold", field_no+1)
        model.mesh.field.setNumber(field_no+1,"InField",field_no)
        model.mesh.field.setNumber(field_no+1,"SizeMin",outer_mesh_size)
        model.mesh.field.setNumber(field_no+1,"SizeMax",inner_mesh_size)
        model.mesh.field.setNumber(field_no+1,"DistMin",0.0)
        model.mesh.field.setNumber(field_no+1,"DistMax",1.0e-4)
        model.mesh.field.setNumber(field_no+1,"Sigmoid",0)
        model.mesh.field.setNumber(field_no+1,"StopAtDistMax",0)
    
        field_list = field_list + [field_no+1]

    return [model,field_list]












'''   
for cavity_no in range(0, no_cavities):
    field_no = 401+cavity_no
    centre = numpy.copy(cavities[cavity_no].centre)
    normal = numpy.copy(cavities[cavity_no].orientation_normal)
    start_pt = centre - 0.25*septal_vessel_length*normal
    
    diff = (centre-start_pt)+1.2*cavities[cavity_no].major_axis*normal
    x_diff = diff[0]
    y_diff = diff[1]
    z_diff = diff[2]
    
    model.mesh.field.add("Cylinder",field_no)
    model.mesh.field.setNumber(field_no,"VIn",CavityMeshSize)
    model.mesh.field.setNumber(field_no,"VOut",DomSize)
    model.mesh.field.setNumber(field_no,"Radius",cavities[cavity_no].minor_axis*2.0)
    model.mesh.field.setNumber(field_no,"XCenter",start_pt[0])
    model.mesh.field.setNumber(field_no,"YCenter",start_pt[1])
    model.mesh.field.setNumber(field_no,"ZCenter",start_pt[2])
    model.mesh.field.setNumber(field_no,"XAxis",x_diff)
    model.mesh.field.setNumber(field_no,"YAxis",y_diff)
    model.mesh.field.setNumber(field_no,"ZAxis",z_diff)
    
    field_list = field_list + [field_no]
'''
'''Old cyl
for cavity_no in range(0, no_cavities):
    field_no = 401+cavity_no
    
    centre = numpy.copy(cavities[cavity_no].centre)
    normal = numpy.copy(cavities[cavity_no].orientation_normal)
    start_pt = centre - 0.25*septal_vessel_length*normal
    
    diff = (centre-start_pt)+1.2*cavities[cavity_no].major_axis*normal
    x_diff = diff[0]
    y_diff = diff[1]
    z_diff = diff[2]
    
    model.mesh.field.add("Cylinder",field_no)
    model.mesh.field.setNumber(field_no,"VIn",CavityMeshSize)
    model.mesh.field.setNumber(field_no,"VOut",DomSize)
    model.mesh.field.setNumber(field_no,"Radius",cavities[cavity_no].minor_axis*2.0)
    model.mesh.field.setNumber(field_no,"XCenter",start_pt[0])
    model.mesh.field.setNumber(field_no,"YCenter",start_pt[1])
    model.mesh.field.setNumber(field_no,"ZCenter",start_pt[2])
    model.mesh.field.setNumber(field_no,"XAxis",x_diff)
    model.mesh.field.setNumber(field_no,"YAxis",y_diff)
    model.mesh.field.setNumber(field_no,"ZAxis",z_diff)
    
    field_list = field_list + [field_no]
'''