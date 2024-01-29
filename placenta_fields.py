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
        
        if (nx<0.0):
            nx = f"({nx})"
        else:
            nx = f"{nx}"
        if (ny<0.0):
            ny = f"({ny})"
        else:
            ny = f"{ny}"
        if (nz<0.0):
            nz = f"({nz})"
        else:
            nz = f"{nz}"
        if (x0<0.0):
            x0 = f"({x0})"
        else:
            x0 = f"{x0}"
        if (y0<0.0):
            y0 = f"({y0})"
        else:
            y0 = f"{y0}"
        if (z0<0.0):
            z0 = f"({z0})"
        else:
            z0 = f"{z0}"
        
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