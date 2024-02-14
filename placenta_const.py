import sys
import logging

import math

import numpy

num_err = 1.0e-12
tol = 1.0e-12
geom_tol = 1.0e-6
large_geom_tol = 1.0e-2
open_gmsh = 1

char_length = 10.0
######################## USER INFO ########################
#placenta_radius = 255.0/2.0
placenta_radius = 90.7 / char_length
placenta_volume = 428e3 / char_length**3 #mm^3
#placenta_height = 53.0

placentone_radius = 40.0/(2.0*char_length)
placentone_wall_thickness = 2.0 / char_length

placentone_removal_height = 18.0 / char_length

lobule_wall_thickness = placentone_wall_thickness/1.5
lobule_removal_height = placentone_removal_height/2.0
lobule_foronoi_type = 'standard' # ConcavePolygon ('concave') or Polygon ('standard')

no_random_placentones = 6 #If fixed_cotyledon_pts True, this is not used; else no_placentones = <----
no_inner_placentones = 2 #Unused now?

no_lobules_outer = 5 # How much splitting to do on placentones on boundary
no_lobules_inner = 5 # ^ inner

septal_artery_radius = 1.0 / char_length
septal_artery_funnel_radius = 0.75*septal_artery_radius
septal_vein_radius = 1.0 / char_length
septal_vein_funnel_radius = 0.75*septal_vein_radius

# Cavity info
cavity_minor_axis = 4.0*septal_artery_radius
cavity_height_divisor = 3.0
cavity_mesh_thickness = 8.0/(char_length*cavity_minor_axis)

#septal_vessel_separation = 2.0*septal_artery_radius + 8.0
septal_vessel_gap = 2.0 / char_length #+8 is way too big to allocate two per lobule, so just reducing to +2
septal_vessel_separation = cavity_minor_axis+septal_artery_radius+septal_artery_funnel_radius+septal_vessel_gap 
septal_vessel_length = 5.0 / char_length
# Offset is necessary - fusing an object on the sphere surface line that constructs the initial large sphere
# results in this sphere appearing back in the geometry
no_marginal_sinus_veins = 6 # Note, this is basically a maximum no. now
marginal_sinus_vein_radius = math.sqrt(2) / char_length#math.sqrt(2.0)#2.0*septal_vein_radius
marginal_sinus_vein_offset = 0.05*math.pi + 2*(2.0*math.pi/6.0)/4.0
marginal_fillet_radius = 0.75*marginal_sinus_vein_radius # Radius of septal veins fillet, total radius = vein_radius+fillet_radius
marginal_vein_height_from_top = 3.5 / char_length #marginal_vein_height = placenta_height - marginal_vein_height_from_top

#lobule_vein_length = 1.0*septal_vessel_length
vein_length = (3.0/10.0)*(placentone_wall_thickness*2.0)
fillet_radius = 0.75*septal_artery_radius # Radius of septal veins fillet, total radius = vein_radius+fillet_radius



# Cylinder above spherical cap info
top_cyl_height = 2*marginal_sinus_vein_radius #mm

# Radius of inner circle which contains inner Voronoi points and cells
inner_sub_radius = 0.5*placenta_radius

placenta_voronoi_outer_radius_offset = placenta_radius/3.0
outer_angle_variance = lambda no_p : 2.0*math.pi/(1.0*(no_p-no_inner_placentones))

# Artery bias
# 1 == 10% chance, 4 == 40% chance, 5 == 50% chance, 10 == 100% chance of there being an artery in a given lobule
artery_bias = 2
######################## END OF USER INFO ########################

######################## MESH SIZE ########################
DomSize = 5.0 / char_length #Large flow domain size
CavityMeshSize = DomSize/4.0
OuterCavityMeshSize = CavityMeshSize*2.0 #Threshold field, this is outer threshhold size
CavityApexSize = CavityMeshSize/2.0
IOSize = DomSize/12.0 #Inlet, outlet mesh size
VeinSize = IOSize/1.5
MarginalSize = IOSize*1.5
BasalSize = MarginalSize/math.sqrt(2)

mesh_offset_length = 1.0 / char_length
mesh_transition_length = 4.0 / char_length#mesh_offset_length
######################## END OF MESH SIZE ########################




######################## FIXED OBJECTS ########################
placenta_id = 'C'

fixed_cotyledon_pts = True

fixed_septal_wall_veins = False # Septal wall veins

fixed_lobules = True

stored_basal_veins = False

stored_cotyledon_wall_heights = True
random_cotyledon_wall_heights = True
# With random walls, placentone_removal_height - maxima_.. <= wall height <= placentone_rh + maxima_..
maxima_cotyledon_wall_heights = placentone_removal_height/2.0

adjust_stored_septal_vein_height = False # If wanting to choose specific z height along septal wall based on linear comb. between lower and upper pt adjust_septal_height_ratio between 0,1 : 0==bottom pt, 1==top pt
adjust_septal_height_ratio = (3.0/3.0)

# Cotyledon pts
if (fixed_cotyledon_pts):
    if (placenta_id == 'A'):
        # A
        no_stored_cotyledon = 6
        cotyledon_pts = numpy.array([ \
            (5.0,   -30.0),
            (60.0,  -20.0),
            (65.0,  75.0),
            (-55.0, 75.0),
            (-70.0, -25.0),
            (15.0,  -90.0)])     / char_length
        placenta_voronoi_outer_radius_offset = placenta_radius/3.0
    elif (placenta_id == 'B'):
        # B
        no_stored_cotyledon = 5
        cotyledon_pts = numpy.array([ \
            (0.67756,0.21709),
            (0.47257,-0.52778),
            (-0.23575,0.24429),
            (-0.13684,0.78583),
            (-0.35732,-0.50212)])    *placenta_radius
        placenta_voronoi_outer_radius_offset = 0.9*placenta_radius/3.0
    elif (placenta_id == 'C'):
        # C
        no_stored_cotyledon = 6
        cotyledon_pts = numpy.array([ \
            (0.73651,0.13054),
            (0.2479,-0.52718),
            (-0.045791,0.089986),
            (0.13726,0.67105),
            (-0.59743,0.41519),
            (-0.51683,-0.45271)])    *placenta_radius
        placenta_voronoi_outer_radius_offset = 0.25*placenta_radius/3.0
    elif (placenta_id == 'test'):
        # test
        no_stored_cotyledon = 6
        cotyledon_pts = numpy.array([ \
            (5.0,   -30.0),
            (60.0,  -20.0),
            (65.0,  75.0),
            (-55.0, 75.0),
            (-70.0, -25.0),
            (15.0,  -90.0)])     / char_length
    else:
        print(f"Unknown placenta ID: {placenta_id}")
        sys.exit(-1)
    no_placentones = no_stored_cotyledon
else:
    no_placentones = no_random_placentones
      
# Septal ewall veins 
if (fixed_septal_wall_veins):
    if (placenta_id == 'A'):
        # A
        no_stored_septal_wall_veins = 6
        septal_vein_face_nos = [ \
            196,
            199,
            204,
            206,
            209,
            210]
        septal_veins_pts = numpy.array([ \
            [9.939896744737137,36.63996006258339,8.654148512330027],
            [23.502933135385607,22.688453673302213,8.235940362253144],
            [5.051039503046587,47.576505966420086,16.431718530279078],
            [-16.623511560186216,27.55144192911244,9.561095966465395],
            [-33.36911309261268,-38.055342782645155,15.803095398685691],
            [-17.70424124785898,-64.88234738131207,20.474665395597654] ]) / char_length
        septal_veins_norms = numpy.array([ \
            [0.49974377700042866,0.8661732836732763,0.0],
            [-0.982580365023951,-0.1858381722612432,0.0],
            [-0.9999355550280739,-0.011352787794102178,0.0],
            [0.4924755505657512,-0.8703262791016714,0.0],
            [0.997428081042331,-0.07167442464514898,0.0],
            [-0.16431850243496293,0.9864073346024609,0.0] ])
    elif (placenta_id == 'B'):
        print(f"Not made yet")
        sys.exit(-1)
    elif (placenta_id == 'test'):
        print(f"Not made yet")
        sys.exit(-1)

# lobules
if (fixed_lobules):
    if (not(fixed_cotyledon_pts)):
        print(f"fixed_lobules but not fixed_cotyledon_pts")
        sys.exit(-1)
    else:
        f_no_fixed_cotyledon = no_stored_cotyledon
        f_no_lobule_pts = numpy.zeros(f_no_fixed_cotyledon)
        f_lobule_pts = numpy.zeros((5,2,f_no_fixed_cotyledon))
    if (placenta_id == 'A'):
        # Conccave polygon
        lobule_foronoi_type = 'concave'
        f_no_lobule_pts = numpy.array([5,5,5,5,5,5])
        f_lobule_pts[0:5,:,0] = [ \
        (-1.822490822847922, -0.21552970492373785),
        (0.5435076147786234, 2.0055326895206678),
        (-1.7284148612603711, -4.493517193146576),
        (1.2352081484086193, -1.2933575813657914),
        (1.8170747508466458, -4.331509151415845)]
        f_lobule_pts[0:5,:,1] = [ \
        (8.524084944885113, -2.357985543784532),
        (9.206600288451437, 0.9966648842651612),
        (4.930812902310696, -2.9485965232338267),
        (7.326086147344179, -5.910450822567935),
        (4.720390111164037, 1.0269858933135292)]
        f_lobule_pts[0:5,:,2] = [ \
        (2.210469514860771, 4.825164033945731),
        (3.9524918793330737, 7.267858689237109),
        (8.847432261160897, 3.658715481184858),
        (1.6165115038671298, 9.197406518448387),
        (5.830235902454074, 4.700619250505932)]
        f_lobule_pts[0:5,:,3] = [ \
        (-1.2528462389417345, 5.032076873443279),
        (-1.006844276887301, 9.055158836667173),
        (-4.370684556447047, 6.852743511286916),
        (-8.19149644244696, 4.258458251569338),
        (-4.505571213276138, 3.6629432220301297)]
        f_lobule_pts[0:5,:,4] = [ \
        (-4.961480446643668, 0.4035157236659625),
        (-5.91741796585386, -7.0332326876283915),
        (-7.832401133218099, -2.3332391492942737),
        (-9.034554580603958, 1.3689364785050326),
        (-4.899686668795452, -3.6682219144792985)]
        f_lobule_pts[0:5,:,5] = [ \
        (-3.745345504965583, -8.81970413636746),
        (1.9136937409653336, -8.445236980772785),
        (-0.9464082923924635, -7.886779882080132),
        (2.6514485141114865, -6.647190902624636),
        (5.52614413981973, -7.892374455189212)]
    elif (placenta_id == 'B'):
        # Standard polygon
        lobule_foronoi_type = 'standard'
        f_no_lobule_pts = numpy.array([5,5,5,5,5])
        f_lobule_pts[0:5,:,0] = [ \
            (8.816033133388002, -0.6285995131073095),
            (4.377104490172043, 0.5804849819000679),
            (7.825033770959012, 3.341020650163281),
            (4.089932846764778, 4.101573441648585),
            (6.609426445785637, 7.231855305201872)]
        f_lobule_pts[0:5,:,1] = [ \
            (2.1560592600818407, -8.643184919208405),
            (2.461895669002864, -2.8372629668904297),
            (4.448698981713428, -4.8530158384833955),
            (8.345939233096111, -3.979731236432717),
            (6.014066807165979, -7.037954765577831)]
        f_lobule_pts[0:5,:,2] = [ \
            (-9.358345498024297, 1.6293492041672286),
            (-2.6780723366869497, 3.4237106630621654),
            (0.10068370382302723, 1.0439844374158795),
            (-4.7224506921238865, 0.7280006755703694),
            (-7.573372602605648, 4.272413675928849)]
        f_lobule_pts[0:5,:,3] = [ \
            (-7.213860661919951, 7.045190285558808),
            (0.9388615458896103, 5.624398220285967),
            (-3.663781833668631, 6.756951954862664),
            (3.359355383907531, 8.331254169254688),
            (-0.7493205926627897, 7.846118300510383)]
        f_lobule_pts[0:5,:,4] = [ \
            (-5.173281029247199, -2.767166738665433),
            (-1.7343311658214902, -8.74004188096622),
            (-1.463553566806491, -4.0005650642919075),
            (-6.265809860228226, -6.873005778266479),
            (-9.047823239380502, -2.314865335124706)]
    elif (placenta_id == 'C'):
        # Standard polygon
        lobule_foronoi_type = 'standard'
        f_no_lobule_pts = numpy.array([5,5,5,5,5,5])
        f_lobule_pts[0:5,:,0] = [ \
            (7.842361112820369, -1.8664511825769978),
            (4.825624578644058, -0.46874548422555984),
            (6.061795693568777, 1.0806477614642442),
            (8.083348166119384, 2.935010140880837),
            (5.4356941698247585, 3.721517844556705)]
        f_lobule_pts[0:5,:,1] = [ \
            (4.9017230322339, -6.712175059768762),
            (2.2919006217129043, -5.416605367410896),
            (0.9955275082982538, -3.1723990577738386),
            (0.18290413710150194, -7.50340259741484),
            (5.0499028076981505, -3.549285300447282)]
        f_lobule_pts[0:5,:,2] = [ \
            (1.8960554383031148, 1.4673913103913327),
            (1.335867515294676, -0.6413286279468228),
            (-0.22694777144964173, 2.7264701587979603),
            (-1.2133681553225236, -0.9880274689779752),
            (-2.3744086846896773, 1.1960994979376772)]
        f_lobule_pts[0:5,:,3] = [ \
            (-0.8272361745693225, 7.939731673359156),
            (5.073993351676629, 6.414765398523735),
            (-0.07509306343268664, 5.129667325448646),
            (2.850008968522932, 7.556262659956343),
            (2.896681166599391, 4.273343213439627)]
        f_lobule_pts[0:5,:,4] = [ \
            (-7.792897828363486, 1.337719624343952),
            (-6.772037248880285, 5.172533243027208),
            (-5.220526329744117, 3.3931558536404838),
            (-3.4059266791185395, 6.097855611984089),
            (-4.146952594066451, 1.5740414377947347)]
        f_lobule_pts[0:5,:,5] = [ \
            (-6.825320557601007, -4.063381325244571),
            (-3.2199038503624804, -2.968729905410642),
            (-5.419657963775983, -6.035950112568491),
            (-2.779227279862568, -6.9153106657694945),
            (-6.949338337166241, -1.5477502101953458)]
    elif (placenta_id == 'test'):
        print(f"test not in yet")
        sys.exit(-1)

# Basal palte veins
if (stored_basal_veins):
    if (placenta_id == 'A'):
        stored_stop_at_basal_veins_added = numpy.zeros(30,dtype=int)
        stored_basal_vein_powers = numpy.zeros(30,dtype=int)
        stored_no_basal_veins_to_add = numpy.zeros(30,dtype=int)
        stored_stop_at_basal_veins_added[:] = 2
        stored_no_basal_veins_to_add[:] = [1,2,1,1,1,0,2,2,0,2,2,2,2,0,0,1,0,1,0,1,2,2,0,1,1,0,2,1,1,1]
        stored_basal_vein_powers[:] =     [2,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,2,0,0,0,1,1,0,0,1,1,1]
    elif (placenta_id == 'B'):
        stored_stop_at_basal_veins_added = numpy.zeros(25,dtype=int)
        stored_basal_vein_powers = numpy.zeros(25,dtype=int)
        stored_no_basal_veins_to_add = numpy.zeros(25,dtype=int)
        stored_stop_at_basal_veins_added[:] = 2
        stored_no_basal_veins_to_add[:] = [1,1,1,0,2,2,1,0,1,2,1,0,2,1,1,0,2,0,1,1,1,1,2,2,0]
        stored_basal_vein_powers[:] =     [1,1,0,0,0,0,1,0,1,0,2,0,0,1,2,0,0,0,2,2,1,1,0,0,0]
    elif (placenta_id == 'C'):
        stored_stop_at_basal_veins_added = numpy.zeros(30,dtype=int)
        stored_basal_vein_powers = numpy.zeros(30,dtype=int)
        stored_no_basal_veins_to_add = numpy.zeros(30,dtype=int)
        stored_stop_at_basal_veins_added[:] = 2
        stored_no_basal_veins_to_add[:] = [0,1,1,2,2,1,0,0,1,2,2,1,1,2,0,1,0,0,0,1,1,1,0,2,0,1,1,0,1,2]
        stored_basal_vein_powers[:] =     [0,2,1,0,0,2,0,0,2,0,0,2,1,0,0,2,0,0,0,2,2,2,0,0,0,1,2,0,2,0]
    elif (placenta_id == 'test'):
        print(f"test not in")
        sys.exit(-1)
        

cotyledon_wall_heights = numpy.empty(no_placentones)
lobule_wall_heights = numpy.empty(no_placentones)
if (stored_cotyledon_wall_heights):
    if (placenta_id == 'A'):
        cotyledon_wall_heights = [ \
            14.0,14.0,14.0,14.0,14.0,14.0]
    elif (placenta_id == 'C'):
        cotyledon_wall_heights[:] = [ \
             1.66008201,1.82388559,1.5818771 ,0.81603679,1.65974472,1.24322865]
        
        cotyledon_wall_heights[:] = [1.06535941,1.76038427,1.63020321,1.87078275,0.74165982,1.60105765] # remove after
    else:
        print(f"stored_cotyledon_wall_heights: not implemented for case {placenta_id}")
        sys.exit(-1)
elif (random_cotyledon_wall_heights):
    for i in range(0,no_placentones):
        rand_sign = numpy.random.choice([-1,1])
        rand_real = placentone_removal_height + \
            rand_sign*numpy.random.rand()*maxima_cotyledon_wall_heights
        cotyledon_wall_heights[i] = rand_real
else:
    cotyledon_wall_heights[:] = placentone_removal_height
lobule_wall_heights[:] = cotyledon_wall_heights[:]/2.0

#################### END OF FIXED OBJECTS #####################








######################## INITIALISE QUANTITIES ########################
tol = 1.0e-4 / char_length

top_cyl_vol = math.pi*top_cyl_height*placenta_radius**2

placenta_height_roots = numpy.roots([1.0,0.0,3.0*placenta_radius**2,-6.0*placenta_volume/math.pi]) #V = (1/6)*pi*h*(3r^2 + h^2), spherical cap only
#placenta_height_roots = numpy.roots([1.0,0.0,3.0*placenta_radius**2,6.0*(top_cyl_height*placenta_radius**2 - placenta_volume/math.pi)]) #V = (1/6)*pi*h*(3r^2 + h^2)
placenta_height = numpy.real(placenta_height_roots[2])

placentone_height = placenta_height-(1.0/char_length)

initial_sphere_radius = (placenta_radius**2 + placenta_height**2)/(2.0*placenta_height)

initial_sphere_centre = numpy.array([0.0,0.0,initial_sphere_radius])

# Circ eval
circ_eval = lambda x,y : x**2 + y**2
# rsr = removal sphere radius, rh = removal height
removal_sphere_radius = lambda rh : ( placenta_radius**2 + (placenta_height - rh)**2 )/( 2.0 * (placenta_height - rh) )
# Project an (x,y) point onto lower half of larger placentone removal sphere
sphere_surface_eval = lambda x,y,rsr,rh : rsr+rh - math.sqrt(rsr**2 - (x**2 + y**2))

marginal_vein_height = placenta_height-marginal_vein_height_from_top

cyl_offset_length = 1.0e-3 / char_length
mesh_offset_length = 1.0e-1 / char_length

# Values to determine how far up / down the centroid of wall veins should at least be
# Cahgnes whether in / on wall due to funnel shape on wall
wall_vein_buffer_on_wall = septal_vein_radius+1.4*septal_vein_funnel_radius
wall_vein_buffer_in_wall = 1.4*septal_vein_radius
# Track number of inlets, outlets
#no_inlets = 0
#no_outlets = 0
####################### END OF INITIALISE QUANTITIES #######################



############################### PYTHON OBJS ################################

# Set up log
logger = logging.getLogger()
# Clear previous logs
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s \n %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# ERROR, DEBUG, INFO, WARNING, CRITICAL
logger.setLevel(logging.ERROR)



def set_gmsh_optns(gmsh):
    
    gmsh.model.add("placenta")
    
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    
    #gmsh.option.setNumber("Geometry.MatchMeshTolerance",1.0e-6)
    #gmsh.option.setNumber("Geometry.Tolerance",1.0e-8)
    #gmsh.option.setNumber("Geometry.ToleranceBoolean",1.0e-9)
    gmsh.option.setNumber("Geometry.MatchMeshTolerance",1.0e-4)
    gmsh.option.setNumber("Geometry.Tolerance",1.0e-4)
    gmsh.option.setNumber("Geometry.ToleranceBoolean",1.0e-4)
    
    # I initially changed these due to intersection issues with congruent circles on surface of sphere
    # I don't remember if the above or this fixed it, or both, but having too many line and curve nodes
    # leads to ill-shaped elements in mesh
    gmsh.option.setNumber("Mesh.MinimumLineNodes",5)
    gmsh.option.setNumber("Mesh.MinimumCircleNodes",20)
    gmsh.option.setNumber("Mesh.MinimumCurveNodes",5)
    
    #gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    #gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    #gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    ######################## GMSH OPTIONS ########################
    # Support separate numbering system for newp, news, newv etc.
    gmsh.model.geo.OldNewReg = 0

    # Apparently more accurate bounds on the bounding boxes - from t18.py
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    ##################### END OF GMSH OPTIONS #####################
    
    return gmsh

