This repository contains the code needed to generate idealised placental geometries where the base shape is specified as a spherical cap.

The geometry creation heavily depends on OCC through Gmsh and various other geometrical functions and relations, some of which are labelled and commented (mostly in placenta_fns.py) and others which have been hard-coded where they are used. Gmsh is used to generate surface (2D) and volume meshes.

# Install

Set up Python environment via 
```
pip install -r requirements.txt'
```

Configure placenta_const.py parameters, e.g. probability of a vein appearing in a lobule, mesh size in IVS etc.

# Run

Run python_main.py, it will run until it successfully creates a mesh for the given parameters and then save it inside ./meshes.

# Acknowledgements

This work was funded by Wellcome Leap as part of the SWIRL (Stillbirth When Risk is Low) project.

# Notes

12/02/25: The main issue stopping meshing right now is the line that can't be removed which goes along the sphere / spherical cap when it's created in OCC. This seems to have to be somewhere on the surface of the spherical cap which causes issues when doing the fillet -> fragment operation, as it separates the fillet curve into different parts. This means that the line on the spherical cap surface for the e.g. vessel gets intersected, meaning the centroid is no longer where expected.