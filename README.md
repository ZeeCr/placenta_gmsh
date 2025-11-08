# Idealised Placental Geometry Creation (Gmsh)

This repository contains the code needed to generate idealised placental geometries where the base shape is specified as a spherical cap.

The geometry creation heavily depends on OCC through Gmsh and various other geometrical functions and relations, some of which are labelled and commented (mostly in placenta_fns.py) and others which have been hard-coded where they are used. Gmsh is used to generate surface (2D) and volume (3D) meshes.

## Install

Set up Python environment via 
```
pip install -r requirements.txt
```

Configure placenta_const.py parameters, e.g. probability of a vein appearing in a lobule, mesh size in IVS etc.

## Run

Run python_main.py, it will run until it successfully creates a mesh for the given parameters and then save it inside ./meshes.

## Acknowledgements

This work was funded by Wellcome Leap as part of the SWIRL (Stillbirth When Risk is Low) project.

## Examples

### Top-down Interior and Bottom Exterior Surface View of a Geometry

<img src="https://github.com/user-attachments/assets/ee60bad4-48e8-417b-87aa-09c952794630" alt="View 1" width="360" />

<img src="https://github.com/user-attachments/assets/c7cfb92c-c5ca-4680-bddd-9e500ecc1afd" alt="View 2" width="360" />

### Schematic Visualisations of Two Geometries

<img src="https://github.com/user-attachments/assets/573ee47a-6339-48e1-832d-3a23d9d1a3e4" alt="Schematic 1" width="360" />

<img src="https://github.com/user-attachments/assets/2e4a34d3-415b-4fb4-bc04-ea7c29c29db0" alt="Schematic 2" width="360" />

### Surface Mesh

<img src="https://github.com/user-attachments/assets/d203a370-9374-4175-8f46-7584bc407fea" alt="Surface Mesh" width="480" />

### Surface Mesh During Moving Boundary 3D Flow Simulation

<img src="https://github.com/user-attachments/assets/66fe0c4b-1a1d-4a38-bf59-d1101193d750" alt="Asymmetrical Moving Boundary" width="480" />

## Notes

12/02/25: The main issue stopping meshing right now is the line that can't be removed which goes along the sphere / spherical cap when it's created in OCC. This seems to have to be somewhere on the surface of the spherical cap which causes issues when doing the fillet -> fragment operation, as it separates the fillet curve into different parts. This means that the line on the spherical cap surface for the e.g. vessel gets intersected, meaning the centroid is no longer where expected.