# How-to

Configure placenta_const.py parameters, e.g. probability of a vein appearing in a lobule, mesh size in IVS etc.

Run python_main.py, it will run until it successfully creates a mesh for the given parameters and then save it inside ./meshes

# Notes

12/02/25: The main issue stopping meshing right now is the line that can't be removed which goes along the sphere / spherical cap when it's created in OCC. This seems to have to be somewhere on the surface of the spherical cap which causes issues when doing the fillet -> fragment operation, as it separates the fillet curve into different parts.