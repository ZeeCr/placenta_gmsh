import os
from pathlib import Path
import shutil

def read_idx():
    idx_file = Path('id_idx.dat')
    with idx_file.open('r') as file:
        idx = int(file.read())
        file.close()

def read_update_idx():
    idx_file = Path('id_idx.dat')
    with idx_file.open('r') as file:
        idx = int(file.read())
        file.close()
        
    # Increment the integer and save it back to 'idx.dat'
    new_idx = idx + 1
    with idx_file.open('w') as file:
        file.write(str(new_idx))
        file.close()
    return idx

def copy_code_to_directory(dir_path):
    cwd = Path.cwd()  # Get the current working directory path
    
    for filename in os.listdir(cwd):
    
        if (filename.endswith('.ipynb') or \
                filename.endswith('.py')) and \
                filename != 'gmsh_python_api.py':
            shutil.copy(filename,dir_path)

def move_files(meshID,idx) -> None:
    
    # Create subdirectory for the mesh
    cwd = Path.cwd()
    meshes_dir = cwd / 'meshes'
    sub_dir_pth = meshes_dir / meshID
    sub_dir_pth.mkdir(parents=True, exist_ok=True)
    
    # Create new index file, then move it into the new directory
    new_idx_file = sub_dir_pth / 'id_idx.dat'
    with new_idx_file.open('w') as file:
        file.write(str(idx))
        file.close()
    # Create new mesh ID file, then move it into the new directory
    new_id_file = sub_dir_pth / 'mesh_id.dat'
    with new_id_file.open('w') as file:
        file.write(meshID)
        file.close()

    # Move files into the newly created directory
    filenames = ['geom_info.csv', \
                'mesh_info.csv','cavity_info.csv','outlet_info.csv','c_wall_info.csv', \
                f'{meshID}.vtk',f'{meshID}.msh',f'{meshID}_surf.vtk']
    
    for nme in filenames:
        csv_file = Path(nme)
        csv_file.rename(sub_dir_pth / csv_file.name)
       
    copy_code_to_directory(sub_dir_pth) 

def store_mesh(gmsh,mesh_prefix) -> None:
    
    idx = read_update_idx()
    
    meshID = mesh_prefix + "_" + str(idx)
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(f'{meshID}_surf.vtk')
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize(method="Netgen")
    gmsh.write(f'{meshID}.vtk')
    gmsh.write(f'{meshID}.msh')

    move_files(meshID,idx)

    return None