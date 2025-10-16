import os
from pathlib import Path
import shutil

def read_idx():
    idx_file = Path('id_idx.dat')
    with idx_file.open('r') as file:
        idx = int(file.read())
        file.close()
        
    return idx

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
    data_dir_pth = sub_dir_pth / "data"
    code_dir_pth = sub_dir_pth / "code"
    sub_dir_pth.mkdir(parents=True, exist_ok=True)
    data_dir_pth.mkdir(parents=True, exist_ok=True)
    code_dir_pth.mkdir(parents=True, exist_ok=True)
    
    # Create new index file, then move it into the new directory
    new_idx_file = data_dir_pth / 'id_idx.dat'
    with new_idx_file.open('w') as file:
        file.write(str(idx))
        file.close()
    # Create new mesh ID file, then move it into the new directory
    new_id_file = data_dir_pth / 'mesh_id.dat'
    with new_id_file.open('w') as file:
        file.write(meshID)
        file.close()

    # Move files into the newly created directory
    filenames = ['geom_info.csv','mesh_info.csv', \
                'cavity_info.csv','outlet_info.csv','c_wall_info.csv','c_loc.csv', \
                'mesh.vtk','mesh.msh','mesh_surf.vtk']
    
    for nme in filenames:
        csv_file = Path(nme)
        csv_file.rename(data_dir_pth / csv_file.name)
       
    copy_code_to_directory(code_dir_pth)
    
def get_no_eles_in_mesh(gmsh):
    
    _, ele_tags, _ = gmsh.model.mesh.getElements(dim=3)
    no_eles = sum(len(tags) for tags in ele_tags)
    print(f"No. 3D eles: {no_eles}")
    
    return no_eles

def store_mesh(gmsh,mesh_prefix) -> None:
    
    default_err_return = [-1,-1,-1]
    
    idx = read_update_idx()
    
    meshID = str(idx) + "_" + mesh_prefix
    
    gmsh.model.occ.synchronize()
    
    # First run 3D mesher to check it'll mesh
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print(f"Error occurred while generating 3D mesh: {e}")
        return default_err_return

    gmsh.model.mesh.optimize(method="Netgen")

    no_eles = get_no_eles_in_mesh(gmsh)
    print(f"No eles in mesh: {no_eles}")
    
    if (no_eles > 0):
        gmsh.write('mesh.vtk')
        gmsh.write('mesh.msh')
        
        gmsh.model.mesh.generate(2)
        gmsh.write('mesh_surf.vtk')

        return [meshID,idx,no_eles]
    
    else:
        print(f"ERROR: No elements found in mesh")
        print(f"No eles in mesh: {no_eles}")
        
        return default_err_return