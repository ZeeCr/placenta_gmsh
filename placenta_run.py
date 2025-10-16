import importlib

import placenta_main as pm

importlib.reload(pm)

mesh_success = False
        
while not mesh_success:
    try:
        mesh_success = pm.main()
    except (Exception, AttributeError) as e:
        print(f"Error occurred somewhere: {e}")
        mesh_success = False