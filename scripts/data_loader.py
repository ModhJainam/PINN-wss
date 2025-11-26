import torch
import numpy as np
import vtk
from vtk.util import numpy_support as VN
import os

# ==========================================
#   GLOBAL CONFIGURATION (MANUAL PARAMETERS)
# ==========================================

# 1. File Paths
# ------------------------------------------
DATA_DIR = "../Data/3D-aneurysm/" # Base directory
FILE_DOMAIN = "IA_mesh3D_nearwall_small_physical.vtu" # The full fluid domain
FILE_WALL   = "IA_nearwall_wall_small.vtk"            # The rigid wall surface
FILE_INLET  = "IA_nearwall_outer_small.vtk"           # Inlet/Outlet surface points
FILE_FIELD  = "IA_3D_unsteady3.vtu"                   # The ground truth flow field

# 2. Physics & Normalization Scales
# ------------------------------------------
# Scales to normalize spatial coords to approx [0, 1]
X_SCALE  = 3.0   
YZ_SCALE = 2.0   
# Scale to normalize velocity to approx 1.0
U_SCALE  = 1.0   

# 3. Inlet Boundary Condition Profile
# ------------------------------------------
# Defines the parabolic profile: u = y * (Y_TOP - y) / REF_H * U_MAX
U_INLET_MAX   = 0.5     # Peak velocity at inlet
INLET_Y_TOP   = 0.3     # The upper y-coordinate of the inlet
INLET_REF_H   = 0.0225  # Geometric factor for the parabola (usually radius^2 or width^2)

# 4. Sparse Data Sampling
# ------------------------------------------
# Take every Nth point from the boundary file to create sparse measurement points
DATA_SAMPLE_RATE = 200  
VELOCITY_FIELD_NAME = 'f_17' # The array name in the VTK file (check via Paraview)

# ==========================================
#   HELPER FUNCTIONS
# ==========================================

def load_vtk_data(filepath):
    """
    Generic loader that handles both .vtu (XML) and .vtk (Legacy) files.
    Returns the vtkOutput object.
    """
    full_path = os.path.join(DATA_DIR, filepath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    if filepath.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif filepath.endswith('.vtk'):
        reader = vtk.vtkPolyDataReader() # Usually for boundary surfaces
    else:
        # Fallback for generic unstructured grids
        reader = vtk.vtkUnstructuredGridReader()
    
    reader.SetFileName(full_path)
    reader.Update()
    return reader.GetOutput()

def vtk_to_tensor(vtk_data, scale_x, scale_yz):
    """
    Extracts points from VTK object and returns normalized PyTorch tensors.
    """
    # Optimized extraction using numpy_support (much faster than loops)
    points_array = VN.vtk_to_numpy(vtk_data.GetPoints().GetData())
    
    # Split and Normalize
    x = points_array[:, 0:1] / scale_x
    y = points_array[:, 1:2] / scale_yz
    z = points_array[:, 2:3] / scale_yz
    
    return (torch.from_numpy(x).float(), 
            torch.from_numpy(y).float(), 
            torch.from_numpy(z).float())

# ==========================================
#   MAIN DATA PREPARATION FUNCTION
# ==========================================

def get_training_data():
    """
    Loads all files, applies physics BCs, normalizes data, and returns
    a dictionary containing all tensors ready for the training loop.
    """
    print("--- Loading PINN Training Data ---")

    # --------------------------------------
    # 1. Collocation Points (Domain Physics)
    # --------------------------------------
    print(f"Loading Domain: {FILE_DOMAIN}")
    domain_vtk = load_vtk_data(FILE_DOMAIN)
    x_c, y_c, z_c = vtk_to_tensor(domain_vtk, X_SCALE, YZ_SCALE)
    
    # --------------------------------------
    # 2. Wall Boundary Conditions (No-Slip)
    # --------------------------------------
    print(f"Loading Wall BC: {FILE_WALL}")
    wall_vtk = load_vtk_data(FILE_WALL)
    xb, yb, zb = vtk_to_tensor(wall_vtk, X_SCALE, YZ_SCALE)
    
    # Enforce No-Slip (u=v=w=0)
    ub = torch.zeros_like(xb)
    vb = torch.zeros_like(yb)
    wb = torch.zeros_like(zb)

    # --------------------------------------
    # 3. Inlet Boundary Conditions
    # --------------------------------------
    print(f"Loading Inlet BC: {FILE_INLET}")
    inlet_vtk = load_vtk_data(FILE_INLET)
    xb_in, yb_in, zb_in = vtk_to_tensor(inlet_vtk, X_SCALE, YZ_SCALE)
    
    # Calculate Parabolic Profile
    # Logic: u = y_normalized * (Y_TOP_normalized - y_normalized) ...
    # Note: We must use the 'physical' y-coordinates for the profile calculation 
    # before full normalization, or adjust the constants. 
    # Here we replicate the paper's logic using the normalized coordinates * scaling back implicitly 
    # or applying the formula to the tensor directly.
    
    # Replicating original logic: (yb_in) * ( 0.3 - yb_in ) / 0.0225 * U_BC_in
    # Note: In original code, yb_in was ALREADY normalized by YZ_SCALE. 
    # We assume INLET_Y_TOP and INLET_REF_H are tailored to the NORMALIZED coordinate space 
    # OR the physical space. *Based on the paper's code, they calculated on Normalized Y.*
    
    ub_in = (yb_in) * (INLET_Y_TOP - yb_in) / INLET_REF_H * U_INLET_MAX
    vb_in = torch.zeros_like(yb_in)
    wb_in = torch.zeros_like(zb_in)

    # --------------------------------------
    # 4. Sparse Measurement Data (for Loss_Data)
    # --------------------------------------
    print(f"Generating Sparse Data from: {FILE_FIELD}")
    
    # A. Load the geometry to sample FROM (using Inlet/Outer file as base per original script)
    sample_source_vtk = load_vtk_data(FILE_INLET)
    source_points = VN.vtk_to_numpy(sample_source_vtk.GetPoints().GetData())
    
    # B. Subsample indices
    indices = np.arange(0, source_points.shape[0], DATA_SAMPLE_RATE)
    sampled_points = source_points[indices]
    
    # Create a VTK object for the sampled points to use in ProbeFilter
    vtk_sampled_pts = vtk.vtkPoints()
    for pt in sampled_points:
        vtk_sampled_pts.InsertNextPoint(pt[0], pt[1], pt[2])
    
    sample_poly = vtk.vtkPolyData()
    sample_poly.SetPoints(vtk_sampled_pts)
    
    # C. Load the Full Field Data (Ground Truth)
    field_vtk = load_vtk_data(FILE_FIELD)
    
    # D. Probe (Interpolate) the field data at the sampled points
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(sample_poly)
    probe.SetSourceData(field_vtk)
    probe.Update()
    
    # E. Extract Velocity
    valid_mask = probe.GetOutput().GetPointData().GetArray('vtkValidPointMask')
    valid_mask = VN.vtk_to_numpy(valid_mask)
    
    velocity_data = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(VELOCITY_FIELD_NAME))
    
    # Normalize Velocity
    u_data = velocity_data[:, 0:1] / U_SCALE
    v_data = velocity_data[:, 1:2] / U_SCALE
    w_data = velocity_data[:, 2:3] / U_SCALE
    
    # Normalize Coordinates of the sampled points
    x_d = sampled_points[:, 0:1] / X_SCALE
    y_d = sampled_points[:, 1:2] / YZ_SCALE
    z_d = sampled_points[:, 2:3] / YZ_SCALE

    # Convert to Tensor
    xd_t = torch.from_numpy(x_d).float()
    yd_t = torch.from_numpy(y_d).float()
    zd_t = torch.from_numpy(z_d).float()
    ud_t = torch.from_numpy(u_data).float()
    vd_t = torch.from_numpy(v_data).float()
    wd_t = torch.from_numpy(w_data).float()

    print(f"Data Loaded Successfully.")
    print(f"  - Collocation Pts: {len(x_c)}")
    print(f"  - Wall Pts: {len(xb)}")
    print(f"  - Inlet Pts: {len(xb_in)}")
    print(f"  - Sparse Data Pts: {len(xd_t)}")

    return {
        # Collocation
        "collocation": (x_c, y_c, z_c),
        # Wall BC
        "wall": (xb, yb, zb, ub, vb, wb),
        # Inlet BC
        "inlet": (xb_in, yb_in, zb_in, ub_in, vb_in, wb_in),
        # Measurement Data
        "data": (xd_t, yd_t, zd_t, ud_t, vd_t, wd_t)
    }

if __name__ == "__main__":
    # Test the loader
    data = get_training_data()