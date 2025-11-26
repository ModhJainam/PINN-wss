import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from vtk.util import numpy_support as VN
import vtk

# Import local modules
import model as pinn_model
import data_loader

# ==========================================
#   CONFIGURATION
# ==========================================

# Path to the specific experiment run you want to evaluate
# (Change this to match the folder name created by train.py)
EXPERIMENT_DIR = "experiments/run_2023XXXX_XXXXXX" 
CHECKPOINT_EPOCH = 8500  # The epoch you want to load (e.g., last one)

# Output directory for results
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical Scales (Must match train.py/data_loader.py)
X_SCALE  = data_loader.X_SCALE
YZ_SCALE = data_loader.YZ_SCALE
U_SCALE  = data_loader.U_SCALE

def load_trained_model(checkpoint_path):
    """Instantiates models and loads weights from checkpoints."""
    print(f"Loading models from: {checkpoint_path}")
    
    net_u = pinn_model.Net_U().to(DEVICE)
    net_v = pinn_model.Net_V().to(DEVICE)
    net_w = pinn_model.Net_W().to(DEVICE)
    net_p = pinn_model.Net_P().to(DEVICE)

    # Load states (map_location ensures it works even if trained on GPU and eval on CPU)
    net_u.load_state_dict(torch.load(os.path.join(checkpoint_path, "net_u.pt"), map_location=DEVICE))
    net_v.load_state_dict(torch.load(os.path.join(checkpoint_path, "net_v.pt"), map_location=DEVICE))
    net_w.load_state_dict(torch.load(os.path.join(checkpoint_path, "net_w.pt"), map_location=DEVICE))
    net_p.load_state_dict(torch.load(os.path.join(checkpoint_path, "net_p.pt"), map_location=DEVICE))

    return net_u, net_v, net_w, net_p

def compute_full_field_prediction(net_u, net_v, net_w, net_p, x, y, z, batch_size=2048):
    """
    Runs inference on the full domain in batches to avoid Out-Of-Memory errors.
    Returns numpy arrays of predicted u, v, w, p.
    """
    net_u.eval(); net_v.eval(); net_w.eval(); net_p.eval()
    
    n_points = x.shape[0]
    u_pred, v_pred, w_pred, p_pred = [], [], [], []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            # Create batch
            x_b = x[i:i+batch_size].to(DEVICE)
            y_b = y[i:i+batch_size].to(DEVICE)
            z_b = z[i:i+batch_size].to(DEVICE)
            net_in = torch.cat((x_b, y_b, z_b), 1)

            # Predict
            u_pred.append(net_u(net_in).cpu().numpy())
            v_pred.append(net_v(net_in).cpu().numpy())
            w_pred.append(net_w(net_in).cpu().numpy())
            p_pred.append(net_p(net_in).cpu().numpy())

    # Concatenate and flatten
    u_pred = np.concatenate(u_pred, axis=0).flatten()
    v_pred = np.concatenate(v_pred, axis=0).flatten()
    w_pred = np.concatenate(w_pred, axis=0).flatten()
    p_pred = np.concatenate(p_pred, axis=0).flatten()

    return u_pred, v_pred, w_pred, p_pred

def compute_wall_shear_stress(net_u, net_v, net_w, wall_vtk_path, rho=1.0, mu=0.0035):
    """
    Computes WSS magnitude using exact neural network derivatives.
    """
    print(f"Computing WSS along wall: {wall_vtk_path}")
    
    # 1. Load Wall Mesh and Compute Normals
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(wall_vtk_path)
    reader.Update()
    
    # Generate Normals if not present
    normal_gen = vtk.vtkPolyDataNormals()
    normal_gen.SetInputData(reader.GetOutput())
    normal_gen.ComputePointNormalsOn()
    normal_gen.Update()
    wall_data = normal_gen.GetOutput()
    
    # Extract points and normals
    points = VN.vtk_to_numpy(wall_data.GetPoints().GetData())
    normals = VN.vtk_to_numpy(wall_data.GetPointData().GetNormals())
    
    # 2. Prepare Inputs for PINN
    x = torch.tensor(points[:, 0:1] / X_SCALE, dtype=torch.float32, requires_grad=True).to(DEVICE)
    y = torch.tensor(points[:, 1:2] / YZ_SCALE, dtype=torch.float32, requires_grad=True).to(DEVICE)
    z = torch.tensor(points[:, 2:3] / YZ_SCALE, dtype=torch.float32, requires_grad=True).to(DEVICE)
    
    # 3. Compute Gradients (Strain Rate) via Autograd
    # We process in chunks to save memory if needed, but here's the core logic:
    net_in = torch.cat((x, y, z), 1)
    
    u = net_u(net_in)
    v = net_v(net_in)
    w = net_w(net_in)
    
    # Helper for gradients
    def get_grad(val, w_r_t):
        return torch.autograd.grad(val, w_r_t, 
                                   grad_outputs=torch.ones_like(val), 
                                   create_graph=False)[0]

    # Calculate Velocity Gradients (du/dx, du/dy, etc.)
    # Note: We must rescale derivatives back to physical space!
    # du_phys/dx_phys = (du_norm/dx_norm) * (U_SCALE / X_SCALE)
    
    u_x = get_grad(u, x) * (U_SCALE / X_SCALE)
    u_y = get_grad(u, y) * (U_SCALE / YZ_SCALE)
    u_z = get_grad(u, z) * (U_SCALE / YZ_SCALE)
    
    v_x = get_grad(v, x) * (U_SCALE / X_SCALE)
    v_y = get_grad(v, y) * (U_SCALE / YZ_SCALE)
    v_z = get_grad(v, z) * (U_SCALE / YZ_SCALE)
    
    w_x = get_grad(w, x) * (U_SCALE / X_SCALE)
    w_y = get_grad(w, y) * (U_SCALE / YZ_SCALE)
    w_z = get_grad(w, z) * (U_SCALE / YZ_SCALE)
    
    # 4. Compute WSS Vector
    # Stress Tensor (incompressible Newtonian): sigma = mu * (grad_u + grad_u^T)
    # WSS vector = t - (t.n)n, where t = sigma . n
    
    # Convert to numpy for vector math
    nx, ny, nz = normals[:,0], normals[:,1], normals[:,2]
    
    # Assemble Strain Rate Tensor components (S_ij = du_i/dx_j + du_j/dx_i)
    # We do this calculation on CPU/Numpy for simplicity with array shapes
    ux, uy, uz = u_x.detach().cpu().numpy().flatten(), u_y.detach().cpu().numpy().flatten(), u_z.detach().cpu().numpy().flatten()
    vx, vy, vz = v_x.detach().cpu().numpy().flatten(), v_y.detach().cpu().numpy().flatten(), v_z.detach().cpu().numpy().flatten()
    wx, wy, wz = w_x.detach().cpu().numpy().flatten(), w_y.detach().cpu().numpy().flatten(), w_z.detach().cpu().numpy().flatten()
    
    wss_magnitudes = []
    
    for i in range(len(ux)):
        # Deformation rate tensor D
        D = np.array([
            [2*ux[i],    uy[i]+vx[i], uz[i]+wx[i]],
            [vx[i]+uy[i], 2*vy[i],    vz[i]+wy[i]],
            [wx[i]+uz[i], wy[i]+vz[i], 2*wz[i]]
        ])
        
        # Normal vector
        n = np.array([nx[i], ny[i], nz[i]])
        
        # Traction vector t = mu * D * n (simplified for incompressible wall)
        t = mu * D @ n
        
        # Normal component of traction
        tn = np.dot(t, n)
        
        # Shear component vector (tangential)
        tau = t - tn * n
        
        # Magnitude
        wss_magnitudes.append(np.linalg.norm(tau))
        
    return points[:, 0], np.array(wss_magnitudes) # Return X-coords and WSS

def save_vtk_results(original_vtk_path, save_path, predictions):
    """
    Clones the original domain mesh and appends the PINN predictions as new scalar fields.
    This creates a file ready for Paraview visualization.
    """
    # 1. Read the Reference Mesh
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(original_vtk_path)
    reader.Update()
    output_grid = reader.GetOutput()
    
    # 2. Add Predicted Arrays
    # Helper to add array to VTK point data
    def add_array(name, data):
        arr = VN.numpy_to_vtk(data)
        arr.SetName(name)
        output_grid.GetPointData().AddArray(arr)

    # Unpack predictions
    u_p, v_p, w_p, p_p = predictions
    u_ref, v_ref, w_ref = predictions[4], predictions[5], predictions[6] # Ground Truths

    # Add Prediction Fields
    add_array("Velocity_PINN_U", u_p)
    add_array("Velocity_PINN_V", v_p)
    add_array("Velocity_PINN_W", w_p)
    add_array("Pressure_PINN", p_p)

    # Add Error Fields (Absolute difference)
    add_array("Error_U", np.abs(u_p - u_ref))
    add_array("Error_V", np.abs(v_p - v_ref))
    add_array("Error_W", np.abs(w_p - w_ref))

    # 3. Write Output
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(output_grid)
    writer.Write()
    print(f"VTK Results saved to: {save_path}")

def plot_results_extended(x, y, z, u_pred, v_pred, w_pred, wss_x, wss_vals):
    """
    Generates paper-quality plots:
    1. 2D Slice with Velocity Vectors + Magnitude Color
    2. WSS vs X graph
    """
    
    # --- PLOT 1: Velocity Magnitude + Vectors (2D Slice) ---
    # Select a slice (e.g., Z near 0)
    slice_mask = np.abs(z - np.mean(z)) < 0.05
    x_slice = x[slice_mask]
    y_slice = y[slice_mask]
    u_slice = u_pred[slice_mask]
    v_slice = v_pred[slice_mask]
    
    vel_mag = np.sqrt(u_slice**2 + v_slice**2)
    
    plt.figure(figsize=(12, 5))
    
    # Contour for Magnitude
    plt.tricontourf(x_slice, y_slice, vel_mag, levels=100, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Velocity Magnitude (m/s)')
    
    # Quiver for Direction (Subsample points to avoid clutter)
    skip = 10 
    plt.quiver(x_slice[::skip], y_slice[::skip], 
               u_slice[::skip], v_slice[::skip], 
               color='white', scale=20, width=0.002)
    
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Velocity Flow Field (Z-Slice)')
    plt.savefig(os.path.join(RESULTS_DIR, 'flow_vector_slice.png'), dpi=300)
    plt.close()
    
    # --- PLOT 2: WSS vs X Distance ---
    # Sort by X to make a clean line plot
    sorted_indices = np.argsort(wss_x)
    x_sorted = wss_x[sorted_indices]
    wss_sorted = wss_vals[sorted_indices]
    
    # Calculate Rolling Average for smoother visualization (optional)
    wss_smooth = pd.Series(wss_sorted).rolling(window=20).mean()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_sorted, wss_sorted, alpha=0.3, s=1, color='gray', label='Raw Point Data')
    plt.plot(x_sorted, wss_smooth, color='red', linewidth=2, label='Mean WSS')
    
    plt.xlabel('Distance along Vessel (X)')
    plt.ylabel('Wall Shear Stress (Pa)')
    plt.title('Wall Shear Stress Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'wss_distribution.png'), dpi=300)
    plt.close()

def main():
    print("--- Starting Evaluation ---")
    
    # 1. Load Data (Ground Truth & Domain)
    # We load the full domain from the original data loader
    print("Loading Ground Truth Data...")
    
    # Manually load full field VTK to get ground truth vectors
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(os.path.join(data_loader.DATA_DIR, data_loader.FILE_FIELD))
    reader.Update()
    data = reader.GetOutput()
    points = VN.vtk_to_numpy(data.GetPoints().GetData())
    velocities = VN.vtk_to_numpy(data.GetPointData().GetArray(data_loader.VELOCITY_FIELD_NAME))

    # Prepare inputs for PINN (Normalize)
    x_phys = points[:, 0:1]
    y_phys = points[:, 1:2]
    z_phys = points[:, 2:3]
    
    u_true = velocities[:, 0]
    v_true = velocities[:, 1]
    w_true = velocities[:, 2]

    # Normalize inputs for model
    x_in = torch.tensor(x_phys / X_SCALE, dtype=torch.float32)
    y_in = torch.tensor(y_phys / YZ_SCALE, dtype=torch.float32)
    z_in = torch.tensor(z_phys / YZ_SCALE, dtype=torch.float32)

    # 2. Load Model
    ckpt_path = os.path.join(EXPERIMENT_DIR, f"checkpoints_epoch_{CHECKPOINT_EPOCH}")
    net_u, net_v, net_w, net_p = load_trained_model(ckpt_path)

    # 3. Inference
    print("Running Inference on Full Field...")
    u_pred_norm, v_pred_norm, w_pred_norm, p_pred = compute_full_field_prediction(
        net_u, net_v, net_w, net_p, x_in, y_in, z_in
    )

    # 4. Denormalize Predictions (Back to Physical Units)
    u_pred = u_pred_norm * U_SCALE
    v_pred = v_pred_norm * U_SCALE
    w_pred = w_pred_norm * U_SCALE
    # Pressure is usually relative, we leave it scaled or unscaled depending on analysis needs.
    # Here we leave it as the raw network output (non-dimensional pressure).

    # 5. Compute Metrics
    def relative_l2(pred, true):
        return np.linalg.norm(pred - true) / np.linalg.norm(true)

    err_u = relative_l2(u_pred, u_true)
    err_v = relative_l2(v_pred, v_true)
    err_w = relative_l2(w_pred, w_true)
    err_mag = relative_l2(np.sqrt(u_pred**2 + v_pred**2 + w_pred**2), 
                          np.sqrt(u_true**2 + v_true**2 + w_true**2))

    print(f"\nResults (Epoch {CHECKPOINT_EPOCH}):")
    print(f"  Relative L2 Error u: {err_u:.5f}")
    print(f"  Relative L2 Error v: {err_v:.5f}")
    print(f"  Relative L2 Error w: {err_w:.5f}")
    print(f"  Relative L2 Error Magnitude: {err_mag:.5f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        "epoch": CHECKPOINT_EPOCH,
        "error_u": err_u, "error_v": err_v, "error_w": err_w, "error_mag": err_mag
    }])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    # 6. Visualization 1: Export to VTK for Paraview
    vtk_out_path = os.path.join(RESULTS_DIR, "reconstruction_comparison.vtu")
    
    # We pass the original domain file so it copies the mesh topology
    domain_file = os.path.join(data_loader.DATA_DIR, data_loader.FILE_DOMAIN)
    
    save_vtk_results(
        domain_file, 
        vtk_out_path, 
        (u_pred, v_pred, w_pred, p_pred, u_true, v_true, w_true)
    )

    # 7. Visualization 2: Python Slice Plots and WSS over X

    # 1. Compute WSS
    wall_file = os.path.join(data_loader.DATA_DIR, data_loader.FILE_WALL)
    wss_x_coords, wss_values = compute_wall_shear_stress(net_u, net_v, net_w, wall_file)

    # 2. Run Plotting
    plot_results_extended(x_phys.flatten(), y_phys.flatten(), z_phys.flatten(), 
                        u_pred, v_pred, w_pred, 
                        wss_x_coords, wss_values)

    print("\nEvaluation Complete.")
    print(f"1. View 3D results in Paraview: {vtk_out_path}")
    print(f"2. View 2D slice plots and WSS over X: {RESULTS_DIR}")

if __name__ == "__main__":
    # Ensure user has updated the EXPERIMENT_DIR
    if not os.path.exists(EXPERIMENT_DIR):
        print(f"ERROR: Experiment directory not found: {EXPERIMENT_DIR}")
        print("Please update the 'EXPERIMENT_DIR' variable in evaluate.py to match your training output folder.")
    else:
        main()