import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Import local modules
import data_loader
import model as pinn_model

# ==========================================
#   GLOBAL CONFIGURATION (MANUAL PARAMETERS)
# ==========================================

# 1. Training Hyperparameters
BATCH_SIZE    = 512
LEARNING_RATE = 5e-4
EPOCHS        = 8500        # Total training epochs
DECAY_RATE    = 0.5         # Learning rate decay factor
STEP_EPOCH    = 800         # Decay LR every N epochs

# 2. Loss Weights (Balancing the Multi-Objective Loss)
LAMBDA_DATA   = 20.0        # Weight for sparse measurement data
LAMBDA_BC     = 20.0        # Weight for wall boundary conditions
LAMBDA_PHYS   = 1.0         # Weight for Navier-Stokes residuals

# 3. Output & Versioning
OUTPUT_ROOT   = "../experiments"
SAVE_INTERVAL = 1000        # Save model checkpoints every N epochs

# 4. Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_output_dir():
    """Creates a timestamped directory for the current run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_metadata(run_dir, params):
    """Saves hyperparameters to a JSON file for reproducibility."""
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

def save_training_data_subset(run_dir, data_tensors):
    """Saves the specific sparse data points used for this training run."""
    xd, yd, zd, ud, vd, wd = data_tensors
    
    # Convert to numpy for saving
    df = pd.DataFrame({
        'x': xd.cpu().numpy().flatten(),
        'y': yd.cpu().numpy().flatten(),
        'z': zd.cpu().numpy().flatten(),
        'u': ud.cpu().numpy().flatten(),
        'v': vd.cpu().numpy().flatten(),
        'w': wd.cpu().numpy().flatten()
    })
    
    save_path = os.path.join(run_dir, 'training_data_subset.csv')
    df.to_csv(save_path, index=False)
    print(f"Training data subset saved to: {save_path}")

def plot_losses(run_dir, history):
    """Plots and saves loss curves."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(history['epoch'], history['loss_total'], label='Total Loss', linewidth=2)
    plt.semilogy(history['epoch'], history['loss_phys'], label='Physics Loss', linestyle='--')
    plt.semilogy(history['epoch'], history['loss_data'], label='Data Loss', linestyle='--')
    plt.semilogy(history['epoch'], history['loss_bc'], label='BC Loss', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('PINN Training Progress')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'), dpi=300)
    plt.close()

def main():
    print(f"--- Starting PINN Training on {DEVICE} ---")
    
    # 1. Setup Output Directory & Metadata
    run_dir = create_output_dir()
    params = {
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "decay_rate": DECAY_RATE,
        "step_epoch": STEP_EPOCH,
        "lambda_phys": LAMBDA_PHYS,
        "lambda_data": LAMBDA_DATA,
        "lambda_bc": LAMBDA_BC
    }
    save_metadata(run_dir, params)

    # 2. Load Data
    data = data_loader.get_training_data()
    
    # Unpack and Move to Device
    # Collocation points (Domain)
    x_c, y_c, z_c = [t.to(DEVICE) for t in data["collocation"]]
    
    # Boundary points (Wall)
    xb, yb, zb, ub, vb, wb = [t.to(DEVICE) for t in data["wall"]]
    
    # Boundary points (Inlet - used implicitly in data_loader but not always constrained directly in loss if sparse data covers it)
    # The original script does not enforce exact inlet BC if using sparse data, but we can if desired.
    # For now, we follow the original script logic: Loss_BC is mainly for Wall (No-Slip).
    
    # Sparse Measurement Data
    xd, yd, zd, ud, vd, wd = [t.to(DEVICE) for t in data["data"]]
    
    # Save the specific data subset used
    save_training_data_subset(run_dir, (xd, yd, zd, ud, vd, wd))

    # Create DataLoader for Collocation Points (Batching Domain Physics)
    dataset = TensorDataset(x_c, y_c, z_c)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Initialize Model
    net_u = pinn_model.Net_U().to(DEVICE); net_u.apply(pinn_model.init_weights)
    net_v = pinn_model.Net_V().to(DEVICE); net_v.apply(pinn_model.init_weights)
    net_w = pinn_model.Net_W().to(DEVICE); net_w.apply(pinn_model.init_weights)
    net_p = pinn_model.Net_P().to(DEVICE); net_p.apply(pinn_model.init_weights)

    # 4. Optimizers & Schedulers
    # Using Adam as per original paper/script
    opts = [
        optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), eps=1e-15)
        for net in [net_u, net_v, net_w, net_p]
    ]
    opt_u, opt_v, opt_w, opt_p = opts

    schedulers = [
        optim.lr_scheduler.StepLR(opt, step_size=STEP_EPOCH, gamma=DECAY_RATE)
        for opt in opts
    ]

    # Initialize Physics Loss Calculator
    loss_calculator = pinn_model.PINNLoss()

    # 5. Training Loop
    history = {'epoch': [], 'loss_total': [], 'loss_phys': [], 'loss_data': [], 'loss_bc': []}
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_phys = 0.
        epoch_bc   = 0.
        epoch_data = 0.
        batches    = 0

        for x_batch, y_batch, z_batch in dataloader:
            # Zero Gradients
            for opt in opts: opt.zero_grad()

            # --- Calculate Losses ---
            
            # 1. Physics Loss (on batched collocation points)
            l_phys = loss_calculator.physics_loss(x_batch, y_batch, z_batch, net_u, net_v, net_w, net_p)
            
            # 2. BC Loss (Wall No-Slip) - typically enforced on full boundary set per iteration 
            # (or you can random sample if memory is tight)
            l_bc = loss_calculator.bc_loss(xb, yb, zb, net_u, net_v, net_w)
            
            # 3. Data Loss (Sparse Measurements)
            l_data = loss_calculator.data_loss(xd, yd, zd, ud, vd, wd, net_u, net_v, net_w)

            # Weighted Sum
            loss = (LAMBDA_PHYS * l_phys) + (LAMBDA_BC * l_bc) + (LAMBDA_DATA * l_data)

            # Backprop
            loss.backward()
            
            # Step Optimizers
            for opt in opts: opt.step()

            # Accumulate for logging
            epoch_phys += l_phys.item()
            epoch_bc   += l_bc.item()
            epoch_data += l_data.item()
            batches += 1

        # Step Schedulers
        for sch in schedulers: sch.step()

        # Averaging
        avg_phys = epoch_phys / batches
        avg_bc   = epoch_bc / batches
        avg_data = epoch_data / batches
        avg_total = LAMBDA_PHYS * avg_phys + LAMBDA_BC * avg_bc + LAMBDA_DATA * avg_data

        # Logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Total: {avg_total:.6f} | "
                  f"Phys: {avg_phys:.6f} | BC: {avg_bc:.6f} | Data: {avg_data:.6f}")
            
            history['epoch'].append(epoch)
            history['loss_total'].append(avg_total)
            history['loss_phys'].append(avg_phys)
            history['loss_bc'].append(avg_bc)
            history['loss_data'].append(avg_data)

        # Save Checkpoints
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(run_dir, f"checkpoints_epoch_{epoch}")
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(net_u.state_dict(), os.path.join(ckpt_path, "net_u.pt"))
            torch.save(net_v.state_dict(), os.path.join(ckpt_path, "net_v.pt"))
            torch.save(net_w.state_dict(), os.path.join(ckpt_path, "net_w.pt"))
            torch.save(net_p.state_dict(), os.path.join(ckpt_path, "net_p.pt"))

    # Final Wrap Up
    duration = time.time() - start_time
    print(f"\nTraining Completed in {duration/60:.2f} minutes.")
    
    # Save History to CSV
    pd.DataFrame(history).to_csv(os.path.join(run_dir, 'loss_history.csv'), index=False)
    
    # Generate Plots
    plot_losses(run_dir, history)
    print(f"Outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()