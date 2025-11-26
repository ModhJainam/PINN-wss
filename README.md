# PINN-wss

**Uncovering near-wall blood flow from sparse data with physics-informed neural networks** [ArXiv Paper](https://arxiv.org/abs/2104.08249)

This repository contains codes and data to compute Wall Shear Stress (WSS) and reconstruct blood flow from sparse data measurements without full knowledge of boundary conditions, using Physics-Informed Neural Networks (PINNs).

---

## 🚀 Quick Start (New Modular Workflow)

We have modernized the codebase to use **`uv`** for fast, reproducible environment setup and modular scripts (`train.py`, `evaluate.py`) for better readability and extensibility.

### 1. Prerequisites
Install `uv` (an extremely fast Python package manager):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
````

### 2\. Installation

Clone the repo and sync the environment. This automatically installs Python 3.9, VTK, and other dependencies.

```bash
git clone https://github.com/ModhJainam/PINN-wss
cd PINN-wss

# Initialize the environment
uv sync

# IMPORTANT: Install PyTorch with CUDA support (Recommended for GPU training)
# Example for CUDA 12.6:
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3\. Configuration

Before running the scripts, ensure your data paths are correct:

1.  Open **`data_loader.py`**.
2.  Update the **`DATA_DIR`** variable to point to the folder containing your `.vtu` and `.vtk` source files.

### 4\. Usage

#### Train the Model

This script loads the data, initializes the PINN, and runs the training loop. It saves checkpoints and loss metrics to the `experiments/` folder.

```bash
uv run train.py
```

#### Evaluate & Visualize

After training, generate results (VTK files for ParaView and 2D validation plots).  
*Note: Open `evaluate.py` and update the `EXPERIMENT_DIR` variable to point to the specific run folder you want to analyze.*

```bash
uv run evaluate.py
```

**Outputs:**

  - **3D Flow Reconstruction:** `experiments/.../results/reconstruction_comparison.vtu` (Open in **ParaView**)
  - **2D Validation Plots:** WSS distribution graphs and velocity slice contours are saved in the same results folder.

-----

## 📂 Original Implementation Details

PyTorch codes are included for the different examples presented in the paper:

  * 1D advection-diffusion
  * 2D stenosis
  * 2D aneurysm
  * 3D aneurysm
  * Parameter identification (viscosity)

### Data

The input data for the 2D cases are provided in the `Data` folder. For the 3D model, a Google Drive link to the 3D files is typically included in the source code comments or supplementary materials.

### Visualization Note (Legacy)

Sample code is included for converting PyTorch output to VTK format (`Torch2VTK` folder).

  * **Important:** Input coordinates must be normalized before feeding into the network, then mapped back to physical coordinates for visualization.
  * The new `evaluate.py` script handles this normalization/denormalization automatically.

### Legacy Installation (Conda/Pip)

If you prefer the traditional setup without `uv`:

```bash
conda activate pytorch
pip install vtk
```