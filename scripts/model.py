import torch
import torch.nn as nn
import torch.autograd as autograd

# ==========================================
#   ACTIVATION FUNCTION
# ==========================================

class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    """
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

# ==========================================
#   NEURAL NETWORK ARCHITECTURES
# ==========================================

# Default parameters matched to original script
H_N = 200        # Hidden layer width
INPUT_N = 3      # Inputs: x, y, z
LAYERS = 8       # Depth of hidden layers

class BaseNet(nn.Module):
    def __init__(self, h_n=H_N, input_n=INPUT_N):
        super(BaseNet, self).__init__()
        
        # Build network layers dynamically
        layers = []
        # Input layer
        layers.append(nn.Linear(input_n, h_n))
        layers.append(Swish())
        
        # Hidden layers
        for _ in range(LAYERS - 2): 
            layers.append(nn.Linear(h_n, h_n))
            layers.append(Swish())
            
        # Output layer
        layers.append(nn.Linear(h_n, 1))
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Individual classes for U, V, W, P to allow distinct handling if needed
class Net_U(BaseNet): pass
class Net_V(BaseNet): pass
class Net_W(BaseNet): pass
class Net_P(BaseNet): pass

def init_weights(m):
    """Kaiming Normal initialization for Linear layers"""
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

# ==========================================
#   PHYSICS-INFORMED LOSS FUNCTIONS
# ==========================================

class PINNLoss:
    def __init__(self, diff=0.00125, rho=1.0, 
                 x_scale=3.0, yz_scale=2.0, u_scale=1.0):
        """
        Initializes the loss calculator with physical constants and normalization scales.
        """
        self.diff = diff
        self.rho = rho
        
        # Normalization factors for Chain Rule scaling
        self.x_scale = x_scale
        self.yz_scale = yz_scale
        self.u_scale = u_scale
        
        # Pre-calculated scales for derivatives
        self.xx_scale = u_scale * (x_scale**2)
        self.yy_scale = u_scale * (yz_scale**2)
        self.uu_scale = u_scale**2

    def get_gradients(self, u, x, order=1):
        """
        Helper to calculate automatic gradients.
        order=1: returns du/dx
        order=2: returns d2u/dx2
        """
        grads = torch.ones_like(u)
        du_dx = autograd.grad(u, x, grad_outputs=grads, create_graph=True, only_inputs=True)[0]
        if order == 2:
            d2u_dx2 = autograd.grad(du_dx, x, grad_outputs=grads, create_graph=True, only_inputs=True)[0]
            return d2u_dx2
        return du_dx

    def physics_loss(self, x, y, z, net_u, net_v, net_w, net_p):
        """
        Calculates the residuals of the Navier-Stokes equations (Momentum + Continuity).
        """
        # Enable gradients for physics calculations
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        # Concatenate inputs and forward pass
        net_in = torch.cat((x, y, z), 1)
        
        u = net_u(net_in)
        v = net_v(net_in)
        w = net_w(net_in)
        p = net_p(net_in)

        # ----------------------------------------
        # Automatic Differentiation
        # ----------------------------------------
        # Velocity u derivatives
        u_x = self.get_gradients(u, x, 1); u_xx = self.get_gradients(u, x, 2)
        u_y = self.get_gradients(u, y, 1); u_yy = self.get_gradients(u, y, 2)
        u_z = self.get_gradients(u, z, 1); u_zz = self.get_gradients(u, z, 2)

        # Velocity v derivatives
        v_x = self.get_gradients(v, x, 1); v_xx = self.get_gradients(v, x, 2)
        v_y = self.get_gradients(v, y, 1); v_yy = self.get_gradients(v, y, 2)
        v_z = self.get_gradients(v, z, 1); v_zz = self.get_gradients(v, z, 2)

        # Velocity w derivatives
        w_x = self.get_gradients(w, x, 1); w_xx = self.get_gradients(w, x, 2)
        w_y = self.get_gradients(w, y, 1); w_yy = self.get_gradients(w, y, 2)
        w_z = self.get_gradients(w, z, 1); w_zz = self.get_gradients(w, z, 2)

        # Pressure derivatives
        p_x = self.get_gradients(p, x, 1)
        p_y = self.get_gradients(p, y, 1)
        p_z = self.get_gradients(p, z, 1)

        # ----------------------------------------
        # Navier-Stokes Residuals
        # ----------------------------------------
        # X-Momentum
        res_u = (u*u_x / self.x_scale + v*u_y / self.yz_scale + w*u_z / self.yz_scale) \
                - self.diff * (u_xx/self.xx_scale + u_yy/self.yy_scale + u_zz/self.yy_scale) \
                + (1/self.rho) * (p_x / (self.x_scale * self.uu_scale))

        # Y-Momentum
        res_v = (u*v_x / self.x_scale + v*v_y / self.yz_scale + w*v_z / self.yz_scale) \
                - self.diff * (v_xx/self.xx_scale + v_yy/self.yy_scale + v_zz/self.yy_scale) \
                + (1/self.rho) * (p_y / (self.yz_scale * self.uu_scale))

        # Z-Momentum
        res_w = (u*w_x / self.x_scale + v*w_y / self.yz_scale + w*w_z / self.yz_scale) \
                - self.diff * (w_xx/self.xx_scale + w_yy/self.yy_scale + w_zz/self.yy_scale) \
                + (1/self.rho) * (p_z / (self.yz_scale * self.uu_scale))

        # Continuity Equation
        res_c = (u_x / self.x_scale) + (v_y / self.yz_scale) + (w_z / self.yz_scale)

        # Total Mean Squared Error of Residuals
        mse = nn.MSELoss()
        loss_total = mse(res_u, torch.zeros_like(res_u)) + \
                     mse(res_v, torch.zeros_like(res_v)) + \
                     mse(res_w, torch.zeros_like(res_w)) + \
                     mse(res_c, torch.zeros_like(res_c))
        
        return loss_total

    def data_loss(self, xd, yd, zd, ud, vd, wd, net_u, net_v, net_w):
        """
        Calculates loss between predictions and sparse measurement data.
        """
        net_in = torch.cat((xd, yd, zd), 1)
        
        # Predictions
        u_pred = net_u(net_in)
        v_pred = net_v(net_in)
        w_pred = net_w(net_in)
        
        mse = nn.MSELoss()
        loss_d = mse(u_pred, ud) + mse(v_pred, vd) + mse(w_pred, wd)
        return loss_d

    def bc_loss(self, xb, yb, zb, net_u, net_v, net_w):
        """
        Calculates Wall Boundary Condition loss (No-Slip).
        Targets are implicitly zero.
        """
        net_in = torch.cat((xb, yb, zb), 1)
        
        u_b = net_u(net_in)
        v_b = net_v(net_in)
        w_b = net_w(net_in)
        
        mse = nn.MSELoss()
        loss_bc = mse(u_b, torch.zeros_like(u_b)) + \
                  mse(v_b, torch.zeros_like(v_b)) + \
                  mse(w_b, torch.zeros_like(w_b))
        
        return loss_bc