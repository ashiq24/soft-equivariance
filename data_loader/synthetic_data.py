import numpy as np
import sys
import os
import pdb
import pandas as pd
from sympy import im

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'equivariant-MLP'))
try:
    from emlp.reps import Scalar, Vector, T
except Exception:
    Scalar = None
    Vector = None
    T = None
from emlp.groups import O, Lorentz
try:
    from emlp.datasets import GroupAugmentation
except Exception:
    GroupAugmentation = None

class Inertia:
    def __init__(self, N=1024, k=5, sigma=0.0):
        self.dim = (1 + 3) * k
        self.X = np.random.randn(N, self.dim)
        self.X[:, :k] = np.log(1 + np.exp(self.X[:, :k]))
        
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = np.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] * (r2 * I - ri[..., None] * ri[..., None, :])).sum(1)
        self.Y = inertia.reshape(-1, 9)
        
        if sigma > 0:
            X_norm = np.linalg.norm(self.X, axis=1, keepdims=True)
            X_noise = np.random.randn(*self.X.shape) * sigma * X_norm
            self.X = self.X + X_noise
            
            Y_norm = np.linalg.norm(self.Y, axis=1, keepdims=True)
            Y_noise = np.random.randn(*self.Y.shape) * sigma * Y_norm
            self.Y = self.Y + Y_noise
        
        if Scalar is not None and Vector is not None and T is not None:
            self.rep_in = k * Scalar + k * Vector
            self.rep_out = T(2)
        else:
            self.rep_in = None
            self.rep_out = None
        self.symmetry = O(3)
        
        Xmean = self.X.mean(0)
        Xmean[k:] = 0
        Xstd = np.zeros_like(Xmean)
        Xstd[:k] = np.abs(self.X[:, :k]).mean(0)
        Xstd[k:] = (np.abs(self.X[:, k:].reshape(N, k, 3)).mean((0, 2))[:, None] + np.zeros((k, 3))).reshape(k * 3)
        Ymean = 0 * self.Y.mean(0)
        Ystd = np.abs(self.Y - Ymean).mean((0, 1)) + np.zeros_like(Ymean)
        self.stats = (0, 1, 0, 1)
    
    def __getitem__(self, i):
        return (self.X[i], self.Y[i])
    
    def __len__(self):
        return self.X.shape[0]
    
    def default_aug(self, model, compute_equiv_error=False):
        if GroupAugmentation is None:
            raise RuntimeError("GroupAugmentation unavailable; cannot create augmentation wrapper.")
        if self.rep_in is None or self.rep_out is None:
            raise RuntimeError("EMLP representations unavailable; cannot create GroupAugmentation.")
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)


class O5Synthetic:
    def __init__(self, N=1024, sigma=0.0):
        '''
        sigma controls the amount of symmetry-breaking.
        '''
        d = 5
        self.dim = 2 * d
        self.X = np.random.randn(N, self.dim)
        ri = self.X.reshape(-1, 2, 5)
        r1, r2 = ri.transpose(1, 0, 2)
        self.Y = (np.sin(np.sqrt((r1**2).sum(-1))) - 
                  0.5 * np.sqrt((r2**2).sum(-1))**3 + 
                  (r1*r2).sum(-1) / (np.sqrt((r1**2).sum(-1)) * np.sqrt((r2**2).sum(-1))))
        
        if sigma > 0:
            numerator = np.abs(r1[:, 0]) + np.abs(r1[:, 1]) + np.abs(r2[:, 0]) + np.abs(r2[:, 1])
            denominator = np.abs(r1[:, 2]) + np.abs(r1[:, 3]) + np.abs(r2[:, 2]) + np.abs(r2[:, 3]) + 1
            self.Y = self.Y + sigma * (numerator / denominator)
        
        
        if Vector is not None and Scalar is not None:
            self.rep_in = 2 * Vector
            self.rep_out = Scalar
        else:
            self.rep_in = None
            self.rep_out = None
        self.symmetry = O(d)
        self.Y = self.Y[..., None]
        
        Xmean = self.X.mean(0)
        Xscale = (np.sqrt((self.X.reshape(N, 2, d)**2).mean((0, 2)))[:, None] + 0 * ri[0]).reshape(self.dim)
        self.stats = (0, Xscale, self.Y.mean(axis=0), self.Y.std(axis=0))
    
    def __getitem__(self, i):
        return (self.X[i], self.Y[i])
    
    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model, compute_equiv_error=False):
        if GroupAugmentation is None:
            raise RuntimeError("GroupAugmentation unavailable; cannot create augmentation wrapper.")
        if self.rep_in is None or self.rep_out is None:
            raise RuntimeError("EMLP representations unavailable; cannot create GroupAugmentation.")
        return GroupAugmentation(model,self.rep_in,self.rep_out,self.symmetry)


class ParticleInteraction:
    def __init__(self, N=1024, sigma=0.0):
        '''
        sigma controls the amount of symmetry-breaking. 
        '''
        
        self.dim = 4 * 4
        if Vector is not None and Scalar is not None:
            self.rep_in = 4 * Vector
            self.rep_out = Scalar
        else:
            self.rep_in = None
            self.rep_out = None
        self.X = np.random.randn(N, self.dim) / 4

        P = self.X.reshape(N, 4, 4)
        p1, p2, p3, p4 = P.transpose(1, 0, 2)
        eta = np.diag(np.array([1., -1., -1., -1.]))
        dot = lambda v1, v2: ((v1 @ eta) * v2).sum(-1)
        Le = (p1[:, :, None] * p3[:, None, :] - (dot(p1, p3) - dot(p1, p1))[:, None, None] * eta)
        Lmu = ((p2 @ eta)[:, :, None] * (p4 @ eta)[:, None, :] - (dot(p2, p4) - dot(p2, p2))[:, None, None] * eta)
        M = 4 * (Le * Lmu).sum(-1).sum(-1)
        if sigma != 0.0:
            vx2 = P[:, :, 1] ** 2
            vy2 = P[:, :, 2] ** 2
            vz2 = P[:, :, 3] ** 2
            ratio = vx2 / (vy2 + vz2 + 1)
            spatial_ratio_sum = ratio.sum(axis=1)
            M = M + sigma * spatial_ratio_sum  
        self.Y = M
        self.symmetry = Lorentz()
        self.Y = self.Y[..., None]

        self.Xscale = np.sqrt((np.abs((self.X.reshape(N, 4, 4) @ eta) * self.X.reshape(N, 4, 4)).mean(-1)).mean(0))
        self.Xscale = (self.Xscale[:, None] + np.zeros((4, 4))).reshape(-1)
        self.stats = (0, self.Xscale, self.Y.mean(axis=0), self.Y.std(axis=0))


    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model, compute_equiv_error=False):
        if GroupAugmentation is None:
            raise RuntimeError("GroupAugmentation unavailable; cannot create augmentation wrapper.")
        if self.rep_in is None or self.rep_out is None:
            raise RuntimeError("EMLP representations unavailable; cannot create GroupAugmentation.")
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


def compute_equivariance_error(model, x, in_rep, out_rep, group_element, group):
    import torch
    
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    
    with torch.no_grad():
        out = model(torch.FloatTensor(x_np[None])) if x_np.ndim == 1 else model(torch.FloatTensor(x_np))
    out_np = out.detach().cpu().numpy().squeeze()

    
    rep_in_concrete = in_rep(group)
    rho_in_g = np.array(rep_in_concrete.rho_dense(group_element))
    x_p = rho_in_g @ x_np
    
    with torch.no_grad():
        out_p = model(torch.FloatTensor(x_p[None])) if x_p.ndim == 1 else model(torch.FloatTensor(x_p))
    out_p_np = out_p.detach().cpu().numpy().squeeze()
    
    rep_out_concrete = out_rep(group)
    rho_out_g = np.array(rep_out_concrete.rho_dense(group_element))
    
    if out_np.ndim == 0:
        out_expected = rho_out_g[0, 0] * out_np
    else:
        out_expected = rho_out_g @ out_np
    
    if out_p_np.ndim == 0:
        error = np.linalg.norm(out_p_np - out_expected)
    else:
        error = np.linalg.norm(out_p_np - out_expected)
    
    return error
