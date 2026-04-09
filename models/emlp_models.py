"""EMLP models using PyTorch backend."""
import sys
import os

# Add external emlp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'equivariant-MLP'))
from emlp.nn.pytorch import EMLP, MLP, Standardize
from emlp.reps import Scalar, Vector, T
from emlp.groups import O, Lorentz
__all__ = ['EMLP', 'MLP', 'Standardize']


def create_emlp_o5_mlp(model_config):
    rep_in = 2 * Vector
    rep_out = Scalar
    tensor_rep = T(2)
    group = O(5)
    
    if model_config.get('use_non_eq', False):
        print("Using non-equivariant MLP model")
        model = MLP(rep_in, rep_out, group, ch=model_config['hidden_features'][0], num_layers=len(model_config['hidden_features']))
        return model
    
    if model_config['use_original_emlp']:
        print("Using original EMLP model")
        model = EMLP(rep_in,rep_out,group, only_linear=model_config['use_linear'], ch=model_config['hidden_features'], num_layers=len(model_config['hidden_features']), rpp=model_config['use_rpp'], rpp_factor=model_config['rpp_factor'])
    else:
        print("Using modified EMLP model with hidden reps as Vector x Vector")
        hidden_rep= [ch * Vector * Vector for ch in model_config['hidden_features']]
        model = EMLP(rep_in, rep_out, group, ch=hidden_rep, num_layers=len(hidden_rep), only_linear=model_config['use_linear'], rpp=model_config['use_rpp'], rpp_factor=model_config['rpp_factor'])
    
    return model


def create_emlp_lorentz_mlp(model_config):
    rep_in = 4 * Vector
    rep_out = Scalar
    group = Lorentz()

    if model_config.get('use_non_eq', False):
        print("Using non-equivariant MLP model")
        model = MLP(
            rep_in,
            rep_out,
            group,
            ch=model_config['hidden_features'][0],
            num_layers=len(model_config['hidden_features'])
        )
        return model

    if model_config.get('use_original_emlp', True):
        print("Using original EMLP model")
        model = EMLP(
            rep_in,
            rep_out,
            group,
            only_linear=model_config.get('use_linear', True),
            ch=model_config['hidden_features'],
            num_layers=len(model_config['hidden_features']),
            rpp=model_config.get('use_rpp', False),
            rpp_factor=model_config.get('rpp_factor', 0.5)
        )
    else:
        print("Using modified EMLP model with hidden reps as Vector x Vector")
        hidden_rep = [ch * Vector * Vector for ch in model_config['hidden_features']]
        model = EMLP(
            rep_in,
            rep_out,
            group,
            ch=hidden_rep,
            num_layers=len(hidden_rep),
            only_linear=model_config.get('use_linear', True),
            rpp=model_config.get('use_rpp', False),
            rpp_factor=model_config.get('rpp_factor', 0.5)
        )

    return model
    
