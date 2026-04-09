"""
Filtered layer wrappers that apply filters to pretrained weights.

This module re-exports all filtered layer functions for ViT and DINOv2.
For new code, import directly from:
- models.filtered_layers_vit for ViT functions
- models.filtered_layers_dinov2 for DINOv2 functions
"""

# Re-export ViT functions
from .filtered_layers_vit import (
    custom_vitembeddings_forward,
    monkeypatch_vitembeddings,
    custom_vitoutput_forward,
    monkeypatch_vitoutput,
)

# Re-export DINOv2 functions
from .filtered_layers_dinov2 import (
    custom_dinov2embeddings_forward,
    custom_dinov2_with_registers_embeddings_forward,
    monkeypatch_dinov2embeddings,
)

__all__ = [
    # ViT
    'custom_vitembeddings_forward',
    'monkeypatch_vitembeddings',
    'custom_vitoutput_forward',
    'monkeypatch_vitoutput',
    # DINOv2
    'custom_dinov2embeddings_forward',
    'custom_dinov2_with_registers_embeddings_forward',
    'monkeypatch_dinov2embeddings',
]
