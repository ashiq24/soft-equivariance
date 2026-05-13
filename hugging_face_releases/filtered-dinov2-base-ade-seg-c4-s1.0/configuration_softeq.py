"""
Configuration class for Soft-Equivariant Vision Transformer models.

Covers all four model variants:
  - filtered_vit          : ViT for image classification (ImageNet)
  - filtered_dinov2       : DINOv2 for image classification (ImageNet)
  - filtered_vit_seg      : ViT for semantic segmentation (PASCAL VOC / ADE20K)
  - filtered_dino2_seg    : DINOv2 for semantic segmentation (PASCAL VOC / ADE20K)

Only model/architecture/filter parameters are stored. Training hyperparameters
(lr, epochs, batch_size, optimizer, scheduler, etc.) are intentionally excluded.
"""

from transformers import PretrainedConfig


class SoftEqConfig(PretrainedConfig):
    """
    Configuration for Soft-Equivariant filtered vision models.

    Args:
        model_arch (str): Architecture variant. One of:
            "filtered_vit", "filtered_dinov2",
            "filtered_vit_seg", "filtered_dino2_seg".
        pretrained_model (str): HuggingFace identifier of the backbone used to derive
            the architecture config (e.g. "google/vit-base-patch16-224").
        num_labels (int): Number of output classes.
        n_rotations (int): Size of the discrete rotation group (e.g. 4 for C4, 180 for C180).
        soft_thresholding (float): Softness of the patch-embedding filter in [0, 1].
            0 = strict equivariant projection; 1 = no projection (identity).
        soft_thresholding_pos (float): Softness of the positional-embedding filter in [0, 1].
        group_type (str): Symmetry group. "rotation" or "roto_reflection".
        decomposition_method (str): Basis decomposition algorithm. "schur" or "svd".
        filter_patch_embeddings (bool): Apply filter to the patch-embedding Conv2d.
        filter_attention_qkv (bool): Apply filter to Q/K/V projections (no-op for ViT/DINOv2).
        filter_attention_output (bool): Apply filter to attention output projections.
        filter_mlp (bool): Apply filter to MLP layers (no-op for ViT/DINOv2).
        attention_output_filter_list (list[int]): Transformer layer indices where
            attention output filtering is applied.
        soft_thresholding_attention_output (float): Softness for attention output filter.
        preserve_norm (bool): Preserve weight norms after projection.
        hard_mask (bool): Use a hard (step-function) mask instead of exponential damping.
        joint_decomposition (bool): Use joint decomposition for multi-generator groups.
        freeze_patch_embeddings (bool): Freeze the patch embedding projection after init.
        freeze_position_embeddings (bool): Freeze positional embeddings after init.
        ignore_index (int): Label index ignored by the segmentation loss (255 = boundary).
    """

    model_type = "soft_equivariant"

    def __init__(
        self,
        model_arch: str = "filtered_vit",
        pretrained_model: str = "google/vit-base-patch16-224",
        num_labels: int = 1000,
        n_rotations: int = 4,
        soft_thresholding: float = 0.0,
        soft_thresholding_pos: float = 0.0,
        group_type: str = "rotation",
        decomposition_method: str = "schur",
        filter_patch_embeddings: bool = True,
        filter_attention_qkv: bool = False,
        filter_attention_output: bool = False,
        filter_mlp: bool = False,
        attention_output_filter_list=None,
        soft_thresholding_attention_output: float = 0.1,
        preserve_norm: bool = False,
        hard_mask: bool = False,
        joint_decomposition: bool = True,
        freeze_patch_embeddings: bool = False,
        freeze_position_embeddings: bool = False,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_arch = model_arch
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.n_rotations = n_rotations
        self.soft_thresholding = soft_thresholding
        self.soft_thresholding_pos = soft_thresholding_pos
        self.group_type = group_type
        self.decomposition_method = decomposition_method
        self.filter_patch_embeddings = filter_patch_embeddings
        self.filter_attention_qkv = filter_attention_qkv
        self.filter_attention_output = filter_attention_output
        self.filter_mlp = filter_mlp
        self.attention_output_filter_list = attention_output_filter_list if attention_output_filter_list is not None else []
        self.soft_thresholding_attention_output = soft_thresholding_attention_output
        self.preserve_norm = preserve_norm
        self.hard_mask = hard_mask
        self.joint_decomposition = joint_decomposition
        self.freeze_patch_embeddings = freeze_patch_embeddings
        self.freeze_position_embeddings = freeze_position_embeddings
        self.ignore_index = ignore_index

    def _make_filter_config(self) -> dict:
        """Return a filter-config dict compatible with monkeypatching functions."""
        return {
            "filter_patch_embeddings": self.filter_patch_embeddings,
            "filter_attention_qkv": self.filter_attention_qkv,
            "filter_attention_output": self.filter_attention_output,
            "filter_mlp": self.filter_mlp,
            "group_type": self.group_type,
            "n_rotations": self.n_rotations,
            "soft_thresholding": self.soft_thresholding,
            "soft_thresholding_pos": self.soft_thresholding_pos,
            "decomposition_method": self.decomposition_method,
            "attention_output_filter_list": self.attention_output_filter_list,
            "soft_thresholding_attention_output": self.soft_thresholding_attention_output,
            "hard_mask": self.hard_mask,
            "preserve_norm": self.preserve_norm,
            "joint_decomposition": self.joint_decomposition,
            "freeze_patch_embeddings": self.freeze_patch_embeddings,
            "freeze_position_embeddings": self.freeze_position_embeddings,
        }
