import torch
import torch.nn as nn
from transformers.modeling_outputs import SemanticSegmenterOutput, ImageClassifierOutput
import pdb

class ClassificationCanonicalizationWrapper(nn.Module):
    """
    Wraps a classification model with canonicalization.
    
    On forward:
    1. Pass input through CanonicalizationNetwork to get fibre features.
    2. Canonicalize input using get_canonicalized_images().
    3. Pass canonicalized input to the classification model.
    
    The wrapper has the same interface as the original classification model.
    """
    
    def __init__(self, model, canon_net):
        """
        Args:
            model (nn.Module): The base classification model
            canon_net (CanonicalizationNetwork): The canonicalization network
        """
        super().__init__()
        self.model = model
        self.canon_net = canon_net
        # Copy config if available (for HuggingFace models)
        if hasattr(model, 'config'):
            self.config = model.config
        self.prior_loss = None
        
    def forward(self, pixel_values, *args, **kwargs):
        """
        Forward pass with canonicalization for classification.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W)
            *args, **kwargs: Additional arguments passed to the base model
            
        Returns:
            Model output (same as base model)
        """
        # Get fibre features from canonicalization network
        fibre_features = self.canon_net(pixel_values)
        # Compute prior loss if applicable assuming label is 0 
        self.prior_loss = 0.001 * nn.functional.cross_entropy(fibre_features, torch.zeros(fibre_features.size(0), dtype=torch.long, device=fibre_features.device))
        # Canonicalize images
        canon_images, angles, reflect_indicator = self.canon_net.get_canonicalized_images(
            pixel_values, fibre_features
        )
        
        # Pass canonicalized images through the base model
        output = self.model(canon_images, *args, **kwargs)
            
        return output


class SegmentationCanonicalizationWrapper(nn.Module):
    """
    Wraps a segmentation model with canonicalization.
    
    On forward:
    1. Pass input through CanonicalizationNetwork to get fibre features.
    2. Canonicalize input using get_canonicalized_images().
    3. Pass canonicalized input to the segmentation model.
    4. Revert the canonicalization on the output logits.
    5. If labels provided, compute loss on the reverted logits.
    
    The wrapper has the same interface as the original segmentation model.
    """
    
    def __init__(self, model, canon_net):
        """
        Args:
            model (nn.Module): The base segmentation model
            canon_net (CanonicalizationNetwork): The canonicalization network
        """
        super().__init__()
        self.model = model
        self.canon_net = canon_net
        # Copy config if available (for HuggingFace models)
        if hasattr(model, 'config'):
            self.config = model.config
        if hasattr(model, 'num_labels'):
            self.num_labels = model.num_labels
        if hasattr(model, 'ignore_index'):
            self.ignore_index = model.ignore_index
        if hasattr(model, 'loss_fct'):
            self.loss_fct = model.loss_fct
        else:
            # Default to CrossEntropyLoss
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        
    def forward(self, pixel_values, labels=None, *args, **kwargs):
        """
        Forward pass with canonicalization for segmentation.
        
        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W)
            labels (torch.Tensor, optional): Ground truth segmentation masks (B, H, W)
            *args, **kwargs: Additional arguments passed to the base model
            
        Returns:
            SemanticSegmenterOutput with loss computed on reverted logits
        """
        # Get fibre features from canonicalization network
        fibre_features = self.canon_net(pixel_values)
        
        # Canonicalize images
        canon_images, angles, reflect_indicator = self.canon_net.get_canonicalized_images(
            pixel_values, fibre_features
        )
        
        # Pass canonicalized images through the segmentation model WITHOUT labels
        # We'll compute loss after reverting the logits
        output = self.model(canon_images, labels=None, *args, **kwargs)
        
        # Extract logits from output
        if isinstance(output, SemanticSegmenterOutput):
            logits = output.logits
        elif isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Revert the canonicalization on logits
        reverted_logits = self.canon_net.apply_group_action(logits, angles, reflect_indicator)
        
        # cast the reverted logits to the shape of the labels if labels are provided
        # in case of Vit and DINO it is already done, but for SegFormer we need to do it here
        reverted_logits = nn.functional.interpolate(
            reverted_logits,
            size=labels.shape[-2:] if labels is not None else pixel_values.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Compute loss on reverted logits if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fct(reverted_logits, labels)
        
        
        # Return output in the same format as the base model
        if isinstance(output, SemanticSegmenterOutput):
            return SemanticSegmenterOutput(
                loss=loss,
                logits=reverted_logits,
                hidden_states=output.hidden_states if hasattr(output, 'hidden_states') else None,
                attentions=output.attentions if hasattr(output, 'attentions') else None,
            )
        elif isinstance(output, dict):
            return {
                'loss': loss,
                'logits': reverted_logits,
                **{k: v for k, v in output.items() if k not in ['loss', 'logits']}
            }
        else:
            if loss is not None:
                return loss, reverted_logits
            return reverted_logits
