import torch
import torch.nn as nn
import torch.nn.functional as F

from softeq.equi_utils.rotation_filters import get_equivariant_filter_rotation
from softeq.equi_utils.reflection_filters import get_equivariant_filter_reflection
from softeq.equi_utils.roto_reflection_filters import get_equivariant_filter_roto_reflection
from softeq.layers.elinear import ELinear
from softeq.layers.flinear import FLinear
from softeq.equi_utils.equi_constraints import DiscreteRotationConstraints
from softeq.equi_utils.rotation_filters import get_invariant_filter_rotation

class TestModel(nn.Module):
    def __init__(
        self,
        nlayers: int,
        input_size: tuple,
        feature_size: list,
        n_rotations: int,
        nclasses: int,
        soft_thresholding: float = 0.0,
        decomposition_method: str = "svd",
        enforce_type: str = "filter",
    ):
        super().__init__()
        self.nlayers = nlayers

        if enforce_type == 'filter':
            filter = get_invariant_filter_rotation(
                n_rotations=n_rotations,
                input_size=input_size,
                soft_threshold=soft_thresholding,
                decomposition_method=decomposition_method,
                debug=False
            )
            self.inv_layer = FLinear(
                in_features=(input_size[0] * input_size[1] * input_size[2]),
                out_features=feature_size[0],
                filter=filter,
            )
        else:
            rotation_constraints = DiscreteRotationConstraints(
            n_rotations=n_rotations,
            input_size= input_size,
            output_size= None,
            decomposition_method=decomposition_method
            )
            rotation_constraints.create_invariant_basis()
            basis, values = rotation_constraints.get_invariant_basis(soft_thresholding=soft_thresholding)
            self.inv_layer = ELinear(
                in_features=basis.shape[0],
                out_features=feature_size[0],
                basis=basis,
            )

        self.layers = nn.ModuleList(
            [nn.Linear(feature_size[i], feature_size[i + 1]) for i in range(nlayers - 1)]
        )

        self.fc = nn.Linear(feature_size[-1], nclasses)

    def forward(self, x):
        x = self.inv_layer(x)
        x = F.relu(x)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        x = self.fc(x)
        return x


class TestEqModel(nn.Module):
    def __init__(
        self,
        nlayers: int,
        input_size: int,
        feature_size: list,
        n_rotations: int,
        nclasses: int,
        soft_thresholding: float = 0.0,
        soft_thresholding_reflection: float = None,
        soft_thresholding_rotation: float = None,
        decomposition_method: str = "svd",
        enforce_type: str = "filter",
        hard_mask: bool = True,
        no_head: bool = False,
        reflection: bool = False,
    ):
        super().__init__()

        self.nlayers = nlayers
        self.reflection = reflection
        assert len(feature_size) == nlayers, (
            "feature_size length must be equal to nlayers: {}, {}".format(
                len(feature_size), nlayers
            )
        )
        
        # Determine symmetry type
        use_reflection_only = reflection and (n_rotations is None)
        use_roto_reflection = reflection and (n_rotations is not None)
        use_rotation_only = not reflection and (n_rotations is not None)
        
        if n_rotations is not None and n_rotations > 4 and not soft_thresholding == 0:
            soft_thresholding = 0.4 + 0.6 * soft_thresholding
        layer = []
        feature_size = [input_size] + feature_size

        for i in range(nlayers):
            print("feature_size[i]:", feature_size[i])
            print("feature_size[i+1]:", feature_size[i + 1])

            if enforce_type == 'explicit':
                if use_reflection_only:
                    raise NotImplementedError("Explicit basis for reflection not yet implemented. Use enforce_type='filter'.")
                basis = get_basis(
                    n_rotations,
                    feature_size,
                    soft_thresholding,
                    decomposition_method,
                    i,
                )

                layer.append(
                    ELinear(
                        in_features=1 * feature_size[i][1] * feature_size[i][2],
                        out_features=1
                        * feature_size[i + 1][1]
                        * feature_size[i + 1][2],
                        basis=basis,
                        enforce_equivariance=True,
                        in_channels=feature_size[i][0],
                        out_channels=feature_size[i + 1][0],
                    )
                )
            elif enforce_type == 'filter':
                if use_reflection_only:
                    # Use reflection filters
                    filter = get_equivariant_filter_reflection(
                        input_size=(1, feature_size[i][1], feature_size[i][2]),
                        output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                        soft_threshold=soft_thresholding,
                        debug=False,
                    )
                elif use_roto_reflection:
                    # Use roto-reflection filters
                    filter = get_equivariant_filter_roto_reflection(
                        n_rotations=n_rotations,
                        input_size=(1, feature_size[i][1], feature_size[i][2]),
                        output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                        soft_threshold=soft_thresholding,
                        soft_threshold_reflection=soft_thresholding_reflection,
                        soft_threshold_rotation=soft_thresholding_rotation,
                        apply_soft_mask=False,
                        debug=False,
                    )
                elif use_rotation_only:
                    # Use rotation filters (original behavior)
                    filter = get_equivariant_filter_rotation(
                        n_rotations,
                        input_size=(1, feature_size[i][1], feature_size[i][2]),
                        output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                        soft_threshold=soft_thresholding,
                        apply_soft_mask=False,
                    )
                else:
                    raise ValueError(f"Invalid symmetry configuration: reflection={reflection}, n_rotations={n_rotations}")
                
                layer.append(
                    FLinear(
                        in_features=1 * feature_size[i][1] * feature_size[i][2],
                        out_features=1
                        * feature_size[i + 1][1]
                        * feature_size[i + 1][2],
                        filter=filter,
                        enforce_equivariance=True,
                        in_channels=feature_size[i][0],
                        out_channels=feature_size[i + 1][0],
                    )
                )
            else:
                raise ValueError("enforce_type must be either 'explicit' or 'filter'")


        self.last_feature_size = feature_size[-1]
        self.backbone = nn.ModuleList(layer)
        if no_head:
            self.cls_head = nn.Identity()
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(feature_size[-1][0], nclasses),
            )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        for layer in self.backbone:
            x = layer(x)
            x = F.relu(x)

        # reshape x to (batch_size, channels, height, width)
        x = x.reshape(
            -1,
            self.last_feature_size[0],
            self.last_feature_size[1],
            self.last_feature_size[2],
        )

        # global max pooling over H and W -> returns (batch, channels)
        x = x.amax(dim=(2, 3))
        x = self.cls_head(x)
        return x



# residual connection 

class ResidualSoftEqBlock(nn.Module):
    def __init__(self, eq_block: nn.Module, non_eq_block: nn.Module, softness: float):
        super(ResidualSoftEqBlock, self).__init__()
        self.eq_block = eq_block
        self.non_eq_block = non_eq_block
        self.softness = softness
        assert 0.0 <= softness <= 1.0, "Softness must be between 0 and 1"

    def forward(self, x):
        out_eq = self.eq_block(x)
        out_non_eq = self.non_eq_block(x)
        out = self.softness * out_non_eq + (1 - self.softness) * out_eq 
        return out

class TestEqModelResidual(nn.Module):
    def __init__(
        self,
        nlayers: int,
        input_size: int,
        feature_size: list,
        n_rotations: int,
        nclasses: int,
        soft_thresholding: float = 0.0,
        soft_thresholding_reflection: float = None,
        soft_thresholding_rotation: float = None,
        decomposition_method: str = "svd",
        no_head: bool = False,
        reflection: bool = False,
    ):
        super().__init__()

        self.nlayers = nlayers
        self.reflection = reflection
        assert len(feature_size) == nlayers, (
            "feature_size length must be equal to nlayers: {}, {}".format(
                len(feature_size), nlayers
            )
        )

        # Determine symmetry type
        use_reflection_only = reflection and (n_rotations is None)
        use_roto_reflection = reflection and (n_rotations is not None)
        use_rotation_only = not reflection and (n_rotations is not None)

        layer = []
        feature_size = [input_size] + feature_size

        for i in range(nlayers):
            print("feature_size[i]:", feature_size[i])
            print("feature_size[i+1]:", feature_size[i + 1])

            if use_reflection_only:
                # Use reflection filters
                filter_1 = get_equivariant_filter_reflection(
                    input_size=(1, feature_size[i][1], feature_size[i][2]),
                    output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                    soft_threshold=0.0,
                    debug=False,
                )
                filter_2 = get_equivariant_filter_reflection(
                    input_size=(1, feature_size[i][1], feature_size[i][2]),
                    output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                    soft_threshold=1.0,
                    debug=False,
                )
            elif use_roto_reflection:
                # Use roto-reflection filters
                # Filter 1: Equivariant (use separate softness values if provided, otherwise 0.0)
                soft_ref_1 = soft_thresholding_reflection if soft_thresholding_reflection is not None else 0.0
                soft_rot_1 = soft_thresholding_rotation if soft_thresholding_rotation is not None else 0.0
                filter_1 = get_equivariant_filter_roto_reflection(
                    n_rotations=n_rotations,
                    input_size=(1, feature_size[i][1], feature_size[i][2]),
                    output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                    soft_threshold=0.0,
                    soft_threshold_reflection=soft_ref_1,
                    soft_threshold_rotation=soft_rot_1,
                    apply_soft_mask=False,
                    debug=False,
                )
                # Filter 2: Non-equivariant (soft_threshold=1.0 for both)
                filter_2 = get_equivariant_filter_roto_reflection(
                    n_rotations=n_rotations,
                    input_size=(1, feature_size[i][1], feature_size[i][2]),
                    output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                    soft_threshold=1.0,
                    apply_soft_mask=False,
                    debug=False,
                )
            elif use_rotation_only:
                # Use rotation filters (original behavior)
                filter_1 = get_equivariant_filter_rotation(
                    n_rotations,
                    input_size=(1, feature_size[i][1], feature_size[i][2]),
                    output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                    soft_threshold=0.0,
                    apply_soft_mask=False,
                )
                filter_2 = get_equivariant_filter_rotation(
                    n_rotations,
                    input_size=(1, feature_size[i][1], feature_size[i][2]),
                    output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
                    soft_threshold=1.0,
                    apply_soft_mask=False,
                )
            else:
                raise ValueError(f"Invalid symmetry configuration: reflection={reflection}, n_rotations={n_rotations}")
            layer.append(
                    ResidualSoftEqBlock(
                        eq_block=FLinear(
                        in_features=1 * feature_size[i][1] * feature_size[i][2],
                        out_features=1
                        * feature_size[i + 1][1]
                        * feature_size[i + 1][2],
                        filter=filter_1,
                        enforce_equivariance=True,
                        in_channels=feature_size[i][0],
                        out_channels=feature_size[i + 1][0],
                    ),
                        non_eq_block=FLinear(
                        in_features=1 * feature_size[i][1] * feature_size[i][2],
                        out_features=1
                        * feature_size[i + 1][1]
                        * feature_size[i + 1][2],
                        filter=filter_2,
                        enforce_equivariance=True,
                        in_channels=feature_size[i][0],
                        out_channels=feature_size[i + 1][0],
                    ),
                        softness=soft_thresholding
                )
            )

        self.last_feature_size = feature_size[-1]
        self.backbone = nn.ModuleList(layer)
        if no_head:
            self.cls_head = nn.Identity()
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(feature_size[-1][0], nclasses),
            )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        for layer in self.backbone:
            x = layer(x)
            x = F.relu(x)

        # reshape x to (batch_size, channels, height, width)
        x = x.reshape(
            -1,
            self.last_feature_size[0],
            self.last_feature_size[1],
            self.last_feature_size[2],
        )

        # global max pooling over H and W -> returns (batch, channels)
        x = x.amax(dim=(2, 3))
        x = self.cls_head(x)
        return x

def get_basis(n_rotations, feature_size, soft_thresholding, decomposition_method, i):
    cons = DiscreteRotationConstraints(
            n_rotations=n_rotations,
            input_size=(1, feature_size[i][1], feature_size[i][2]),
            output_size=(1, feature_size[i + 1][1], feature_size[i + 1][2]),
            decomposition_method=decomposition_method,
            )

    cons.create_equivariant_basis()

    basis, _ = cons.get_equivariant_basis(
        soft_thresholding=soft_thresholding
    )

    basis = basis.clone().detach()
    return basis