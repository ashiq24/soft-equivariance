import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn

# RPP baseline EqCNN
def _repr_size(kind: str, r2_act):
    """Return channel multiplier for a given representation kind."""
    return 1 if kind == "trivial" else r2_act.fibergroup.order()


def _make_field(r2_act, channels: int, kind: str):
    """Create a FieldType with the desired representation."""
    if kind == "regular":
        rep = r2_act.regular_repr
    else:
        rep = r2_act.trivial_repr
    return enn.FieldType(r2_act, [rep] * channels)


class EquivariantBlock(nn.Module):
    """Two-layer equivariant block (no residual inside)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        r2_act: gspaces.GSpace,
        stride: int = 1,
        in_kind: str = "trivial",
        out_kind: str = "trivial",
    ):
        super().__init__()
        in_type = _make_field(r2_act, in_channels, in_kind)
        out_type = _make_field(r2_act, out_channels, out_kind)

        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = enn.ReLU(out_type, inplace=True)
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = enn.InnerBatchNorm(out_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = enn.GeometricTensor(x, self.conv1.in_type)
        y = self.conv1(gx)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return y.tensor  # return torch.Tensor


class NonEquivariantBlock(nn.Module):
    """Standard two-layer Conv-BN-ReLU-Conv-BN block (no residual inside)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        y = self.bn2(y)
        return y


class ResidualCnn(nn.Module):
    """
    ResNet-like CNN with separate equivariant and non-equivariant blocks.
    Residual mixing happens in the main forward:
        mixed = residual_strength * non_eq + (1 - residual_strength) * eq
        out = ReLU(identity + mixed)
    """

    def __init__(
        self,
        num_layers: int,
        residual_strength: float,
        num_classes: int,
        in_channels: int = 3,
        base_channels: int = 64,
        n_rotations: int = 8,
        group_type: str = "rotation",
        use_regular_representation: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert 0.0 <= residual_strength <= 1.0, "residual_strength must be in [0, 1]"
        self.residual_strength = residual_strength
        if group_type == "roto_reflection":
            self.r2_act = gspaces.FlipRot2dOnR2(N=n_rotations)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=n_rotations)
        self.use_regular_representation = use_regular_representation

        # Stem (both equivariant and non-equivariant versions)
        stem_out_kind = "regular" if use_regular_representation else "trivial"
        stem_rep_size = _repr_size(stem_out_kind, self.r2_act)

        self.eq_stem = nn.Sequential(
            enn.R2Conv(
                _make_field(self.r2_act, in_channels, "trivial"),
                _make_field(self.r2_act, base_channels, stem_out_kind),
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            enn.InnerBatchNorm(_make_field(self.r2_act, base_channels, stem_out_kind)),
            enn.ReLU(_make_field(self.r2_act, base_channels, stem_out_kind), inplace=True),
        )
        self.eq_stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ne_stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * stem_rep_size, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels * stem_rep_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        eq_blocks = []
        ne_blocks = []
        downsamples = []

        prev_kind = stem_out_kind
        prev_rep_size = stem_rep_size

        for i in range(num_layers):
            stride = 2 if i > 0 and i % 3 == 0 else 1

            # Last block: force trivial representation to match classifier width
            out_kind = "trivial" if i == num_layers - 1 else ("regular" if use_regular_representation else "trivial")
            out_rep_size = _repr_size(out_kind, self.r2_act)

            eq_blocks.append(
                EquivariantBlock(
                    base_channels,
                    base_channels,
                    self.r2_act,
                    stride=stride,
                    in_kind=prev_kind,
                    out_kind=out_kind,
                )
            )
            ne_blocks.append(
                NonEquivariantBlock(
                    base_channels * prev_rep_size,
                    base_channels * out_rep_size,
                    stride=stride,
                )
            )

            if stride != 1:
                downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(
                            base_channels * prev_rep_size,
                            base_channels * out_rep_size,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(base_channels * out_rep_size),
                    )
                )
            else:
                downsamples.append(
                    nn.Identity() if prev_rep_size == out_rep_size else nn.Conv2d(
                        base_channels * prev_rep_size,
                        base_channels * out_rep_size,
                        kernel_size=1,
                        bias=False,
                    )
                )

            prev_kind = out_kind
            prev_rep_size = out_rep_size

        self.eq_blocks = nn.ModuleList(eq_blocks)
        self.ne_blocks = nn.ModuleList(ne_blocks)
        self.downsamples = nn.ModuleList(
            [ds if ds is not None else nn.Identity() for ds in downsamples]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final representation is forced to trivial, so channel size is base_channels
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        eq = self.eq_stem(enn.GeometricTensor(x, _make_field(self.r2_act, x.shape[1], "trivial"))).tensor
        eq = self.eq_stem_pool(eq)

        ne = self.ne_stem(x)
        x = self.residual_strength * ne + (1.0 - self.residual_strength) * eq

        for eq_block, ne_block, down in zip(self.eq_blocks, self.ne_blocks, self.downsamples):
            identity = down(x) if not isinstance(down, nn.Identity) else x

            eq = eq_block(x)
            ne = ne_block(x)
            mixed = self.residual_strength * ne + (1.0 - self.residual_strength) * eq

            x = F.relu(identity + mixed, inplace=True)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x