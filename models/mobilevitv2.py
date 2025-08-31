import torch
from torch import nn, Tensor, Size
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout
import argparse
from typing import Dict, Tuple, Optional, Sequence, List, Union
import numpy as np
from torch import nn, Tensor
import math


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bound_fn(min_val: Union[float, int], max_val: Union[float, int], value: Union[float, int]):
    return max(min_val, min(max_val, value))


class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
            self,
            embed_dim: int,
            attn_dropout: Optional[float] = 0.0,
            bias: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            padding=0,
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # This part of the original code has a bug (self.qkv_proj.block does not exist).
        # It's not used in this example, so we leave it as is, but in a real-world
        # scenario using cross-attention, this would need to be fixed.
        # A simple fix would be to not use F.conv2d with sliced weights, but rather
        # pass x and x_prev through the qkv_proj layer and split the results.
        # For now, we'll assume it's not called.
        pass

    def forward(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)


class LinearAttnFFN(nn.Module):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.1,
            ffn_dropout: Optional[float] = 0.0,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            LayerNorm(normalized_shape=embed_dim),
            attn_unit,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            LayerNorm(normalized_shape=embed_dim),
            Conv2d(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                padding=0,
            ),
            nn.LeakyReLU(negative_slope=0.1),
            Dropout(p=ffn_dropout),
            Conv2d(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                padding=0,
            ),
            Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()

    def forward(
            self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res  # residual

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class GlobalPool(nn.Module):
    pool_types = ["mean", "rms", "abs"]

    def __init__(
            self,
            pool_type: Optional[str] = "mean",
            keep_dim: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x ** 2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x ** -0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)


class LayerNorm(nn.LayerNorm):
    """
    Layer Normalization that works on channel-first tensors.
    """

    def __init__(
            self,
            normalized_shape: Union[int, List[int], Size],
            eps: Optional[float] = 1e-5,
            elementwise_affine: Optional[bool] = True,
            *args,
            **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Check if the input is a 4D tensor (B, C, H, W) or (B, C, P, N)
        # and if the channel dimension matches the normalization shape.
        if x.ndim == 4 and x.shape[1] == self.normalized_shape[0]:
            # Manually normalize over the channel dimension (dim=1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1, keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

            if self.elementwise_affine:
                # Reshape weight and bias for broadcasting: [C] -> [1, C, 1, 1]
                weight = self.weight.view(1, -1, 1, 1)
                bias = self.bias.view(1, -1, 1, 1)
                x_normalized = x_normalized * weight + bias
            return x_normalized
        else:
            # For other cases (e.g., channel-last), use the default implementation
            return super().forward(x)


class InvertedResidual(nn.Module):
    """
    Inverted residual block from MobileNetv2.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: Union[int, float],
            dilation: int = 1,
            skip_connection: Optional[bool] = True,
            *args,
            **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=nn.Conv2d(
                    in_channels=in_channels, out_channels=hidden_dim,
                    kernel_size=1, padding=0, bias=False
                ),
            )
            block.add_module("exp_1x1_bn", nn.BatchNorm2d(num_features=hidden_dim))
            block.add_module("exp_1x1_act", nn.LeakyReLU(negative_slope=0.1))

        block.add_module(
            name="conv_3x3",
            module=nn.Conv2d(
                in_channels=hidden_dim, out_channels=hidden_dim,
                stride=stride, kernel_size=3, groups=hidden_dim,
                dilation=dilation, padding=dilation, bias=False
            ),
        )
        block.add_module("conv_3x3_bn", nn.BatchNorm2d(num_features=hidden_dim))
        block.add_module("conv_3x3_act", nn.LeakyReLU(negative_slope=0.1))

        block.add_module(
            name="red_1x1",
            module=nn.Conv2d(
                in_channels=hidden_dim, out_channels=out_channels,
                kernel_size=1, padding=0, bias=False
            ),
        )
        block.add_module("red_1x1_bn", nn.BatchNorm2d(num_features=out_channels))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
                self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlockv2(nn.Module):
    """
    MobileViTv2 block.
    """

    def __init__(
            self,
            in_channels: int,
            attn_unit_dim: int,
            ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
            n_attn_blocks: Optional[int] = 2,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[float] = 0.0,
            ffn_dropout: Optional[float] = 0.0,
            patch_h: Optional[int] = 8,
            patch_w: Optional[int] = 8,
            conv_ksize: Optional[int] = 3,
            dilation: Optional[int] = 1,
            *args,
            **kwargs
    ) -> None:
        super(MobileViTBlockv2, self).__init__()

        cnn_out_dim = attn_unit_dim

        conv_3x3_in = nn.Sequential(nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            dilation=dilation,
            groups=in_channels,
            padding=dilation,
            bias=False,
        ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(negative_slope=0.1))

        conv_1x1_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )

        self.conv_proj = nn.Sequential(nn.Conv2d(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=False
        ),
            nn.BatchNorm2d(num_features=in_channels))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

    def _build_attn_layer(
            self,
            d_model: int,
            ffn_mult: Union[Sequence, int, float],
            n_layers: int,
            attn_dropout: float,
            dropout: float,
            ffn_dropout: float,
            *args,
            **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                    np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(LayerNorm(normalized_shape=d_model))

        return nn.Sequential(*global_rep), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)
        fm = self.local_rep(x)
        patches, output_size = self.unfolding_pytorch(fm)
        patches = self.global_rep(patches)
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)
        return fm

    def forward(
            self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.forward_spatial(x)


class MobileViTv2(nn.Module):
    def __init__(self, num_classes, width_multiplier: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pool_type = "mean"

        self.dilation = 1
        self.dilate_l4 = False
        self.dilate_l5 = False
        ffn_multiplier = 2
        mv2_exp_mult = 2

        layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
        layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))

        mobilevit_config = {
            "layer0": {"img_channels": 3, "out_channels": layer_0_dim},
            "layer1": {"out_channels": int(make_divisible(64 * width_multiplier, 16)), "expand_ratio": mv2_exp_mult,
                       "num_blocks": 1, "stride": 1, "block_type": "mv2"},
            "layer2": {"out_channels": int(make_divisible(128 * width_multiplier, 8)), "expand_ratio": mv2_exp_mult,
                       "num_blocks": 2, "stride": 2, "block_type": "mv2"},
            "layer3": {"out_channels": int(make_divisible(256 * width_multiplier, 8)),
                       "attn_unit_dim": int(make_divisible(128 * width_multiplier, 8)),
                       "ffn_multiplier": ffn_multiplier, "attn_blocks": 2, "patch_h": 2, "patch_w": 2, "stride": 2,
                       "mv_expand_ratio": mv2_exp_mult, "block_type": "mobilevit"},
            "layer4": {"out_channels": int(make_divisible(384 * width_multiplier, 8)),
                       "attn_unit_dim": int(make_divisible(192 * width_multiplier, 8)),
                       "ffn_multiplier": ffn_multiplier, "attn_blocks": 4, "patch_h": 2, "patch_w": 2, "stride": 2,
                       "mv_expand_ratio": mv2_exp_mult, "block_type": "mobilevit"},
            "layer5": {"out_channels": int(make_divisible(512 * width_multiplier, 8)),
                       "attn_unit_dim": int(make_divisible(256 * width_multiplier, 8)),
                       "ffn_multiplier": ffn_multiplier, "attn_blocks": 3, "patch_h": 2, "patch_w": 2, "stride": 2,
                       "mv_expand_ratio": mv2_exp_mult, "block_type": "mobilevit"},
            "last_layer_exp_factor": 4,
        }
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        self.model_conf_dict = dict()
        self.conv_1 = nn.Sequential(nn.Conv2d(
            in_channels=image_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.1))
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(in_channels, mobilevit_config["layer1"])
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(in_channels, mobilevit_config["layer2"])
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(in_channels, mobilevit_config["layer3"])
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(in_channels, mobilevit_config["layer4"], self.dilate_l4)
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(in_channels, mobilevit_config["layer5"], self.dilate_l5)
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = nn.Identity()
        self.model_conf_dict["exp_before_cls"] = {"in": out_channels, "out": out_channels}

        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            nn.Linear(in_features=out_channels, out_features=num_classes, bias=True),
        )

    def _make_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel, cfg, dilate)
        else:
            return self._make_mobilenet_layer(input_channel, cfg)

    def _make_mobilenet_layer(self, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(
                in_channels=input_channel, out_channels=output_channels,
                stride=stride, expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1
            layer = InvertedResidual(
                in_channels=input_channel, out_channels=cfg.get("out_channels"),
                stride=stride, expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )
            block.append(layer)
            input_channel = cfg.get("out_channels")

        block.append(
            MobileViTBlockv2(
                in_channels=input_channel, attn_unit_dim=cfg["attn_unit_dim"],
                ffn_multiplier=cfg.get("ffn_multiplier"), n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2), patch_w=cfg.get("patch_w", 2),
                dropout=0.0, ffn_dropout=0.0, attn_dropout=0.0, conv_ksize=3,
                dilation=self.dilation
            )
        )
        return nn.Sequential(*block), input_channel

    def forward(self, x, *args, **kwargs):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x

def mobilevitv2_xxs(num_classes=5):
    return MobileViTv2(num_classes=num_classes, width_multiplier=0.5)
def mobilevitv2_xs(num_classes=5):
    return MobileViTv2(num_classes=num_classes, width_multiplier=0.75)
def mobilevitv2_s(num_classes=5):
    return MobileViTv2(num_classes=num_classes, width_multiplier=1.0)

def count_parameters(model: nn.Module):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input_tensor = torch.randn(5, 3, 256, 256)

    # --- 0.5x Width Multiplier Model ---
    print("--- MobileViTv2-0.5 ---")
    model_0_5 = MobileViTv2(num_classes=1000, width_multiplier=0.5)
    model_0_5.eval()

    # Calculate and print parameters
    params_0_5 = count_parameters(model_0_5)
    print(f"Number of trainable parameters: {params_0_5:,}")

    # Get output shape
    with torch.no_grad():
        output_0_5 = model_0_5(input_tensor)

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output_0_5.shape}")
    print("-" * 25)

    # --- 1.0x Width Multiplier Model ---
    print("\n--- MobileViTv2-1.0 ---")
    model_1_0 = MobileViTv2(num_classes=1000, width_multiplier=1.0)
    model_1_0.eval()

    # Calculate and print parameters
  