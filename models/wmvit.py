import torch
from torch import nn, Tensor
import argparse
from typing import Dict, Tuple, Optional
import torch.nn.functional as F
import pywt
from tools.layers import ConvLayer, LinearLayer, GlobalPool, Identity
from tools.mobilevit_block import MobileViTBlockv3_wave as WMViTBase
from tools.utils import make_divisible
from tools.base_cls import BaseEncoder
from tools.mobilevit_block import InvertedResidual
from tools.config import get_configuration
from mobilevitv3 import create_mock_opts

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """Create the filters required for wavelet transform and inverse transform"""
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    """Perform wavelet decomposition"""
    b, c, h, w = x.shape
    pad_h = (filters.shape[2] - 2) // 2
    pad_w = (filters.shape[3] - 2) // 2
    x = F.conv2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    """ Perform inverse wavelet transform """
    b, c, _, h_half, w_half = x.shape
    pad_h = (filters.shape[2] - 2) // 2
    pad_w = (filters.shape[3] - 2) // 2
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    return x


# ========== HybridViTMambaBlock ===========
class MobileWaveViTBlock(nn.Module):
    """
    A hybrid module that splits the input channels.
    Some of the channels go through the MobileViT attention path and the wavelet transform path,
    while the other part of the channels go through the identity mapping directly
    and then they are concatenated and fused.
    """
    def __init__(self,
                 opts,
                 in_channels: int,
                 attn_unit_dim: int,
                 ffn_multiplier: int,
                 process_ratio: float = 0.5,
                 n_attn_blocks: int = 1,
                 patch_h: int = 2,
                 patch_w: int = 2,
                 dropout: float = 0.0,
                 ffn_dropout: float = 0.0,
                 attn_dropout: float = 0.0,
                 attn_norm_layer: str = "layer_norm_2d",
                 wt_levels: int = 1,
                 wt_type: str = 'db1',
                 wt_conv_kernel_size: int = 5):
        super().__init__()
        self.process_channels = make_divisible(int(in_channels * process_ratio), 8)
        self.identity_channels = in_channels - self.process_channels

        if self.process_channels > 0:  # ratio<1
            # Path A: MobileViT ATTN path
            self.vit_path = WMViTBase(
                opts=opts,
                in_channels=self.process_channels,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=n_attn_blocks,
                patch_h=patch_h,
                patch_w=patch_w,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
                conv_ksize=3,
                attn_norm_layer=attn_norm_layer,
                process_channels=self.process_channels
            )
        # Path B: Wavelet Transform path
        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, self.process_channels, self.process_channels
        )
        self.register_buffer('wavelet_filter', self.wt_filter)
        self.register_buffer('iwavelet_filter', self.iwt_filter)

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(self.process_channels * 4, self.process_channels * 4,
                       kernel_size=wt_conv_kernel_size, padding='same',
                       groups=self.process_channels * 4, bias=False)
             for _ in range(self.wt_levels)]
        )
        self.proj = ConvLayer(opts, in_channels, in_channels, kernel_size=1, use_norm=True, use_act=True)

    def _forward_wavelet(self, x: Tensor) -> Tensor:
        """Encapsulate the forward propagation of the wavelet path"""
        curr_x_ll = x
        if self.wt_levels != 1: raise NotImplementedError()
        curr_shape = curr_x_ll.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
            curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
            curr_x_ll = F.pad(curr_x_ll, curr_pads)

        curr_x_decomposed = wavelet_transform(curr_x_ll, self.wavelet_filter)
        shape_x = curr_x_decomposed.shape
        curr_x_tag = curr_x_decomposed.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
        curr_x_tag = self.wavelet_convs[0](curr_x_tag)
        curr_x_processed = curr_x_tag.reshape(shape_x)

        reconstructed_x = inverse_wavelet_transform(curr_x_processed, self.iwavelet_filter)
        reconstructed_x = reconstructed_x[:, :, :curr_shape[2], :curr_shape[3]]
        return reconstructed_x

    def forward(self, x: Tensor) -> Tensor:
        x_to_process, x_identity = torch.split(x, [self.process_channels, self.identity_channels], dim=1)
        vit_out = self.vit_path(x_to_process)
        wavelet_out = self._forward_wavelet(x_to_process)
        processed_fused = vit_out + wavelet_out 
        
        if self.identity_channels > 0: # ratio<1
            x_concatenated = torch.cat([processed_fused, x_identity], dim=1)
        else:  # ratio=1,identity=0
            x_concatenated = processed_fused + x

        return self.proj(x_concatenated)

# ========= MobileWaveViT ========
class MobileWaveViT(BaseEncoder):
    def __init__(self, opts, *args, **kwargs) -> None:
        self.dilate_l4 = False
        self.dilate_l5 = False
        self.dilation = 1
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        mobilevit_config = get_configuration(opts=opts)
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        super().__init__(opts, *args, **kwargs)
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(opts, image_channels, out_channels, 3, 2, use_norm=True, use_act=True)
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(opts, in_channels, mobilevit_config["layer1"])
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(opts, in_channels, mobilevit_config["layer2"])
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(opts, in_channels, mobilevit_config["layer3"])
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(opts, in_channels, mobilevit_config["layer4"],
                                                      dilate=self.dilate_l4)
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(opts, in_channels, mobilevit_config["layer5"],
                                                      dilate=self.dilate_l5)
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}
        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {"in": out_channels, "out": out_channels}
        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            LinearLayer(opts=opts, in_features=out_channels, out_features=num_classes, bias=True),
        )

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_hybrid_layer(opts, input_channel, cfg, dilate=dilate)  # Differences from MobileViT
        else:
            return self._make_mobilenet_layer(opts, input_channel, cfg)

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(opts, input_channel, output_channels, stride, expand_ratio)
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_hybrid_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1
            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )
            block.append(layer)
            input_channel = cfg.get("out_channels")

        process_ratio = getattr(opts, "model.classification.hybrid.process_ratio", 0.4)

        block.append(
            MobileWaveViTBlock(
                opts=opts,
                in_channels=input_channel,
                process_ratio=process_ratio,
                attn_unit_dim=cfg["attn_unit_dim"],
                ffn_multiplier=cfg.get("ffn_multiplier"),
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
            )
        )
        return nn.Sequential(*block), input_channel


def create_mock_opts_hybrid(width_multiplier: float, num_classes: int,
                            process_ratio: float = 0.5) -> argparse.Namespace:
    """Create a simulated opts object for the hybrid model"""
    opts = create_mock_opts(width_multiplier, num_classes)
    setattr(opts, "model.classification.hybrid.process_ratio", process_ratio)
    return opts


def wmvit_xxs(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=0.5, num_classes=num_classes,process_ratio=ratio) 
    return MobileWaveViT(opts=mock_opts)
def wmvit_xs(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=0.75, num_classes=num_classes, process_ratio=ratio)
    return MobileWaveViT(opts=mock_opts)
def wmvit_s(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=1.0, num_classes=num_classes, process_ratio=ratio)
    return MobileWaveViT(opts=mock_opts)


def wmvit_050(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=0.5, num_classes=num_classes,process_ratio=ratio) 
    return MobileWaveViT(opts=mock_opts)
def wmvit_060(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=0.60, num_classes=num_classes, process_ratio=ratio)
    return MobileWaveViT(opts=mock_opts)
def wmvit_075(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=0.75, num_classes=num_classes, process_ratio=ratio)
    return MobileWaveViT(opts=mock_opts)
def wmvit_080(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts_hybrid(width_multiplier=0.80, num_classes=num_classes, process_ratio=ratio)
    return MobileWaveViT(opts=mock_opts)


if __name__ == '__main__':
    mock_opts_hybrid = create_mock_opts_hybrid(
        width_multiplier=0.5, 
        num_classes=1000,
        process_ratio=0.4 
    )
    print("--- Testing HybridMobileViTMamba ---")
    hybrid_model = MobileWaveViT(opts=mock_opts_hybrid)
    hybrid_model.eval()

    num_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    input_tensor = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = hybrid_model(input_tensor)

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (2, 1000)
    print("\nModel instantiated and forward pass completed successfully!")