import torch
from torch import nn
import argparse
from typing import Dict, Tuple, Optional
# from ..classification import register_cls_models
from tools.base_cls import BaseEncoder
from tools.layers import ConvLayer, LinearLayer, GlobalPool, Identity
from tools.mobilevit_block import MobileViTBlockv3 as Block
from tools.mobilevit_block import InvertedResidual
from tools.config import get_configuration
from tools.utils import make_divisible


class MobileViTv3(BaseEncoder):
    """
    This class defines the MobileViTv3 architecture
    """

    def __init__(self, opts, external_helper=None, *args, **kwargs) -> None:
        # Since this is a BaseEncoder, it has some attributes we need to define
        # even if not used in this standalone script.
        self.dilate_l4 = False
        self.dilate_l5 = False
        self.external_helper = external_helper
        self.dilation = 1

        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        mobilevit_config = get_configuration(opts=opts)
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]

        super().__init__(opts, *args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()

        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            LinearLayer(opts=opts, in_features=out_channels, out_features=num_classes, bias=True),
        )

        # In a standalone script, we might not need these checks/initializations
        # or we ensure the mock opts has the required fields.
        # self.check_model()
        # self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # This function is for command-line training, not needed for instantiation.
        # We will add these arguments to our mock opts manually.
        pass

    def _make_layer(
            self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
            opts, input_channel: int, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    def _make_mit_layer(
            self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        # self.dilation is used here, so we must define it in __init__
        prev_dilation = self.dilation if hasattr(self, "dilation") else 1
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                if hasattr(self, "dilation"):
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
            input_channel = cfg.get("out_channels")  # input_channel 被更新为 InvertedResidual 块的输出通道数

        process_ratio = getattr(opts, "model.classification.mitv3.process_ratio",
                                1.0)  # 从 opts 中获取全局比例因子，如果未设置，则默认为1.0（即原始逻辑）
        calculated_process_channels = None
        if process_ratio < 1.0:
            calculated_process_channels = make_divisible(int(input_channel * process_ratio), 8)
        else:
            calculated_process_channels = input_channel

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")
        dropout = getattr(opts, "model.classification.mitv3.dropout", 0.0)

        block.append(
            Block(
                opts=opts,
                in_channels=input_channel,  # mobilevitv3Block的输入维度
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(opts, "model.classification.mitv3.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mitv3.attn_dropout", 0.0),
                conv_ksize=3,
                attn_norm_layer=getattr(opts, "model.classification.mitv3.attn_norm_layer", "layer_norm_2d"),
                dilation=prev_dilation,
                process_channels=calculated_process_channels
            )
        )
        return nn.Sequential(*block), input_channel


def create_mock_opts(width_multiplier: float, num_classes: int, process_ratio: float = 1.0) -> argparse.Namespace:
    """
    Creates a mock opts object with the necessary configuration for MobileViTv3.
    """

    default_config = {
        "model": {
            "classification": {
                "name": "mobilevit",
                "classifier_dropout": 0.1,
                "mit": {
                    "mode": "x_small",
                    "ffn_dropout": 0.0,
                    "attn_dropout": 0.0,
                    "dropout": 0.1,
                    "no_fuse_local_global_features": False,
                    "conv_kernel_size": 3,
                    "width_multiplier": 1.0,
                    "process_ratio": 1.0,
                    "attn_norm_layer": "layer_norm_2d"
                }
            },
            "normalization": {
                "name": "batch_norm_2d",
                "momentum": 0.1,
                "layer": {"name": "layer_norm_2d"}
            },
            "activation": {
                "name": "swish"
            },
            "layer": {
                "global_pool": "mean",
                "conv_init": "kaiming_normal",
                "linear_init": "trunc_normal",
                "linear_init_std_dev": 0.02,
                "norm_type": "batch_norm",
                "act_type": "swish"
            }
        },
        "dev": {
            "device": "cpu"
        }
    }
    # Merge the default configuration with the provided final_config
    opts = argparse.Namespace()

    model_conf = default_config['model']
    cls_conf = model_conf['classification']
    mit_conf = cls_conf['mit']
    norm_conf = model_conf['normalization']
    act_conf = model_conf['activation']
    layer_conf = model_conf['layer']

    # General model configs
    setattr(opts, "model.classification.n_classes", num_classes)
    setattr(opts, "model.layer.global_pool", layer_conf['global_pool'])
    setattr(opts, "model.layer.norm_type", layer_conf['norm_type'])
    setattr(opts, "model.layer.act_type", act_conf['name'])
    setattr(opts, "model.activation.name", act_conf['name'])
    setattr(opts, "model.normalization.momentum", norm_conf['momentum'])

    # MobileViTv3 specific configs
    setattr(opts, "model.classification.mitv3.width_multiplier", width_multiplier)
    setattr(opts, "model.classification.mitv3.process_ratio", process_ratio)
    setattr(opts, "model.classification.mitv3.attn_dropout", mit_conf['attn_dropout'])
    setattr(opts, "model.classification.mitv3.ffn_dropout", mit_conf['ffn_dropout'])
    setattr(opts, "model.classification.mitv3.dropout", mit_conf['dropout'])
    setattr(opts, "model.classification.mitv3.attn_norm_layer", mit_conf['attn_norm_layer'])

    # For MobileViTBlockv3, it might need some fusion options
    setattr(opts, "model.normalization.layer.name", norm_conf['layer']['name'])
    setattr(opts, "dev.device", default_config['dev']['device'])

    return opts


def mobilevitv3_xxs(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts(width_multiplier=0.5, num_classes=num_classes, process_ratio=ratio)
    return MobileViTv3(opts=mock_opts)


def mobilevitv3_xs(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts(width_multiplier=0.75, num_classes=num_classes, process_ratio=ratio)
    return MobileViTv3(opts=mock_opts)


def mobilevitv3_s(num_classes=5, ratio=1.0):
    mock_opts = create_mock_opts(width_multiplier=1.0, num_classes=num_classes, process_ratio=ratio)
    return MobileViTv3(opts=mock_opts)


if __name__ == '__main__':
    # Define different scales for MobileViT-v3 based on width_multiplier
    model_scales = {
        "S (1.0x)": 1.0,
        "XS (0.75x)": 0.75,
        "XXS (0.5x)": 0.5,
    }

    input_tensor = torch.randn(5, 3, 256, 256)

    for scale_name, width_mult in model_scales.items():
        print(f"--- Testing MobileViT-v3-{scale_name} ---")

        # 1. Create the mock configuration object
        mock_opts = create_mock_opts(width_multiplier=width_mult, num_classes=1000, process_ratio=0.5)

        # 2. Instantiate the model
        model = MobileViTv3(opts=mock_opts)
        model.eval()

        # 3. Perform a forward pass and check shapes
        with torch.no_grad():
            output = model(input_tensor)

        print(f"Input shape:  {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print("-" * 40 + "\n")