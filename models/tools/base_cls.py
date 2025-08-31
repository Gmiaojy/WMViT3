#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Dict, Tuple, Union
import argparse
from tools.layers import norm_layers_tuple, LinearLayer
from tools.init_utils import initialize_weights, initialize_fc_layer
from tools.mobilevit_block import MobileViTBlockv3


class BaseEncoder(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the backbone
        # We allow that using output_stride and replace_stride_with_dilation arguments
        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True

        self.model_conf_dict = dict()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""
        return parser

    def check_model(self):
        assert (
            self.model_conf_dict
        ), "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def reset_parameters(self, opts):
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())

    def update_classifier(self, opts, n_classes: int):
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        linear_init_type = getattr(opts, "model.layer.linear_init", "normal")
        if isinstance(self.classifier, nn.Sequential):
            in_features = self.classifier[-1].in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)
            self.classifier[-1] = layer
        else:
            in_features = self.classifier.in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)

            # re-init head
            head_init_scale = 0.001
            layer.weight.data.mul_(head_init_scale)
            layer.bias.data.mul_(head_init_scale)

            self.classifier = layer
        return

    def extract_end_points_all(
            self,
            x: Tensor,
            use_l5: Optional[bool] = True,
            use_l5_exp: Optional[bool] = False,
            *args,
            **kwargs
    ) -> Dict[str, Tensor]:
        out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy
        x = self.conv_1(x)
        x = self.layer_1(x, **kwargs)  # 传递 kwargs
        out_dict["out_l1"] = x
        x = self.layer_2(x, **kwargs)  # 传递 kwargs
        out_dict["out_l2"] = x
        x = self.layer_3(x, **kwargs)  # 传递 kwargs
        out_dict["out_l3"] = x
        x = self.layer_4(x, **kwargs)  # 传递 kwargs
        out_dict["out_l4"] = x
        if use_l5:
            x = self.layer_5(x, **kwargs)  # 传递 kwargs
            out_dict["out_l5"] = x
            if use_l5_exp:
                x = self.conv_1x1_exp(x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def extract_end_points_l4(self, x: Tensor, *args, **kwargs) -> Dict[str, Tensor]:
        return self.extract_end_points_all(x, use_l5=False)

    def extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        # 不再需要在这里添加 original_image，让调用者决定

        x = self.conv_1(x)

        for layer_name in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]:
            layer_sequential = getattr(self, layer_name)
            if layer_sequential is None:
                continue

            # 手动迭代 Sequential 容器中的子模块
            for sub_module in layer_sequential:
                # --- 手动类型判断 ---
                # 如果子模块是我们自定义的、需要kwargs的特殊块
                if isinstance(sub_module, MobileViTBlockv3):
                    x = sub_module(x, **kwargs)
                # 否则，对于标准的 nn.Module (如 InvertedResidual)
                else:
                    x = sub_module(x)

        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # 将 kwargs 原封不动地传递给 extract_features
        x = self.extract_features(x, **kwargs)
        x = self.classifier(x)
        return x

    def freeze_norm_layers(self) -> None:
        """Freeze normalization layers"""
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

