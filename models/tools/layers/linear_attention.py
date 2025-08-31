#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
from torch import Tensor
from typing import Optional, Tuple
from torch.nn import functional as F

from .base_layer import BaseLayer
from .conv_layer import ConvLayer
from .dropout import Dropout
import cv2
import numpy as np
from glob import glob
import os


class LinearSelfAttention(BaseLayer):
    """
    This layer applies a self-attention with linear complexity, as described in `this paper <>`_
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
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
        opts,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim
        
        self.last_attention = None  # store the last attention scores


    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    @staticmethod
    def visualize_context_scores(
        context_scores: torch.Tensor,
        model_name_param: str,
        original_image_tensor: Optional[torch.Tensor] = None,
        output_dir: str = "attn_res_ratio",
        alpha: float = 0.6,
        beta: float = 0.4):

        # [B, 1, P, N]
        batch_size, channels, num_pixels, num_patches = context_scores.shape
        assert batch_size == 1, "For visualization purposes, use batch size of 1"
        assert (
            channels == 1
        ), "The inner-product between input and latent node (query) is a scalar"

        up_scale_factor = int(num_pixels ** 0.5)
        patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
        # [1, 1, P, N] --> [1, P, h, w]
        context_scores = context_scores.reshape(1, num_pixels, patch_h, patch_w)
        # Fold context scores [1, P, h, w] using pixel shuffle to obtain [1, 1, H, W]
        context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
        # [1, 1, H, W] --> [H, W]
        context_map = context_map.squeeze()

        # For ease of visualization, we do min-max normalization
        min_val = torch.min(context_map)
        max_val = torch.max(context_map)
        context_map = (context_map - min_val) / (max_val - min_val)
        
        model_specific_output_dir = os.path.join(output_dir, model_name_param)
        os.makedirs(model_specific_output_dir, exist_ok=True)
        display_img = None
        file_prefix = ""

        try:
            # 1. 将热图转换为 8-bit 整数并应用颜色映射
            heatmap_8bit = (context_map * 255).byte().cpu().numpy()
            heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

            # 检查是否传入了原始图像
            if original_image_tensor is not None:
                # 2. 将原始图像Tensor转换为OpenCV图像 (numpy array)
                # 反归一化: (tensor * std) + mean
                img_tensor = original_image_tensor.clone().squeeze(0) # (3, H, W)
                CHANNEL_MEAN = [0.5, 0.5, 0.5]
                CHANNEL_STD = [0.5, 0.5, 0.5]
                for i in range(3):
                    img_tensor[i] = (img_tensor[i] * CHANNEL_STD[i]) + CHANNEL_MEAN[i]

                # 转换为 [0, 255] 范围的 numpy 数组，并调整通道顺序为 HWC
                original_img_np = (img_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
                # PyTorch RGB -> OpenCV BGR
                original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

                # 3. 确保热图和原图尺寸一致
                target_size = (original_img_bgr.shape[1], original_img_bgr.shape[0]) # (width, height)
                heatmap_resized = cv2.resize(heatmap_color, target_size)

                # 4. 使用 cv2.addWeighted 进行融合
                alpha = 0.6  # 原图权重
                beta = 0.4   # 热图权重
                gamma = 0
                superimposed_img = cv2.addWeighted(original_img_bgr, alpha, heatmap_resized, beta, gamma)
                
                display_img = superimposed_img
                base_f_name = f"h{patch_h}_w{patch_w}"
            
            else:
                # 如果没有原图，则只显示热图
                heatmap_resized = cv2.resize(heatmap_color, (256, 256), interpolation=cv2.INTER_NEAREST)
                display_img = heatmap_resized
                base_f_name = f"heatmap_h{patch_h}_w{patch_w}"

            # 5. 保存结果          
            idx = 0
            while True:
                f_name = os.path.join(model_specific_output_dir, f"{base_f_name}_index_{idx}.png")
                if not os.path.exists(f_name):
                    break
                idx += 1
        
            cv2.imwrite(f_name, display_img)
            # print(f"Attempting to save visualization to: {f_name}")
            return display_img

        except ModuleNotFoundError:
            print("Please install OpenCV (`pip install opencv-python`) to visualize context maps")
            return context_map.cpu().numpy()

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
        
        self.last_attention = context_scores.detach() # Store the last attention scores for visualization
        
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
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M]
        context_vector = key * context_scores
        # [B, d, P, M] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if kwargs.get('x_prev', None) is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, *args, **kwargs)
