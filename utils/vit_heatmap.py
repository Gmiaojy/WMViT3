# -*- coding: utf-8 -*-
# @Author  : Miao Guo
# @University  : ZheJiang University
import logging
import torch
import cv2
import numpy as np


def find_image_content_area(image_np):
    """
    Finds the bounding box of the non-black content in a padded image.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(c)
    else:
        return 0, 0, image_np.shape[1], image_np.shape[0]

def smart_overlay_heatmap(base_image_np, heatmap_color_np):
    """
    Overlays a heatmap onto the central content area of a padded base image.
    """
    x, y, w, h = find_image_content_area(base_image_np)
    if w > 0 and h > 0:
        heatmap_resized = cv2.resize(heatmap_color_np, (w, h))
        content_area = base_image_np[y:y+h, x:x+w]
        blended_content = cv2.addWeighted(content_area, 0.5, heatmap_resized, 0.5, 0)

        output_image = base_image_np.copy()
        output_image[y:y+h, x:x+w] = blended_content
        return output_image
    return base_image_np

def save_vit_attention_heatmap(heatmap_tensor, original_image_tensor, save_path):
    """
    Generates and saves a heatmap for ViT-style attention maps.
    """
    num_pixels = heatmap_tensor.shape[2]
    patch_h = patch_w = int(heatmap_tensor.shape[-1] ** 0.5)
    heatmap_tensor = heatmap_tensor.reshape(1, num_pixels, patch_h, patch_w)
    context_map = torch.nn.functional.pixel_shuffle(heatmap_tensor, upscale_factor=int(num_pixels ** 0.5))
    
    min_val, max_val = torch.min(context_map), torch.max(context_map)
    if max_val > min_val:
        context_map = (context_map - min_val) / (max_val - min_val)
    
    heatmap_8bit = (context_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_PLASMA)
    
    img_tensor = original_image_tensor.clone().detach().squeeze(0).cpu()
    img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    original_img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
    
    superimposed_img = smart_overlay_heatmap(original_img_bgr, heatmap_color)
    cv2.imwrite(save_path, superimposed_img)


def save_cam_heatmap(cam_map, original_image_tensor, save_path):
    """
    Generates and saves a heatmap from a 2D CAM numpy array.
    """
    heatmap_8bit = (cam_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_PLASMA)

    img_tensor = original_image_tensor.clone().detach().squeeze(0).cpu()
    img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    original_img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
    
    superimposed_img = smart_overlay_heatmap(original_img_bgr,heatmap_color)
    cv2.imwrite(save_path, superimposed_img)


def get_cam_target_layers(model_name, net):
    """
    Identifies a suitable convolutional layer and the classifier layer for CAM.
    """
    # --- Custom MobileViT/WaveViT ---
    if "wmvit" in model_name or "mobilevit" in model_name:
        try:
            final_conv = net.conv_1x1_exp
            classifier = net.classifier[-1] if isinstance(net.classifier, torch.nn.Sequential) else net.classifier
            logging.info(f"CAM Rule: Custom MobileViT/WaveViT. Target: {final_conv.__class__.__name__}")
            return final_conv, classifier
        except AttributeError:
            logging.warning(f"Could not apply custom CAM rule for {model_name}. Falling back...")

    # --- Modern hybrid architectures ---
    elif 'edgenext_xx_small' in model_name:
        try:
            target_conv = net.stages[-2]
            classifier = net.head.fc
            logging.info(f"CAM Rule: EdgeNeXt. Target: stages[-2]")
            return target_conv, classifier
        except Exception as e:
            logging.error(f"Error applying 'EdgeNeXt' CAM rule for {model_name}: {e}")
            return None, None
            
    # --- inception_next_tiny ---
    elif 'inception_next_tiny' in model_name:
        try:
            target_conv = net.stages[-2]
            classifier = net.head 
            logging.info(f"CAM Rule: InceptionNeXt. Target: stages[-2]")
            return target_conv, classifier
        except Exception as e:
            logging.error(f"Error applying 'InceptionNeXt' CAM rule for {model_name}: {e}")
            return None, None

    elif 'fasternet' in model_name:
        try:
            target_conv = net.stages[-2]
            classifier = net.classifier
            logging.info(f"CAM Rule: FasterNet. Target: stages[-2]")
            return target_conv, classifier
        except Exception as e:
            logging.error(f"Error applying 'FasterNet' CAM rule for {model_name}: {e}")
            return None, None
        
    elif 'efficientvit' in model_name:
        try:
            target_conv = net.stages[-2]
            classifier = net.head
            logging.info(f"CAM Rule: EfficientViT. Target: stages[-2]")
            return target_conv, classifier
        except Exception as e:
             logging.error(f"Error applying 'EfficientViT' CAM rule for {model_name}: {e}")
             return None, None
         
    elif 'poolformer' in model_name:
        try:
            target_conv = net.stages[-2]
            classifier = net.head
            logging.info(f"CAM Rule: PoolFormer. Target: stages[-2]")
            return target_conv, classifier
        except Exception as e:
             logging.error(f"Error applying 'PoolFormer' CAM rule for {model_name}: {e}")
             return None, None

    # ---  Classic CNNs ---
    elif 'shufflenet_v2' in model_name:
        try:
            target_conv = net.conv5
            classifier = net.fc
            logging.info(f"CAM Rule: ShuffleNetV2. Target: conv5")
            return target_conv, classifier
        except Exception as e:
             logging.error(f"Error applying 'ShuffleNetV2' CAM rule for {model_name}: {e}")
             return None, None

    elif model_name in ['mobilenetv2_100', 'mobilenetv3_large_100']:
        try:
            target_conv = net.conv_head
            classifier = net.classifier
            logging.info(f"CAM Rule: TIMM MobileNet. Target: conv_head")
            return target_conv, classifier
        except Exception as e:
             logging.error(f"Error applying 'TIMM MobileNet' CAM rule for {model_name}: {e}")
             return None, None
             
    # --- Pure Transformers ---
    elif 'pvt_v2' in model_name:
        logging.warning(f"CAM is not applicable to pure Transformer architectures like {model_name}. Skipping.")
        return None, None

    # --- Fallback ---
    elif "resnet" in model_name:
        return net.layer4, net.fc
    elif "efficientnet" in model_name:
        return net.features, net.classifier[-1]

    logging.warning(f"Could not find CAM target layers for model {model_name}. Please update function.")
    return None, None