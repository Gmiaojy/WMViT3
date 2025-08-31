# -*- coding: utf-8 -*-
# @Author  : Miao Guo
# @University  : ZheJiang University

# From the example-datas folder ("input_example") load datas,
# Perform inference and save the prediction results and heatmaps
import torch
import numpy as np
import time
import argparse
import os
import cv2
import logging
import random
import pandas as pd
import pickle
from thop import profile
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
from models.get_model import get_model
from utils.dataset import TripletDataset
from utils.print_plot import plot_confusion_matrix, setup_logging
from utils.vit_heatmap import get_cam_target_layers, save_vit_attention_heatmap, save_cam_heatmap



class TripletInferenceDataset(Dataset):
    """
    Dataset class for inference.
    It scans a folder, finds all groups of files ending with _0, _1, _2.png,
    and stitches them together into a three-channel RGB image.
    - _0.png -> R channel
    - _1.png -> G channel
    - _2.png -> B channel
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Scan the folder and find the base_name
        all_files = os.listdir(root_dir)
        base_names = set()
        for f in all_files:
            if f.endswith('_0.png') or f.endswith('_1.png') or f.endswith('_2.png'):
                base_names.add(f[:-6])
        
        # Sort to ensure that the order of load is right
        self.sample_basenames = sorted(list(base_names))
        
        logging.info(f"in folder {root_dir} find {len(self.sample_basenames)} 3 channel image groups.")
        if not self.sample_basenames:
            logging.warning("warning: in specified folder doesn't find (*_0.png, *_1.png, *_2.png)")

    def __len__(self):
        return len(self.sample_basenames)

    def __getitem__(self, idx):
        base_name = self.sample_basenames[idx]
        path_0 = os.path.join(self.root_dir, f"{base_name}_0.png")
        path_1 = os.path.join(self.root_dir, f"{base_name}_1.png")
        path_2 = os.path.join(self.root_dir, f"{base_name}_2.png")
        
        try:
            img_0 = Image.open(path_0).convert('L')
            img_1 = Image.open(path_1).convert('L')
            img_2 = Image.open(path_2).convert('L')
        except FileNotFoundError as e:
            logging.error(f"load '{base_name}' failure, file lost: {e}")
            raise IOError(f"File not found for triplet {base_name}") from e

        # Merge three single-channel images into an RGB image
        rgb_image = Image.merge('RGB', (img_0, img_1, img_2))
        
        # Apply image transformation
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        output_filename = f"{base_name}.png"
        return rgb_image, output_filename


def disable_inplace_activations(model):
    for module in model.modules():
        if hasattr(module, 'inplace'): module.inplace = False
    return model

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
    return seed_worker

def measure_inference_time(net, device, input_size, num_iterations=200):
    net.eval()
    dummy_sample = torch.randn(1, 3, input_size, input_size, device=device)
    timings = []
    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(20): _ = net(dummy_sample)
        with torch.no_grad():
            for _ in range(num_iterations):
                starter.record()
                _ = net(dummy_sample)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = net(dummy_sample)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)
    return np.mean(timings) if timings else 0

def infer_and_visualize(net, data_iter, device, output_dir, model_name):
    net.eval()
    predictions = []
    hook_handles = []
    vit_heatmaps_storage = {}
    
    def get_vit_attention_hook(name):
        def hook(module, input, output):
            if hasattr(module, 'last_attention') and module.last_attention is not None:
                vit_heatmaps_storage[name] = module.last_attention.cpu()
        return hook
    
    if "mobilevit" or "wmvit" in model_name:
        try:
            target_layers = {'layer3': net.layer_3, 'layer4': net.layer_4, 'layer5': net.layer_5}
            for layer_key, layer_module in target_layers.items():
                attention_module = None
                if "wmvit" in model_name:
                    attention_module = layer_module[1].vit_path.global_rep[0].pre_norm_attn[1]
                    logging.info(f" RULE [WMVIT]: in {layer_key} find attention module")
                elif "mobilevitv3" in model_name:
                    mobilevit_block = None
                    for mod in layer_module:
                        if "MobileViTBlock" in mod.__class__.__name__:
                            mobilevit_block = mod
                            break
                    if mobilevit_block:
                        found = False
                        for sub_module in mobilevit_block.modules():
                            # LinearSelfAttention is goal module
                            if "LinearSelfAttention" in sub_module.__class__.__name__:
                                attention_module = sub_module
                                logging.info(f"RULE [MobileViT]: in {layer_key} find LinearSelfAttention")
                                found = True
                                break
                        if not found:
                             logging.warning(f"RULE [MobileViT]: in {layer_key} miss LinearSelfAttention")
                    else:
                        logging.warning(f"RULE [MobileViT]: in {layer_key} miss MobileViTBlock")
                
                if attention_module:
                    handle = attention_module.register_forward_hook(get_vit_attention_hook(layer_key))
                    hook_handles.append(handle)
                else:
                    logging.warning(f"in {layer_key} miss {model_name} matched attention rule")
        except Exception as e: 
            logging.error(f"register ViT hooks fail: {e}")
    grad_cam_storage = {'features': None, 'grads': None}
    target_layer, _ = get_cam_target_layers(model_name, net)
    if target_layer:
        def get_features_hook(module, input, output): 
            grad_cam_storage['features'] = output
        def get_grads_hook(module, grad_in, grad_out): 
            grad_cam_storage['grads'] = grad_out[0].clone()
        handle_features = target_layer.register_forward_hook(get_features_hook)
        handle_grads = target_layer.register_full_backward_hook(get_grads_hook)
        hook_handles.extend([handle_features, handle_grads])
    
    # Inference loop
    for i, (X, basenames) in enumerate(data_iter):
        basename = basenames[0]
        vit_heatmaps_storage.clear()
        grad_cam_storage.clear()
        X = X.to(device)
        X.requires_grad = True
        y_hat = net(X)
        pred_scores = y_hat.gather(1, y_hat.argmax(dim=1, keepdim=True)).squeeze()
        with torch.no_grad():
            pred_index = y_hat.argmax(dim=1).cpu().item()
            predictions.append({'filename': basename, 'predicted_index': pred_index})
        if target_layer and pred_scores.numel() > 0:
            net.zero_grad()
            pred_scores.backward(retain_graph=True)
            if grad_cam_storage['grads'] is not None and grad_cam_storage['features'] is not None:
                grads = grad_cam_storage['grads'].squeeze(0)
                features = grad_cam_storage['features'].squeeze(0)
                weights = torch.mean(grads, dim=[1, 2])
                grad_cam = torch.nn.functional.relu(torch.einsum('c,chw->hw', weights, features))
                if torch.max(grad_cam) > 0: 
                    grad_cam = grad_cam / torch.max(grad_cam)
                heatmap_dir = os.path.join(output_dir, "heatmaps_grad_cam_example")
                os.makedirs(heatmap_dir, exist_ok=True)
                save_cam_heatmap(grad_cam.detach().cpu().numpy(), X, os.path.join(heatmap_dir, f"{basename}.png"))
        with torch.no_grad():
            if vit_heatmaps_storage:
                for layer_name, heatmap_tensor in vit_heatmaps_storage.items():
                    heatmap_dir = os.path.join(output_dir, "heatmaps_vit_attention_example", layer_name)
                    os.makedirs(heatmap_dir, exist_ok=True)
                    save_vit_attention_heatmap(heatmap_tensor, X, os.path.join(heatmap_dir, f"{basename}.png"))
    for handle in hook_handles:
        handle.remove()
    return predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference and Visualization at example datas")
    parser.add_argument('--data-dir', type=str, default='datas/input_example', help="Root directory of the example dataset.")
    # If download the full dataset, you can remove the following line of comment.
    # parser.add_argument('--train-data-dir', type=str, default='wmvit3/datas/Input', help="Root directory of the training dataset to fetch class names.")
    parser.add_argument('--output-dir', type=str, default='outputs', help="Root directory for loading models and saving results.")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for inference. Must be 1 for heatmap visualization.")
    parser.add_argument('--input-size', type=int, default=256, help="Input image size for the model.")
    parser.add_argument('--fusion-method', type=str, default='rgb_0_1_2', help="Fusion method, e.g., 'rgb_0_1_2'.")
    parser.add_argument('--num-workers', type=int, default=1, help="Number of worker threads for data loading.")
    parser.add_argument('--gpu-id', type=int, default=0, help="GPU ID to use for inference.")
    return parser.parse_args()


def main():
    worker_init_fn = set_seed(42)
    args = parse_args()
    if args.batch_size != 1:
        logging.warning("For heatmap visualization, batch_size must be 1. Setting batch_size=1.")
        args.batch_size = 1
    
    setup_logging(args.output_dir, logger_name = 'inference_log.txt')
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"in device {device} start inference...")

    TEST_LIST = {
                    # # ours
                    # test model width_multiplier(C=0.5,0.6,0.75,0.8) at identity ratio = 0.5
                    # 'wmvit_080':[0.5],
                    # 'wmvit_075':[0.5],
                    'wmvit_060':[0.5],
                    # 'wmvit_050':[0.5],
                    
                    # test model identity ratio(r=[0.1:1.1:0.1]) at width_multiplier=0.6
                    # 'wmvti_060':np.arange(0.1, 1.1, 0.1).tolist(),
                   
                    # # Lightweight CNNs models (from timm)
                    'edgenext_xx_small': [1.0], 
                    'shufflenet_v2_x1_0': [1.0],
                    'fasternet_t0': [1.0],       
                    'inception_next_tiny': [1.0],
                    'mobilenetv2_100': [1.0],
                    'mobilenetv3_large_100': [1.0], 

                    # # Transformer & Hybrid models (from timm)
                    'efficientvit_b0.r224_in1k': [1.0],
                    'poolformer_s12': [1.0],
                    'pvt_v2_b0': [1.0],
                    # Transformer & Hybrid models (from defined in models)
                    'mobilevit_s': [1.0], # MobileViT v1
                    'mobilevitv2_s': [1.0], # MobileViT v2
                    "mobilevitv3_s": [1.0],  
                    "mobilevitv3_xs": [1.0], 
                }
    
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    """
    # Get class names from the training dataset
    # If the complete dataset has been downloaded, this comment can be removed
    
    try:
        logging.info(f" from'{args.train_data_dir}' load class names...")
        train_dataset_for_classes = TripletDataset(root_dir=args.train_data_dir, fusion_method=args.fusion_method, transform=test_transform)
        class_names = train_dataset_for_classes.classes
        num_classes = len(class_names)
        logging.info(f"Successfully obtained {num_classes} classes: {class_names}")
    except Exception as e:
        logging.error(f"Fail to obtain class names from '{args.train_data_dir}'. Error: {e}")
        return
    """
    class_names = ["ABS500", "PA6500", "PE500", "PP500", "PS500"] 
    num_classes = len(class_names)
    
    inference_dataset = TripletInferenceDataset(root_dir=args.data_dir, transform=test_transform)
    if len(inference_dataset) == 0:
        logging.error("No valid triplet image groups found for inference. Please check the data directory.")
        return
        
    inference_loader = DataLoader(
        inference_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    ) 

    for model_name_base, ratios in TEST_LIST.items():
        for ratio in ratios:
            model_run_identifier = f'best_model_{model_name_base}' if ratio == 1.0 else f'best_model_{model_name_base}_ratio_{ratio:.1f}'
            model_path = os.path.join(args.output_dir, model_run_identifier + '.pth')
            model_name_for_log = f'{model_name_base}' if ratio == 1.0 else f'{model_name_base}_ratio_{ratio:.1f}'
            
            logging.info("\n" + "="*80)
            logging.info(f"The model is being tested: {model_name_for_log}")
            logging.info(f"model path: {model_path}")

            if not os.path.exists(model_path):
                logging.warning(f"Skip: Model file not found in {model_path}")
                continue

            net = get_model(model_name_base, num_classes, args.input_size, ratio=ratio)
            net.load_state_dict(torch.load(model_path, map_location=device))
            net = disable_inplace_activations(net)
            net.to(device)

            run_output_dir = os.path.join(args.output_dir, model_name_for_log)
            os.makedirs(run_output_dir, exist_ok=True)

            predictions = infer_and_visualize(net, inference_loader, device, run_output_dir, model_name_for_log)
            
            if not predictions:
                logging.warning(f"model {model_name_for_log} no prediction results were generated.")
                continue

            for p in predictions: p['predicted_class'] = class_names[p['predicted_index']]
            
            predictions_df = pd.DataFrame(predictions)
            csv_save_path = os.path.join(run_output_dir, f'predictions_{model_name_for_log}.csv')
            try:
                predictions_df.to_csv(csv_save_path, index=False)
                logging.info(f"The prediction results have been successfully saved to: {csv_save_path}")
            except Exception as e:
                logging.error(f"Failed to save the prediction results to a CSV file: {e}")

            avg_inference_time = measure_inference_time(net, device, args.input_size)
            logging.info(f"--- model {model_name_for_log} reason finished ---")
            logging.info(f"  -> Average inference time per sheet: {avg_inference_time:.4f} ms")
            logging.info("="*80)

if __name__ == '__main__':
    main()