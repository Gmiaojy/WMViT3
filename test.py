# -*- coding: utf-8 -*-
# @Author  : Miao Guo
# @University  : ZheJiang University

# From the full datas folder ("Input") load datas,
# Perform evaluation and save the results and heatmaps
import torch
import numpy as np
import time
import argparse
import os
import logging
import random
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from models.get_model import get_model
from utils.dataset import TripletDataset
from utils.print_plot import plot_confusion_matrix, setup_logging
from utils.vit_heatmap import get_cam_target_layers, save_vit_attention_heatmap, save_cam_heatmap


def disable_inplace_activations(model):
    """
    Recursively finds all activation layers with an 'inplace' attribute and sets it to False.
    """
    for module in model.modules():
        if hasattr(module, 'inplace') and isinstance(module, (torch.nn.Hardswish, torch.nn.SiLU, torch.nn.ReLU)):
            logging.info(f"Disabling 'inplace=True' for layer: {module.__class__.__name__}")
            module.inplace = False
    return model


def seed_worker(worker_id):
    """Sets the seed for a DataLoader worker to ensure deterministic data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    """Sets all relevant random seeds for reproducibility and returns a worker_init_fn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    return seed_worker


def evaluate_and_visualize(net, data_iter, device, output_dir, model_name):
    net.eval()
    all_labels, all_preds = [], []
    hook_handles = []

    # --- ViT-style Attention Hook Registration ---
    vit_heatmaps_storage = {}
    if "wmvit" in model_name or "mobilevit" in model_name:
        def get_vit_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'last_attention') and module.last_attention is not None:
                    vit_heatmaps_storage[name] = module.last_attention.cpu()
            return hook
        try:
            target_layers = {'layer3': net.layer_3, 'layer4': net.layer_4, 'layer5': net.layer_5}
            for name, layer_module in target_layers.items():
                attention_module = layer_module[1].vit_path.global_rep[0].pre_norm_attn[1]
                handle = attention_module.register_forward_hook(get_vit_attention_hook(name))
                hook_handles.append(handle)
            logging.info("Successfully registered ViT forward hooks on layers 3, 4, 5.")
        except Exception as e:
            logging.error(f"Failed to register ViT hooks. Error: {e}")

    # --- CAM Hook Registration ---
    grad_cam_storage = {'features': None, 'grads': None}
    target_layer, _ = get_cam_target_layers(model_name, net)
    
    if target_layer:
        def get_features_hook(module, input, output):
            grad_cam_storage['features'] = output

        def get_grads_hook(module, grad_in, grad_out):
            # grad_out is a tuple, we need the first element
            grad_cam_storage['grads'] = grad_out[0].clone()
        
        # Register hooks for Grad-CAM
        handle_features = target_layer.register_forward_hook(get_features_hook)
        handle_grads = target_layer.register_full_backward_hook(get_grads_hook)
        hook_handles.extend([handle_features, handle_grads])
        logging.info(f"Successfully registered Grad-CAM hooks on layer: {target_layer.__class__.__name__}")
    
    # --- Main Evaluation Loop ---
    for i, (X, y, basenames) in enumerate(data_iter):
        basename = basenames[0]
        
        vit_heatmaps_storage.clear()
        grad_cam_storage.clear()
    
        X, y = X.to(device), y.to(device)
        X.requires_grad = True
        y_hat = net(X) 
        pred_scores = y_hat.gather(1, y_hat.argmax(dim=1, keepdim=True)).squeeze()
        
        with torch.no_grad():
            if hasattr(y_hat, 'logits'): y_hat_for_preds = y_hat.logits
            else: y_hat_for_preds = y_hat
            preds = y_hat_for_preds.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        # --- Generate and Save Grad-CAM Heatmap ---
        if target_layer and pred_scores.numel() > 0:
            net.zero_grad()
            pred_scores.backward(retain_graph=True)

            if grad_cam_storage['grads'] is not None and grad_cam_storage['features'] is not None:
                grads = grad_cam_storage['grads'].squeeze(0) # Shape: [C, H, W]
                features = grad_cam_storage['features'].squeeze(0) # Shape: [C, H, W]
                weights = torch.mean(grads, dim=[1, 2]) # Shape: [C]
                grad_cam = torch.einsum('c,chw->hw', weights, features)
   
                grad_cam = torch.nn.functional.relu(grad_cam)
                if torch.max(grad_cam) > 0:
                    grad_cam = grad_cam / torch.max(grad_cam)
                grad_cam = grad_cam.detach().cpu().numpy()

                heatmap_dir = os.path.join(output_dir, "heatmaps_grad_cam_example")
                os.makedirs(heatmap_dir, exist_ok=True)
                save_path = os.path.join(heatmap_dir, f"{basename}.png")
                save_cam_heatmap(grad_cam, X, save_path)

        # --- Save ViT Attention Heatmaps  ---
        with torch.no_grad():
            if vit_heatmaps_storage:
                for layer_name, heatmap_tensor in vit_heatmaps_storage.items():
                    heatmap_dir = os.path.join(output_dir, "heatmaps_vit_attention_example", layer_name)
                    os.makedirs(heatmap_dir, exist_ok=True)
                    save_path = os.path.join(heatmap_dir, f"{basename}.png")
                    save_vit_attention_heatmap(heatmap_tensor, X, save_path)
        
    # --- Clean up all hooks ---
    for handle in hook_handles:
        handle.remove()
    if hook_handles:
        logging.info("Removed all forward hooks.")

    # --- Calculate final metrics ---
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1, all_labels, all_preds


def measure_inference_time(net, device, input_size, num_iterations=200):
    """Measures the average inference time of the model."""
    net.eval()
    dummy_sample = torch.randn(1, 3, input_size, input_size, device=device)
    timings = []

    if device.type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.Event(enable_timing=True)
        for _ in range(20):  # Warm-up
            _ = net(dummy_sample)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                starter.record()
                _ = net(dummy_sample)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
    else: # CPU inference
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = net(dummy_sample)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)

    avg_time_ms = np.mean(timings) if timings else 0
    return avg_time_ms

def parse_args():
    parser = argparse.ArgumentParser(description="Model Ablation Study Testing Script")
    parser.add_argument('--data-dir', type=str, default='wmvit3/datas/Input', help="Root directory of the dataset.")
    parser.add_argument('--output-dir', type=str, default='wmvit3/outputs', help="Root directory for loading models and saving results.")
    parser.add_argument('--split-file', type=str, default='wmvit3/utils/dataset_splits_train_test.pkl', help="Path to the .pkl file containing train/test index splits.")
    
    parser.add_argument('--batch-size', type=int, default=1, help="Test batch size. Must be 1 for heatmap visualization.")
    parser.add_argument('--input-size', type=int, default=256, help="Input image size for the model.")
    parser.add_argument('--fusion-method', type=str, default='rgb_0_1_2', help="Method to fuse triplet images.")
    
    parser.add_argument('--num-workers', type=int, default=1, help="Number of workers for data loading.")
    parser.add_argument('--gpu-id', type=int, default=0, help="GPU device ID to use.")  
    return parser.parse_args()


def main():
    args = parse_args()
    worker_init_fn = set_seed(42)
    
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
    
    if args.batch_size != 1:
        logging.warning("Batch size is being forced to 1 for heatmap visualization.")
        args.batch_size = 1
        
    log_file_path = os.path.join(args.output_dir, 'test_log.txt')
    setup_logging(args.output_dir, log_file_path)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting ablation study test on device: {device}")

    # Prepare data loader
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_base_dataset = TripletDataset(root_dir=args.data_dir, fusion_method=args.fusion_method, transform=test_transform)
    class_names = test_base_dataset.classes
    num_classes = len(class_names)
    
    # load the pre-saved splits
    with open(args.split_file, 'rb') as f:
        splits = pickle.load(f)
    test_indices = splits['test_indices']
    
    print("\n" + "="*20 + " Loading Data from Pre-saved Splits " + "="*20)
    print(f"Split file loaded from: {args.split_file}")
    print(f"  - Test samples: {len(test_indices)}")
    
    test_subset = Subset(test_base_dataset, test_indices)
    test_loader = DataLoader(
        test_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn
    ) 

    summary_results = []
    for model_name, ratios in TEST_LIST.items():
        for ratio in ratios:
            if ratio == 1.0:
                model_run_identifier = f'best_model_{model_name}'
            else:
                model_run_identifier = f'best_model_{model_name}_ratio_{ratio:.1f}'
            
            model_path = os.path.join(args.output_dir, 'weights', model_run_identifier+'.pth')
            logging.info("\n" + "="*80)
            
            if ratio == 1.0:
                model_name = f'{model_name}'
            else:
                model_name = f'{model_name}_ratio_{ratio:.1f}'
            logging.info(f"TESTING: Model - {model_name}")
            logging.info(f"Model Path: {model_path}")

            if not os.path.exists(model_path):
                logging.warning(f"SKIPPING: Model file not found at {model_path}")
                continue

            net = get_model(model_name, num_classes, args.input_size, ratio=ratio)
            net.load_state_dict(torch.load(model_path, map_location=device))
            net = disable_inplace_activations(net)
            net.to(device)

            run_output_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(run_output_dir, exist_ok=True)

            # --- Pass model_name to the evaluation function ---
            test_acc, test_prec, test_recall, test_f1, test_labels, test_preds = evaluate_and_visualize(
                net, test_loader, device, run_output_dir, model_name)
            
            if test_acc is None:
                logging.error(f"Evaluation failed for {model_name}. Check hook registration.")
                continue
            
            avg_inference_time = measure_inference_time(net, device, args.input_size)
            
            cm_path = os.path.join(run_output_dir, f'confusion_matrix_{model_name}.jpg')
            cm = confusion_matrix(test_labels, test_preds)
            plot_confusion_matrix(cm, class_names, save_path=cm_path, normalize=True, title=f"CM - {model_name}")
            
            logging.info(f"--- Results for {model_run_identifier} ---")
            results = {
                "Test Accuracy": f"{test_acc:.4f} ({test_acc*100:.2f}%)",
                "Test Precision (Macro)": f"{test_prec:.4f}",
                "Test Recall (Macro)": f"{test_recall:.4f}",
                "Test F1-Score (Macro)": f"{test_f1:.4f}",
                "Avg. Inference Time/Sample": f"{avg_inference_time:.4f} ms"
            }
            for key, value in results.items():
                logging.info(f"  -> {key:<30}: {value}")
            
            current_result = {
                "Model": model_name,
                "Test Accuracy (%)": f"{test_acc * 100:.2f}",
                "Test Precision (Macro)": f"{test_prec:.4f}",
                "Test recall (Macro)": f"{test_recall:.4f}",
                "Test F1-Score (Macro)": f"{test_f1:.4f}",
                "Avg. Inference Time/Sample (ms)": f"{avg_inference_time:.4f}",
                "Model Path": model_path
            }
            summary_results.append(current_result)
            logging.info("="*80)
            
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(args.output_dir, 'compare_test.csv')
        file_exists = os.path.exists(summary_csv_path)
        try:
            summary_df.to_csv(summary_csv_path, mode='a', header=not file_exists,index=False)
            logging.info(f"\nSUCCESS: Test study summary saved to {summary_csv_path}")
        except Exception as e:
            logging.error(f"\nERROR: Failed to save summary CSV file. Error: {e}")
    else:
        logging.warning("\nNo models were tested. The summary file was not created.")

if __name__ == '__main__':
    main()