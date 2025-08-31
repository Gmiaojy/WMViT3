# -*- coding: utf-8 -*-
# @Author  : Miao Guo
# @University  : ZheJiang University
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import copy
import random
import sys
import logging
import argparse
import pickle
from torch.utils.data import DataLoader, Subset
from utils.dataset import TripletDataset 
from utils.dataloader import create_weighted_sampler
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torchvision import transforms
from utils.evaluate_model import count_model_flops
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from models.get_model import get_model
from utils.print_plot import setup_logging, plot_confusion_matrix, plot_training_curves

# =================================== GLOBAL CONFIG & SEEDING ===================================
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
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return seed_worker
        
# =================================== ARGUMENT PARSING ===================================
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="wmvit3 Training Script")
    # Path Arguments
    parser.add_argument('--data-dir', type=str, default='/home/user/gm/bert/datas/Input', help="Root directory of the dataset.")
    parser.add_argument('--output-dir', type=str, default="outputs", help="Directory to save logs, models, and plots.")
    parser.add_argument('--split-file', type=str, default='utils/dataset_splits_train_test.pkl', help="Path to the .pkl file containing train/test index splits.")
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=2, help="Total number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.002, help="Learning rate.")
    parser.add_argument('--wd', type=float, default=0.01, help="Weight decay coefficient for the AdamW optimizer (L2 penalty).")
    parser.add_argument('--batch-size', type=int, default=64, help="Training/Validation/Test batch size.")
    parser.add_argument('--label-smoothing', type=float, default=0.2, help="Coefficient for label smoothing.")
    
    # Model & Data Arguments
    parser.add_argument('--input-size', type=int, default=256, help="Input image size for the model.")
    parser.add_argument('--fusion-method', type=str, default='rgb_0_1_2', help="Method to fuse triplet images for TripletDataset.")
    
    # System & Execution Arguments
    parser.add_argument('--num-workers', type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument('--gpu-id', type=int, default=1, help="GPU device ID to use.")
    
    return parser.parse_args()

# =================================== CORE TRAINING & VALIDATION ===================================
def train_one_epoch(net, data_loader, loss_fn, optimizer, device, scaler):
    net.train()
    total_loss, total_acc, total_samples = 0.0, 0.0, 0
    for X, y, _ in data_loader: # Ignore basename
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float32, enabled=True):
            y_hat = net(X)
            if hasattr(y_hat, 'logits'): 
                y_hat = y_hat.logits
            loss = loss_fn(y_hat, y)
            
        if torch.isnan(loss):
            logging.error("!!! NaN loss detected. Halting training. !!!")
            torch.save(X, 'bad_batch_X.pt')
            torch.save(y, 'bad_batch_y.pt')
            sys.exit("Stopping due to NaN loss.")
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            total_loss += loss.item() * y.size(0)
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
    return total_loss / total_samples, total_acc / total_samples

def validate_one_epoch(net, data_loader, device):
    net.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for X, y, _ in data_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            if hasattr(y_hat, 'logits'):
                y_hat = y_hat.logits
            preds = y_hat.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1, all_labels, all_preds


def train_and_validate(net, model_name, train_iter, val_iter, args, device, class_names):
    """Orchestrates the training and validation process to find and save the best model."""
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=(device.type == 'cuda'),init_scale=2.**12)

    best_val_acc = 0.0
    best_epoch = -1
    best_val_f1 = 0
    best_model_path = ""
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_labels, best_val_preds = None, None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(net, train_iter, loss_fn, optimizer, device, scaler)
        val_acc, val_f1,val_labels, val_preds = validate_one_epoch(net, val_iter, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        logging.info(f"  Epoch [{epoch+1}/{args.epochs}] | "
                     f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                     f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            best_val_labels, best_val_preds = val_labels, val_preds
            
            best_model_state = copy.deepcopy(net.state_dict())
            model_save_name = f'best_model_{model_name}.pth'
            best_model_path = os.path.join(args.output_dir, model_save_name)
            torch.save(best_model_state, best_model_path)
    
    logging.info(f"  \nTraining finished! \nBest Val Acc: {best_val_acc:.4f} at epoch {best_epoch+1}.")
    curves_save_path = os.path.join(args.output_dir, f'Training_Curves_{model_name}.jpg')
    plot_training_curves(history, curves_save_path, title=f'Loss/Acc Curves - {model_name}')
    
    if best_val_labels is not None and best_val_preds is not None:
        cm = confusion_matrix(best_val_labels, best_val_preds)
        cm_save_path = os.path.join(args.output_dir, f'CM_Validation_{model_name}.jpg')
        plot_confusion_matrix(cm, class_names, save_path=cm_save_path, normalize=True,
                              title=f"Best Val CM - {model_name} (Epoch {best_epoch+1})")
    
    return {'best_val_acc': best_val_acc, 'best_val_f1': best_val_f1}, best_model_path


def measure_inference_time(net, device, input_size, num_iterations=200):
    """Measures the average inference time of the model."""
    net.eval().to(device)
    dummy_sample = torch.randn(1, 3, input_size, input_size, device=device)
    timings = []
    with torch.no_grad():
        for _ in range(20): 
            _ = net(dummy_sample) # Warm-up
        if device.type == 'cuda':
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.Event(enable_timing=True)
            for _ in range(num_iterations):
                starter.record()
                _ = net(dummy_sample)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        else:
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = net(dummy_sample)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)
    return np.mean(timings) if timings else 0

# =================================== MAIN EXECUTION BLOCK =============================================
def main():
    worker_init_fn = set_seed(42)
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    setup_logging(args.output_dir, logger_name='test_log.txt')
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    ABLATION_CONFIG = {
        # # ours
        # test model width_multiplier(C=0.5,0.6,0.75,0.8) at identity ratio = 0.5
        # 'wmvit_080':[0.5],
        # 'wmvit_075':[0.5],
        'wmvit_060':[0.5],
        # 'wmvit_050':[0.5],
        
        # test model identity ratio(r=[0.1:1.1:0.1]) at width_multiplier=0.6
        # 'wmvit_060':np.arange(0.1, 1.1, 0.1).tolist(),
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

    # --- 1. Data Preparation ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    logging.info(f"Loading data splits from {args.split_file}.. .")
    with open(args.split_file, 'rb') as f:
        splits = pickle.load(f)
    train_val_indices = splits['train_indices']
    test_indices = splits['test_indices']

    full_dataset = TripletDataset(root_dir=args.data_dir, fusion_method=args.fusion_method)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    train_val_labels = np.array(full_dataset.get_labels())[train_val_indices]
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices, 
        train_val_labels, 
        test_size=0.1, 
        stratify=train_val_labels, 
        random_state=42
    )
    logging.info(f"Data loaded. Train set: {len(train_indices)}, Val set: {len(val_indices)}, Test set: {len(test_indices)}.")
    
    # --- 2. Main Experiment Loop ---
    summary_csv_path = os.path.join(args.output_dir, "model_compare_summary.csv")
    all_results = []

    for model_base_name, ratios_to_test in ABLATION_CONFIG.items():
        for ratio in ratios_to_test:
            model_name = f"{model_base_name}_ratio_{ratio:.1f}"if ratio != 1.0 else model_base_name
            logging.info(f"\n{'='*25} Starting Experiment: {model_name.upper()} {'='*25}")

            # --- Pre-calculation of Model Properties ---
            temp_net = get_model(model_base_name, num_classes, args.input_size, ratio=ratio)
            flops, params = count_model_flops(copy.deepcopy(temp_net),
                                              torch.randn(1, 3, args.input_size, args.input_size))
            fps = 1000.0 / measure_inference_time(temp_net, device, args.input_size)
            del temp_net; torch.cuda.empty_cache()

            # --- Train and Validate Model ---
            train_dataset_transformed = TripletDataset(
                root_dir=args.data_dir, 
                fusion_method=args.fusion_method, 
                transform=train_transform)
            val_test_dataset_transformed = TripletDataset(
                root_dir=args.data_dir, 
                fusion_method=args.fusion_method, 
                transform=val_test_transform)
            
            sampler = create_weighted_sampler(train_dataset_transformed, train_indices)
            train_subset = Subset(train_dataset_transformed, train_indices)
            val_subset = Subset(val_test_dataset_transformed, val_indices)
            
            train_iter = DataLoader(train_subset, batch_size=args.batch_size, 
                                    sampler=sampler, shuffle=False, 
                                    num_workers=args.num_workers, pin_memory=True, 
                                    worker_init_fn=worker_init_fn)
            val_iter = DataLoader(val_subset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=args.num_workers, 
                                  pin_memory=True, worker_init_fn=worker_init_fn)
            
            net = get_model(model_base_name, num_classes, args.input_size, ratio=ratio)
            best_val_metrics, best_model_path = train_and_validate(net, model_name, 
                                                                   train_iter, val_iter, 
                                                                   args, device, class_names)
            
            # --- Consolidate All Results ---
            current_result = {
                "Model": f"{model_name}", 
                "Params (M)": f"{(params/1e6):.2f}",
                "FLOPs (G)": f"{(flops/1e9):.2f}", "FPS": f"{fps:.2f}",
                "Best Val Acc (%)": f"{best_val_metrics['best_val_acc']*100:.2f}",
                "Best Val F1": f"{best_val_metrics['best_val_f1']:.4f}"
            }
            all_results.append(current_result)
            file_exits = os.path.exists(summary_csv_path)
            pd.DataFrame([current_result]).to_csv(
                summary_csv_path, 
                mode='a',
                header=not file_exits,
                index=False
            )
            logging.info(f"--- Experiment {model_name} complete. ")

            del net, train_iter, val_iter; torch.cuda.empty_cache()
            time.sleep(3)

    logging.info("\n\n" + "="*40 + " ALL EXPERIMENTS FINISHED " + "="*40)
    df_summary = pd.DataFrame(all_results)
    logging.info("Final Experiment Summary Report:")
    logging.info(df_summary.to_string(index=False))

if __name__ == '__main__':
    main()