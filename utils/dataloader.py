import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, Dataset
from torchvision import transforms
import os
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from utils.dataset import TripletDataset
import pickle


def create_weighted_sampler(base_dataset, train_indices, verbose=True):
    """ Create a weighted sampler for the training dataset."""
    if verbose:
        print("\n" + "="*10 + " Create Weighted Sampler " + "="*10)
    
    train_labels = base_dataset.get_labels()[train_indices]
    class_counts = np.bincount(train_labels, minlength=len(base_dataset.classes))
    
    weights_per_class = 1.0 / (class_counts + 1e-6)
    
    if verbose:
        np.set_printoptions(precision=4, suppress=True)
        print(f"  Calculated weight for each CLASS (1/count): {weights_per_class}")
        
    sample_weights = np.array([weights_per_class[label] for label in train_labels])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    if verbose:
        print("  WeightedRandomSampler created successfully.")
    return sampler

def create_dataloaders_from_splits(
    root_dir,
    split_file_path,
    train_transform,
    test_transform,
    batch_size_train,
    batch_size_val,
    num_workers,
    fusion_method,
    pin_memory=True,
    verbose=True,
    worker_init_fn=None
):
    """
    Create DataLoaders for training, validation, and test sets using pre-saved splits.
    """
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(
            f"Split file not found at {split_file_path}. "
            "Please run a script to generate dataset_splits.pkl first."
        )

    # load the full dataset
    train_base_dataset = TripletDataset(root_dir=root_dir, fusion_method=fusion_method, transform=train_transform)
    val_test_base_dataset = TripletDataset(root_dir=root_dir, fusion_method=fusion_method, transform=test_transform)
    class_names = train_base_dataset.classes
    num_classes = len(class_names)
    
    # load the pre-saved splits
    with open(split_file_path, 'rb') as f:
        splits = pickle.load(f)
    train_indices = splits['train_indices']
    test_indices = splits['test_indices']

    if verbose:
        print("\n" + "="*20 + " Loading Data from Pre-saved Splits " + "="*20)
        print(f"Split file loaded from: {split_file_path}")
        print(f"  - Train samples: {len(train_indices)}")
        print(f"  - Test samples: {len(test_indices)}")
        print("="*71)


    # create Subsets
    train_subset = Subset(train_base_dataset, train_indices)
    test_subset = Subset(val_test_base_dataset, test_indices)
    
    # create DataLoaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size_train, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
        worker_init_fn=worker_init_fn
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size_val, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )
    
    test_loader = DataLoader(
        test_subset, batch_size=batch_size_val, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader, class_names, num_classes