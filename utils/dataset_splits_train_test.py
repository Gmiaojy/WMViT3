import numpy as np
import os
from sklearn.model_selection import train_test_split
from dataset import TripletDataset
import pickle

def main():
    """
    Stratify into a training set and a test set
    """
    DATA_ROOT = 'wmvit/datas/Input'   
    OUTPUT_DIR = 'wmvit/utils' 
    TEST_RATIO = 0.1                                   

    os.makedirs(OUTPUT_DIR, exist_ok=True)

  
    try:
        full_dataset = TripletDataset(root_dir=DATA_ROOT, fusion_method='direct')
        all_labels = full_dataset.get_labels()
        all_indices = np.arange(len(full_dataset))
    except Exception as e:
        print(f"Error in initializing the dataset: {e}")
        print("Please ensure the 'dataset.py' file and the data directory are set correctly.")
        return

    # Hierarchical division
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_RATIO,
        random_state=42, 
        stratify=all_labels  # Maintain consistent category distribution
    )

    print("The dataset division is completed")
    print(f"total samples num: {len(all_indices)}")
    print(f"train set samples num: {len(train_indices)}")
    print(f"test set samples num: {len(test_indices)}")

    splits = {
        'train_indices': train_indices,
        'test_indices': test_indices
    }

    save_path = os.path.join(OUTPUT_DIR, 'dataset_splits_train_test.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)

    print(f"\nThe dataset division index has been successfully saved to: {save_path}")

if __name__ == '__main__':
    main()