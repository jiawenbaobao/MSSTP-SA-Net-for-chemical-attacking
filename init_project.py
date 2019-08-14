import os
import torch
import numpy as np

from argparse import Namespace

def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Set Numpy and PyTorch seeds
def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def args_set(pattern):
    """ Parse input argument"""
    if pattern == 'small':
        args = Namespace(
            seed=1234,
            cuda=False,
            shuffle=True,
            data_folder="/home/zhyhou/xjw/processed_data_test",
            spatial_model_state_file="spatial_model.pth",
            time_model_state_file='time_model.pth',
            save_dir="./result/checkpoint_test",
            save_sample_path='/home/zhyhou/xjw/sample_test',
            input_seq_len=10,
            pred_seq_len=5,
            train_size=0.2,
            val_size=0.2,
            test_size=0.2,
            alpha=0.8,
            num_epochs=1,
            extract_num=4,
            early_stopping_criteria=5,
            learning_rate=1e-3,
            batch_size=1,
            dropout_p=0.5,
            teacher_forcing_ratio=0.5,
            resume=0
        )
    elif pattern == 'big':
        args = Namespace(
            seed=1234,
            cuda=False,
            shuffle=True,
            data_folder="/home/zhyhou/xjw/processed_data",
            spatial_model_state_file="spatial_model.pth",
            time_model_state_file='time_model.pth',
            save_dir="./result/train29",
            save_sample_path='/home/zhyhou/xjw/sample',
            input_seq_len=10,
            pred_seq_len=5,
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            alpha=0.8,
            num_epochs=50,
            extract_num=4,
            early_stopping_criteria=5,
            learning_rate=1e-3,
            batch_size=32,
            dropout_p=0.5,
            teacher_forcing_ratio=0.5,
            resume=0
        )
    else:
        raise KeyError(pattern)
    return args

def args_train_state(early_stopping_criteria, learning_rate):
    train_state = {
    'done_training': False,
    'stop_early': False,
    'early_stopping_step': 0,
    'early_stopping_best_val': 1e8,
    'early_stopping_criteria': early_stopping_criteria,
    'learning_rate': learning_rate,
    'epoch_index': 0,
    'train_loss': [],
    'val_loss': [],
    }
    return train_state