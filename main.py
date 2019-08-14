import os
import time

import torch
import torch.optim as optim
import numpy as np

from init_project import create_dirs, set_seeds, args_set, args_train_state
from dataset_new import SpatialTimeDataset
from spatialmodel import SpatialModel
from seq2seq import Seq2seq
from seq2seq_atten import Seq2seq_attn
from seq2seq_mlp import Seq2seq_mlp
from train import Trainer, collate_fn
from test import Tester
from evaluation import plot_performance


def main():
    # Arguments
    args = args_set('big')

    # Create save dir
    create_dirs(args.save_dir)

    # Check CUDA
    if torch.cuda.is_available():
        args.cuda = True
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set seeds
    set_seeds(seed=1234, cuda=args.cuda)

    dataset = SpatialTimeDataset(args.save_sample_path)

    # create model
    model_spatial = SpatialModel(num_input_channels=dataset[0][0].shape[1],
                                 out_num=1053,
                                 dropout_p=args.dropout_p)

    # model_time = Seq2seq_mlp(num_features=1053,
    #                          input_seq_len=args.input_seq_len,
    #                          pred_seq_len=args.pred_seq_len,
    #                          batch_size=args.batch_size, device=args.device)
    model_time = Seq2seq(num_features=1053,
                         hidden_size=512, input_seq_len=args.input_seq_len,
                         pred_seq_len=args.pred_seq_len,
                         batch_size=args.batch_size)
    # model_time = Seq2seq_attn(num_features=data.targets_time['train'].shape[2],
    #                           input_seq_len=args.input_seq_len,
    #                           pred_seq_len=args.pred_seq_len,
    #                           batch_size=args.batch_size,
    #                           dropout=args.dropout_p)

    optimizer = optim.Adam([{'params': model_spatial.parameters()},
                            {'params': model_time.parameters()}],
                           lr=args.learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=[12, 25, 37],
    #                                            gamma=0.1,
    #                                            last_epoch = start_epoch-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.5, patience=1)

    start_epoch = args.resume
    train_state = args_train_state(
        early_stopping_criteria=args.early_stopping_criteria,
        learning_rate=args.learning_rate)

    if args.resume:
        resume = os.path.join(args.save_dir, 'check_point_{}'.format(
            args.resume))
        print('Resuming model check point from {}\n'.format(resume))
        check_point = torch.load(resume)
        start_epoch = check_point['epoch']

        model_spatial.load_state_dict(check_point['model_spatial'])
        model_spatial.to(args.device)

        model_time.load_state_dict(check_point['model_time'])
        model_time.to(args.device)

        optimizer.load_state_dict(check_point['optimizer'])

        train_state = check_point['train_state']

        scheduler.optimizer = optimizer
        scheduler.last_epoch = start_epoch - 1
        scheduler.cooldown_counter = check_point['lr']['cooldown_counter']
        scheduler.best = check_point['lr']['best']
        scheduler.num_bad_epochs = check_point['lr']['num_bad_epochs']
        scheduler.mode_worse = check_point['lr']['mode_worse']
        scheduler.is_better = check_point['lr']['is_better']

    # define train class
    trainer = Trainer(dataset=dataset, model_spatial=model_spatial,
                      model_time=model_time, optimizer=optimizer,
                      scheduler=scheduler, device=args.device,
                      teacher_forcing_ratio=args.teacher_forcing_ratio,
                      train_state=train_state)

    # train & validation
    print('start train29 training...')
    for epoch_index in range(start_epoch, args.num_epochs):
        epoch_start = time.time()

        trainer.train_state['epoch_index'] = epoch_index + 1

        dataset.set_split('train')
        batch_generator_train = dataset.generate_batches(
            batch_size=args.batch_size, collate_fn=collate_fn,
            shuffle=args.shuffle, device=args.device)
        trainer.run_train_loop(batch_generator_train, args.alpha,
                               device=args.device)

        epoch_end = time.time()

        print('\nEntire epoch train time cost: {:.2f} min'.format(
            (epoch_end - epoch_start) / 60
        ))

        dataset.set_split('val')
        batch_generator_val = dataset.generate_batches(
            batch_size=args.batch_size, collate_fn=collate_fn,
            shuffle=False, device=args.device)
        trainer.run_val_loop(batch_generator_val, device=args.device)

        # check point
        save_name = os.path.join(args.save_dir, 'check_point_{}'.format(
            trainer.train_state['epoch_index']))
        check_point = {
            'epoch': trainer.train_state['epoch_index'],
            'model_spatial': trainer.model_spatial.state_dict(),
            'model_time': trainer.model_time.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'train_state': trainer.train_state,
            'lr': {'cooldown_counter':trainer.scheduler.cooldown_counter,
                   'best':trainer.scheduler.cooldown_counter,
                   'num_bad_epochs':trainer.scheduler.num_bad_epochs,
                   'mode_worse': trainer.scheduler.mode_worse,
                   'is_better': trainer.scheduler.is_better}
        }

        torch.save(check_point, save_name)

        if trainer.train_state['stop_early']:
            break

    #plot loss
    plot_performance(trainer.train_state['train_loss'],
                     trainer.train_state['val_loss'], args.save_dir)

    print('start testing...')

    test_exps = np.load('exp_list.npy',allow_pickle=True)
    scales = np.load('scales.npy',allow_pickle=True)
    # test
    tester = Tester(test_exps=test_exps, data_folder=args.data_folder,
                    scales=scales, input_seq_len=args.input_seq_len,
                    pred_seq_len=args.pred_seq_len,
                    model_spatial=model_spatial,
                    model_time=model_time, extract_num=4,
                    save_dir=args.save_dir,
                    save_sample_path=args.save_sample_path,
                    device='cuda')
    tester.run_test_loop()

if __name__ == '__main__':
    main()
