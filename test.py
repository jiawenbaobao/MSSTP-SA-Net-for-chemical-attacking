import time
import os
import pickle
import random as r

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pylab import *

from init_project import create_dirs, set_seeds, args_set
from spatialmodel import SpatialModel
from seq2seq import Seq2seq
from seq2seq_atten import Seq2seq_attn
from seq2seq_mlp import Seq2seq_mlp
from dataprepare_new import DataPrepare, data_classify
from evaluation import correlation_coefficient, compute_RMSE, accuracy, \
    fit_performance

class Tester(object):
    def __init__(self, test_exps, data_folder, scales, input_seq_len,
                 pred_seq_len, model_spatial, model_time, extract_num, save_dir,
                 save_sample_path, device):

        self.test_exps = test_exps
        self.data_folder = data_folder
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.scales = scales
        self.model_spatial = model_spatial.to(device)
        self.model_time = model_time.to(device)
        self.device = device
        self.extract_num = extract_num
        self.save_dir = save_dir
        self.save_sample_path = save_sample_path

        self.y_pred_list = []
        self.y_target_list = []

        self.test_exps_num = 2
        self.doc = open(os.path.join(save_dir, 'out.txt'),'w')


    def run_test_loop(self):

        self.model_time.batch_size = 1
        self.model_spatial.eval()
        self.model_time.eval()

        rmse = 0
        r_delta = 0
        accuracy_rate = 0

        for exp in self.test_exps:
            input_path = os.path.join(self.save_sample_path, 'test',
                                      exp, 'inputs')
            target_time_path = os.path.join(self.save_sample_path, 'test',
                                            exp, 'target_time')

            length_data = len(os.listdir(input_path))
            y_pred_exp = []
            y_tar_exp = []
            for i in range(length_data):
                input_sample_path = os.path.join(input_path,
                                                 'sample_{0}.pkl'.format(i))
                target_time_sample_path = os.path.join(target_time_path,
                                                 'sample_{0}.pkl'.format(i))
                with open(input_sample_path, 'rb') as f:
                    input_sample = pickle.load(f)

                with open(target_time_sample_path, 'rb') as f:
                    target_time_sample = pickle.load(f)

                input_i = torch.FloatTensor(input_sample)
                y_time_i = torch.FloatTensor(target_time_sample).squeeze(-1)

                y_pred_i, y_target_i = \
                    self.pred(self.model_spatial, self.model_time, input_i,
                              y_time_i, self.device)

                y_pred_exp.append(y_pred_i)
                y_tar_exp.append(y_target_i)

            rmse_exp, r_exp, accuracy_rate_exp = \
                self.evaluation_index(y_pred_exp, y_tar_exp)

            rmse += rmse_exp
            # r += abs(r_exp)
            r_delta += abs(1 - r_exp)
            accuracy_rate += accuracy_rate_exp

            self.y_pred_list.append(y_pred_exp)
            self.y_target_list.append(y_tar_exp)

        self.evaluation_mse_figure()
        print("Test RMSE: {0:.3f}".format(rmse / len(self.test_exps)))
        print("Test r: {0:.3f}".format(1 - (r_delta / len(self.test_exps))))
        print(
            "Test accuracy: {0:.3f}".format(accuracy_rate / len(self.test_exps))
            )
        print("Test RMSE: {0:.3f}".format(rmse/ len(self.test_exps)),
              file=self.doc)
        print("Test r: {0:.3f}".format(1 - (r_delta/ len(self.test_exps))), file=self.doc)
        print("Test accuracy: {0:.3f}".format(accuracy_rate/ len(self.test_exps))
              ,file=self.doc)
        self.evaluation_fit_figure()
        self.space_concentration_figure()

    def pred(self, model_spatial, model_time, input_data, y_time, device):

        y_pred_spatial = model_spatial(input_data.to(device))
        # y_pred = model_time(y_pred_spatial, z_tar=y_time)
        y_pred = model_time(y_pred_spatial, z_tar=[], device=device,
                            use_teacher_forcing=False)
        y_pred = pow(y_pred, 2)

        y_time = y_time.cpu()
        y_pred = y_pred.squeeze().cpu()

        y_pre = Variable(y_pred).squeeze().numpy()
        y_tar = Variable(y_time).squeeze().numpy()

        return y_pre, y_tar


    def random_select_data(self):
        '''
        y_pred/target_points = [
        1 time: point1 poin2 poin3 point4
        2 time: point1 poin2 poin3 point4
        ...
        600time: point1 poin2 poin3 point4

        pred_target_pairs = [
        [y_pred_points:[1time, 2time, ... 600time],
        y_target_points:[1time, 2time, ... 600time]
        ],
        ...
        ]
        '''
        # choose a exp as example
        random_exp = r.sample(list(range(len(self.test_exps))), 1)[0]
        # the be choosed exp's y_pred
        y_pred_exp = np.stack(self.y_pred_list[random_exp])
        # the be choosed exp's y_target
        y_target_exp = np.stack(self.y_target_list[random_exp])
        points_num = y_pred_exp.shape[2]

        # random choice 5 points in target points
        random_extraction_points = r.sample(list(range(points_num)),
                                                 self.extract_num)
        print(random_extraction_points, file=self.doc)

        y_pred_points = y_pred_exp[0][:, random_extraction_points]
        y_target_points = y_target_exp[0][:, random_extraction_points]
        for j in range(1, len(y_pred_exp)):
            y_pred_points = np.concatenate((y_pred_points,
                y_pred_exp[j][-1, random_extraction_points].reshape(1,-1)))
            y_target_points = np.concatenate((
                y_target_points,
                y_target_exp[j][-1, random_extraction_points].reshape(1,-1)))

        pred_target_pairs = [[y_pred_points[:, i], y_target_points[:, i]]
                             for i in range(self.extract_num)]

        return pred_target_pairs

    def evaluation_mse_figure(self):

        pred_target_pairs = self.random_select_data()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        t = np.arange(0, 118, 0.2)

        for ax, data in zip(axes.flatten(), pred_target_pairs):
            ax.plot(t, data[0],'.-', label='y_pred')
            ax.plot(t, data[1], label='y_target')
            ax.set_xlabel('time/s')
            ax.set_ylabel('concentration (ppm)')

            ax.legend()

        plt.savefig(os.path.join(self.save_dir, "mse_performance.png"))
        plt.show()

    def evaluation_index(self, y_pred, y_tar):
        y_pred_cat = y_pred[0].reshape(-1)
        y_target_cat = y_tar[0].reshape(-1)
        for j in range(1, len(y_pred)):
            y_pred_cat = np.concatenate((y_pred_cat,
                                         y_pred[j].reshape(-1)))
            y_target_cat = np.concatenate((y_target_cat,
                                           y_tar[j].reshape(-1)))

        rmse = compute_RMSE(y_pred_cat, y_target_cat)
        r = correlation_coefficient(y_pred_cat, y_target_cat)
        # fb = FB(y_pred_cat, y_target_cat)
        accuracy_rate = accuracy(y_pred_cat, y_target_cat)

        return rmse, r, accuracy_rate

    def evaluation_fit_figure(self):
        random_extraction_exps = r.sample(list(range(len(self.test_exps))),
                                               self.test_exps_num)

        fig, axes = plt.subplots(2, int(self.test_exps_num / 2),
                                figsize=(10, 8))
        for ax, i in zip(axes.flatten(), random_extraction_exps):
            y_pred = np.stack(self.y_pred_list[i]).reshape(-1, 1)
            y_target = np.stack(self.y_target_list[i]).reshape(-1, 1)
            w, line = fit_performance(y_pred, y_target)
            ax.scatter(y_pred, y_target, label='Adjusted data')
            ax.plot(y_pred, line, c='r', label='Fit: y=' + str(w) + 'x')
            ax.set_xlabel('Model output')
            ax.set_ylabel('CFD output')

            ax.legend()

        plt.savefig(os.path.join(self.save_dir, "test_performance.png"))

        plt.show()


    def space_concentration_figure(self):
        # choose a exp as example
        random_exp = r.sample(list(range(len(self.test_exps))), 1)[0]
        # the be choosed exp's y_pred
        choose_y_pred_exp = np.stack(self.y_pred_list[random_exp])
        # the be choosed exp's y_target
        choose_y_target_exp = np.stack(self.y_target_list[random_exp])
        seq_num = choose_y_pred_exp.shape[0]

        # random choice 5 points in target points
        random_extraction_seq = r.sample(list(range(seq_num)), 1)
        print(random_extraction_seq, file=self.doc)

        y_pred_seq = self.y_change(choose_y_pred_exp, random_extraction_seq)
        y_tar_seq = self.y_change(choose_y_target_exp, random_extraction_seq)
        torch.save(y_pred_seq, os.path.join(self.save_dir, 'y_pred_seq'))
        torch.save(y_tar_seq, os.path.join(self.save_dir, 'y_tar_seq'))


    def y_change(self, y_exp, random_extraction_seq):
        '''
        map data in real space
        '''
        y_out = []
        for i in range(self.pred_seq_len):
            y_time = y_exp[random_extraction_seq, i, :]
            y_insert_time = np.insert(y_time, 263, 0)
            y_insert_time = np.insert(y_insert_time, 527, 0)
            y_insert_time = np.insert(y_insert_time, 791, 0)
            y_change = y_insert_time.reshape(24, 11, -1)
            y_out.append(y_change)
        return y_out

def main():

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

    # load state
    model_spatial = SpatialModel(num_input_channels=5,
                                 out_num=1053,
                                 dropout_p=args.dropout_p)

    model_time = Seq2seq(num_features=1053,
                         hidden_size=512, input_seq_len=args.input_seq_len,
                         pred_seq_len=args.pred_seq_len,
                         batch_size=1)
    # model_time = Seq2seq_attn(num_features=1053,
    #                           input_seq_len=args.input_seq_len,
    #                           pred_seq_len=args.pred_seq_len,
    #                           batch_size=1,
    #                           dropout=args.dropout_p)
    # model_time = Seq2seq_mlp(num_features=1053,
    #                          input_seq_len=args.input_seq_len,
    #                          pred_seq_len=args.pred_seq_len,
    #                          batch_size=1, device=args.device)

    resume = os.path.join(args.save_dir, 'check_point_{}'.format(40))
    print('Resuming model check point from {}\n'.format(40))
    check_point = torch.load(resume)
    model_spatial.load_state_dict(check_point['model_spatial'])
    model_spatial.to(args.device)
    model_time.load_state_dict(check_point['model_time'])
    model_time.to(args.device)

    # data = DataPrepare(save_dir=args.save_dir, data_folder=args.data_folder,
    #                    train_size=args.train_size,
    #                    val_size=args.val_size,
    #                    test_size=args.test_size,
    #                    input_seq_len=args.input_seq_len,
    #                    pred_seq_len=args.pred_seq_len, shuffle=True)
    # data.create_data()

    test_exps = np.load('exp_list.npy')
    scales = np.load('scales.npy')

    tester = Tester(test_exps=test_exps, data_folder=args.data_folder,
                    scales=scales, input_seq_len=args.input_seq_len,
                    pred_seq_len=args.pred_seq_len,
                    model_spatial=model_spatial,
                    model_time=model_time, extract_num=4,
                    save_dir = args.save_dir,
                    save_sample_path=args.save_sample_path,
                    device='cuda')
    tester.run_test_loop()



if __name__ == '__main__':
    main()