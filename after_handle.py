import os
import random as r
import collections

import numpy as np
import torch
import matplotlib.pyplot as plt
from pylab import *

from init_project import set_seeds
from evaluation import correlation_coefficient, compute_RMSE, accuracy, \
    fit_performance


def random_select_data(random_exp, random_extraction_points, y):
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
    y_exp = np.stack(y[random_exp])

    y_points = y_exp[0][:, random_extraction_points]
    for j in range(1, len(y_exp)):
        y_points = np.concatenate((y_points,
                                    y_exp[j][-1, random_extraction_points].
                                    reshape(1, -1)))

    return y_points


def evaluation_mse_figure(model_target, model_result):

    t = np.arange(2, 120, 0.2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, model_target, '.-', label='Experiment data')
    for key, value in model_result.items():
        ax.plot(t, value, '.-', label=key)

    ax.set_xlabel('time/s')
    ax.set_ylabel('concentration (ppm)')
    ax.legend()
    plt.savefig(os.path.join('./result', "test.png"))
    plt.show()

def space_concentration_figure(y_pred_list, y_tar_list):
    # choose a exp as example
    random_exp = r.sample(list(range(len(y_tar_list))), 1)[0]
    print(random_exp)
    # the be choosed exp's y_pred
    choose_y_pred_exp = np.stack(y_pred_list[random_exp])
    # the be choosed exp's y_target
    choose_y_target_exp = np.stack(y_tar_list[random_exp])
    seq_num = choose_y_pred_exp.shape[0]

    # random choice 5 points in target points
    random_extraction_seq = r.sample(list(range(seq_num)), 1)

    y_pred_seq = y_change(choose_y_pred_exp, random_extraction_seq)
    y_tar_seq = y_change(choose_y_target_exp, random_extraction_seq)
    torch.save(y_pred_seq, os.path.join('./result', 'y_pred_seq'))
    torch.save(y_tar_seq, os.path.join('./result', 'y_tar_seq'))

def y_change(y_exp, random_extraction_seq, pred_seq_len=5):
    '''
    map data in real space
    '''
    y_out = []

    for i in range(pred_seq_len):
        y_time = y_exp[random_extraction_seq, i, :]
        y_insert_time = np.insert(y_time, 263, 0)
        y_insert_time = np.insert(y_insert_time, 527, 0)
        y_insert_time = np.insert(y_insert_time, 791, 0)
        # y_change = y_insert_time.reshape(24, 11, -1)
        x = 0
        y_change = np.zeros([24, 11, 4])
        for z in list(range(4)):
            for y in list(range(11)):
                y_change[:,y,z] = y_insert_time[x: x+24]
                x += 24

        y_out.append(y_change)
    return y_out

def multi_model_compare_figure():
    model_result = {}
    result_path = {}
    result_path['my_model'] = '../seq2seq/result/train29'
    result_path['mq-mlp'] = '../seq2seq/result/train23'
    result_path['spatial-GRU'] = '/home/zhyhou/xjw/single_predict/result/checkpoint'
    result_path['temporal-only'] = '/home/zhyhou/xjw/timemodel_only/result/checkpoint'

    y_tar_list = torch.load(
        os.path.join(result_path['my_model'], 'y_tar_list'))

    test_exp_len = len(y_tar_list)
    # choose a exp as example
    random_exp = r.sample(list(range(test_exp_len)), 1)[0]

    # random choice 1 points in target points
    random_extraction_points = r.sample(list(range(1053)), 1)

    model_target = random_select_data(random_exp,
                                      random_extraction_points,
                                      y_tar_list)

    for key, item in result_path.items():
        y_pred_list_path = os.path.join(result_path[key], 'y_pred_list')
        y_pred_list = torch.load(y_pred_list_path)
        model_result[key] = random_select_data(random_exp,
                                               random_extraction_points,
                                               y_pred_list)

    evaluation_mse_figure(model_target, model_result)


    print('test')

def evaluation_fit_figure(y_pred, y_target, tag):
    # random_extraction_exps = r.sample(list(range(len(self.test_exps))),
    #                                            self.test_exps_num)

    # for i in range(len(y_pred_list)):
    #     y_pred = np.stack(y_pred_list[i]).reshape(-1, 1)
    #     y_target = np.stack(y_target_list[i]).reshape(-1, 1)
    #     w, line = fit_performance(y_pred, y_target)
    #     print(i, w)

    # fig, axes = plt.subplots(2, int(4 / 2),
    #                             figsize=(12, 8))
    # for ax, i in zip(axes.flatten(), [3,4,11,24]):
    #     y_pred = np.stack(y_pred_list[i]).reshape(-1, 1)
    #     y_target = np.stack(y_target_list[i]).reshape(-1, 1)
    #     w, line = fit_performance(y_pred, y_target)
    #     ax.scatter(y_pred, y_target, label='Adjusted data')
    #     ax.plot(y_pred, line, c='r', label='Fit: y=' + str(w) + 'x')
    #     ax.set_xlabel('Model output')
    #     ax.set_ylabel('CFD output')
    #     # ax.set_title(str(i))
    #     ax.legend()
    plt.subplot()
    y_pred = np.stack(y_pred).reshape(-1, 1)
    y_target = np.stack(y_target).reshape(-1, 1)
    w, line = fit_performance(y_pred, y_target)
    plt.scatter(y_pred, y_target, label='Adjusted data')
    plt.plot(y_pred, line, c='r', label='Fit: y=' + str(w) + 'x')
    plt.xlabel('Model output')
    plt.ylabel('CFD output')
    # ax.set_title(str(i))
    plt.legend()

    plt.savefig(os.path.join('./result/train29/fit_figure',"test_performance_{}.png".format(tag)))

    plt.show()

def evaluation_index(y_pred, y_tar):
    y_pred_cat = y_pred[0].reshape(-1)
    y_target_cat = y_tar[0].reshape(-1)
    for j in range(1, len(y_pred)):
        y_pred_cat = np.concatenate((y_pred_cat, y_pred[j].reshape(-1)))
        y_target_cat = np.concatenate((y_target_cat, y_tar[j].reshape(-1)))

    rmse = compute_RMSE(y_pred_cat, y_target_cat)
    r = correlation_coefficient(y_pred_cat, y_target_cat)
    accuracy_rate = accuracy(y_pred_cat, y_target_cat)

    return rmse, r, accuracy_rate

def evaluation_mse(y_pred_list, y_target_list):

    pred_target_pairs = random_select(y_pred_list, y_target_list)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(0, 118, 0.2)

    for ax, data in zip(axes.flatten(), pred_target_pairs):
        ax.plot(t, data[0],'.-', label='y_pred')
        ax.plot(t, data[1], label='y_target')
        ax.set_xlabel('time/s')
        ax.set_ylabel('concentration (ppm)')

        ax.legend()

    plt.savefig(os.path.join('./result/train29', "mse_performance.png"))
    plt.show()

def random_select(y_pred_list, y_target_list):
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
    random_exp = 24
    extract_num = 4
    # the be choosed exp's y_pred
    y_pred_exp = np.stack(y_pred_list[random_exp])
    # the be choosed exp's y_target
    y_target_exp = np.stack(y_target_list[random_exp])
    points_num = y_pred_exp.shape[2]

    # random choice 5 points in target points
    # random_extraction_points = r.sample(list(range(points_num)), extract_num)
    random_extraction_points = [780, 740, 745, 746]
    print(random_extraction_points)

    y_pred_points = y_pred_exp[0][:, random_extraction_points]
    y_target_points = y_target_exp[0][:, random_extraction_points]
    for j in range(1, len(y_pred_exp)):
        y_pred_points = np.concatenate((y_pred_points,
            y_pred_exp[j][-1, random_extraction_points].reshape(1,-1)))
        y_target_points = np.concatenate((
            y_target_points,
            y_target_exp[j][-1, random_extraction_points].reshape(1,-1)))

    pred_target_pairs = [[y_pred_points[:, i], y_target_points[:, i]]
                             for i in range(extract_num)]

    return pred_target_pairs


def main():
    set_seeds(seed=1234, cuda=False)
    path = '../seq2seq/result/train29'
    y_tar_list = torch.load(os.path.join(path, 'y_tar_list'))
    y_pred_list = torch.load(os.path.join(path, 'y_pred_list'))
    # space_concentration_figure(y_pred_list, y_tar_list)
    # evaluation_fit_figure(y_pred_list, y_tar_list)
    # evaluation_mse(y_pred_list, y_tar_list)
    for i in range(len(y_tar_list)):
        evaluation_fit_figure(y_pred_list[i], y_tar_list[i], i)
        r,rmse,_ =evaluation_index(y_pred_list[i], y_tar_list[i])
        print(i, r, rmse)



    print('test')

if __name__ == '__main__':
    main()