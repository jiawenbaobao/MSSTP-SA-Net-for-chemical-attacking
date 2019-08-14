import os
import math
import pickle

import pandas as pd
import numpy as np

from init_project import create_dirs, set_seeds, args_set


class DataPrepare(object):
    def __init__(self, save_dir, data_folder, save_sample_path, train_size,
                 val_size, test_size, input_seq_len, pred_seq_len, shuffle=True):
        self.save_dir = save_dir
        self.data_folder = data_folder
        self.save_sample_path = save_sample_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.shuffle = shuffle

        self.scales = []

        self.train_exps, self.val_exps, self.test_exps = self.exps_split()
        np.save('exp_list.npy', self.test_exps)

        self.find_max()
        np.save('scales.npy', self.scales)
        # print('scales: ', self.scales)

    def create_data(self):
        split(data_folder=self.data_folder, exps=self.train_exps,
              input_seq_len=self.input_seq_len, pred_seq_len=self.pred_seq_len,
              save_sample_path=self.save_sample_path, tag='train',
              scales=self.scales)

        split(data_folder=self.data_folder, exps=self.val_exps,
              input_seq_len=self.input_seq_len, pred_seq_len=self.pred_seq_len,
              save_sample_path=self.save_sample_path, tag='val',
              scales=self.scales)

    def exps_split(self):
        '''
        shuffle exp_list
        split input, space_target, time_target as train, val, test

        '''
        exp_list = os.listdir(self.data_folder)
        while '.DS_Store' in exp_list:
            exp_list.remove('.DS_Store')
        if self.shuffle:
            np.random.shuffle(exp_list)

        exp_length = len(exp_list)
        n_train = math.ceil(self.train_size * exp_length)  # 向上取整
        n_val = math.ceil(self.val_size * exp_length)

        train_exps = exp_list[:n_train]
        val_exps = exp_list[n_train:n_train + n_val]
        test_exps = exp_list[n_train + n_val:]

        return train_exps, val_exps, test_exps

    def find_max(self):
        '''
        data scaler: max_abs_scaler
        val/test_targets: square
        '''
        # caculate scales
        file_names = ['c_monitor.csv', 'fai.csv', 'T.csv',
                      'theta.csv', 'v.csv']

        for file in file_names:
            abs_max = 0
            for exp in self.train_exps:
                path = os.path.join(self.data_folder, exp, file)
                df_exp = pd.read_csv(path, header=None)
                abs_max = max(np.max(df_exp.iloc[:,:-1].values), abs_max)
            self.scales.append(abs_max)

    def test_data_prepare(self):
        for exp in self.test_exps:
            path = os.path.join(self.data_folder, exp)
            exp_inputs, exp_space, exp_time = \
                data_classify(path, self.input_seq_len,
                              self.pred_seq_len, tag='test')

            exp_inputs = np.stack(exp_inputs, axis=0)
            exp_time = np.stack(exp_time, axis=0)
            # data transform
            for i in range(len(self.scales)):
                exp_inputs[:, :, i, ...] /= self.scales[i]

            exp_time = pow(exp_time, 2)

            self.save_data(exp, exp_inputs, exp_time)

    def save_data(self, exp, exp_inputs, exp_time):

        inputs_path = os.path.join(self.save_sample_path, 'test', exp, 'inputs')
        target_time_path = os.path.join(self.save_sample_path, 'test', exp,
                                        'target_time')

        create_dirs(inputs_path)
        create_dirs(target_time_path)

        for j in range(len(exp_inputs)):
            sample_name = 'sample_' + str(j) + '.pkl'

            with open(os.path.join(inputs_path, sample_name), 'wb') as f:
                pickle.dump(exp_inputs[j], f)

            with open(os.path.join(target_time_path, sample_name), 'wb') as f:
                pickle.dump(exp_time[j], f)


def data_transform(inputs, scales):
    file_names = ['c_monitor.csv', 'fai.csv', 'T.csv',
                  'theta.csv', 'v.csv']
    for i in range(len(file_names)-1):
        inputs[:, :, i, ...] /= scales[i]

    return inputs


def split(data_folder, exps, input_seq_len, pred_seq_len, save_sample_path,
          scales, tag):
    '''
    concat every exp which is arranged as input, target_space,
    target_time respectively
    '''

    inputs_path = os.path.join(save_sample_path, tag, 'inputs')
    target_space_path = os.path.join(save_sample_path, tag, 'target_space')
    target_time_path = os.path.join(save_sample_path, tag, 'target_time')

    create_dirs(inputs_path)
    create_dirs(target_space_path)
    create_dirs(target_time_path)

    i = 0
    for exp in exps:
        path = os.path.join(data_folder, exp)
        inputs, target_space, target_time \
            = data_classify(path, input_seq_len, pred_seq_len, tag)
        inputs = data_transform(inputs, scales)
        for j in range(len(inputs)):
            sample_name = 'sample_'+str(j+i)+'.pkl'

            with open(os.path.join(inputs_path,sample_name), 'wb') as f:
                pickle.dump(inputs[j], f)

            with open(os.path.join(target_space_path, sample_name), 'wb') as f:
                pickle.dump(target_space[j], f)

            with open(os.path.join(target_time_path, sample_name), 'wb') as f:
                pickle.dump(target_time[j], f)

        i = i + len(inputs)


def data_classify(path, input_seq_len, pred_seq_len, tag):
    '''
    To every exp, as input_seq_len=10, pred_seq_len=5 for an example:
    row_targets/row_inputs/test_inputs/test_space_targets = [
    [1,2,3...,10],
    [2,3,4,...11],
    ...
    [586,..,594,595]

    test_time=
    [
    [11,12,..15],
    [12,13,..16],
    ..
    [596,597,..600]
    ]

    train/val inputs space_targets = [
    [1,2,3...,10],
    [11,12,...20],
    ..

    ]
    train/val time_targets =[
    [11,12,..15],
    [21,22,..25],
    ...

    ]
    '''
    file_names = ['c_monitor.csv', 'c_tar.csv', 'fai.csv', 'T.csv', 'theta.csv'
                  , 'v.csv']

    while '.DS_Store' in file_names:
        file_names.remove('.DS_Store')
    df_file = pd.read_csv(os.path.join(path, 'T.csv'), header=None)
    times = list(map(int, set(df_file.values[:, -1])))

    tar_file_name = 'c_tar.csv'
    assert tar_file_name in file_names
    tar_df = pd.read_csv(os.path.join(path, tar_file_name), header=None)
    last_columns = tar_df[tar_df.columns[-1]]

    # targets
    row_targets = [np.float32(tar_df[last_columns == time].iloc[:, :-1].values)
                   for time in times]
    # row_targets = [tar_df[last_columns == time].values for time in times]

    # inputs
    file_names.remove('c_tar.csv')
    dfs = [pd.read_csv(os.path.join(path, file), header=None)
           for file in file_names]
    row_inputs = []
    for time in times:
        cur_data = []
        for df in dfs:
            last_columns = df[df.columns[-1]]
            df = df[last_columns == time].iloc[:, :-1]
            # df = df[last_columns == time]
            cur_data.append(np.float32(df.values))
        row_inputs.append(np.stack(cur_data))

    inputs_ = [np.stack(row_inputs[i: i + input_seq_len])
               for i in range((len(times) - input_seq_len) + 1)]

    targets_ = [np.stack(row_targets[i: i + input_seq_len])
                for i in range((len(times) - input_seq_len) + 1)]

    if tag == 'test':
        for _ in range(pred_seq_len):
            inputs_.pop()
            targets_.pop()
        inputs = inputs_
        y_space = targets_

        for i in range(input_seq_len):
            row_targets.pop(0)
        y_time = [row_targets[i: i + pred_seq_len]
                  for i in range(len(row_targets) - pred_seq_len + 1)]

    elif tag == 'train' or tag == 'val':

        for _ in range(pred_seq_len):
            inputs_.pop()
            targets_.pop()

        # create blocks
        blocks_x = [inputs_[i::input_seq_len] for i in range(input_seq_len)]
        blocks_y = [targets_[i::input_seq_len] for i in range(input_seq_len)]

        # inputs
        inputs = sum(blocks_x, [])

        # space_targets
        y_space = sum(blocks_y, [])

        # time_targets
        for _ in range(input_seq_len):
            row_targets.pop(0)
        y_time_ = [np.stack(row_targets[i:i+pred_seq_len]) for i in
                range(len(row_targets)-pred_seq_len+1)]
        blocks_y_time = [y_time_[i::input_seq_len] for i in range(input_seq_len)]
        y_time = sum(blocks_y_time, [])
    else:
        raise Exception('no tag!')

    return np.stack(inputs), np.stack(y_space), np.stack(y_time)


def main():
    args = args_set('big')
    set_seeds(seed=1234, cuda=args.cuda)
    data_prepare = DataPrepare(data_folder=args.data_folder,
                               save_dir=args.save_dir,
                               save_sample_path=args.save_sample_path,
                               train_size=args.train_size,
                               val_size=args.val_size,
                               test_size=args.test_size,
                               input_seq_len=args.input_seq_len,
                               pred_seq_len=args.pred_seq_len, shuffle = True)
    print(data_prepare.test_exps)
    print(data_prepare.scales)
    # data_prepare.create_data()
    # data_prepare.test_data_prepare()
    # input_train_data = np.load(save_root+'/test_data/input_data/input_train_data.npy')

if __name__ == '__main__':
    main()