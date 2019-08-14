import os
import pickle

from torch.utils.data import Dataset, DataLoader

from dataprepare_new import DataPrepare
from train import collate_fn


class SpatialTimeDataset(Dataset):
    def __init__(self, save_sample_path):
        self.save_sample_path = save_sample_path
        self.set_split('train') # 将'train'设置为target_df


    def set_split(self, split="train"):
        self.target_split = split
        self.input_sample_path = os.path.join(self.save_sample_path, split,
                                              'inputs')
        self.target_space_sample_path = os.path.join(self.save_sample_path, split,
                                              'target_space')
        self.target_time_sample_path = os.path.join(self.save_sample_path, split,
                                              'target_time')
        self.sample_num = len(os.listdir(self.input_sample_path))

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(
            self.target_split, self.sample_num)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        input_path = os.path.join(self.input_sample_path,
                                  'sample_{0}.pkl'.format(str(index)))
        target_space_path = os.path.join(self.target_space_sample_path,
                                         'sample_{0}.pkl'.format(str(index)))
        target_time_path = os.path.join(self.target_time_sample_path,
                                        'sample_{0}.pkl'.format(str(index)))
        with open(input_path, 'rb') as f:
            input_sample = pickle.load(f)

        with open(target_space_path, 'rb') as f:
            target_space_sample = pickle.load(f)

        with open(target_time_path, 'rb') as f:
            target_time_sample = pickle.load(f)

        row = (input_sample, target_space_sample, target_time_sample)
        return row

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self, batch_size, collate_fn, shuffle, device,
                         drop_last=True):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                collate_fn=collate_fn, shuffle=shuffle,
                                drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict

def main():
    root = "/home/zhyhou/xjw/processed_data"
    data_prepare = DataPrepare(data_folder=root,
                               save_dir="./result/checkpoint_test",
                               save_sample_path='/home/zhyhou/xjw/sample_test',
                               train_size=0.2, val_size=0.2,
                               test_size=0.2, input_seq_len=10,
                               pred_seq_len=5, shuffle = True)

    # input_train_data = np.load(save_root+'/test_data/input_data/input_train_data.npy')
    dataset = SpatialTimeDataset('/home/zhyhou/xjw/sample_test')
    batch_generator_train = dataset.generate_batches(
        batch_size=4, collate_fn=collate_fn,
        shuffle=True, device='cuda')
    for batch_index, batch_dict in enumerate(batch_generator_train):
        print('test')

    sample = dataset[0]
if __name__ == '__main__':
    main()