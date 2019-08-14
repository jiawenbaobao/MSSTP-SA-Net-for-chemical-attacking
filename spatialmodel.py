import torch.nn as nn
import torch.nn.functional as F
import torch

class SpatialModel(nn.Module):
    def __init__(self, num_input_channels, out_num, dropout_p):
        super(SpatialModel, self).__init__()

        # Conv weights
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=5, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.dropout = nn.Dropout(dropout_p)

        self.fc3 = nn.Linear(1280,  out_num)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(128)


    def forward(self, z, channel_first=False, apply_softmax=False):

        # Rearrange input so num_input_channels is in dim 1 (N, C, L)
        # if not channel_first:
        #     x = x.transpose(1, 2)

        # Conv outputs
        z = F.relu(self.conv_bn1(self.conv1(z)))
        z = F.max_pool2d(z, 2)

        z = F.relu(self.conv_bn2(self.conv2(z)))
        z = F.max_pool2d(z, 2)

        z = self.dropout(z)

        # FC layer
        z = z.view(-1, self.num_flat_features(z))
        # z = self.fc1(z) #relu?
        # z = self.fc2(z)
        y_pred = self.fc3(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    model = SpatialModel(num_input_channels=5, out_num=1053, dropout_p=0.1)
    input_data = torch.randn(1, 5, 17, 31)
    model(input_data)


if __name__ == '__main__':

    main()