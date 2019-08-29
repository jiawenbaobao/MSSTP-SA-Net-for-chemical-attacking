import torch.nn as nn
import torch.nn.functional as F
import torch

class Seq2seq_new(nn.Module):
    def __init__(self, num_features, hidden_size, input_seq_len, pred_seq_len,
                 batch_size):
        super(Seq2seq_new, self).__init__()
        self.layer = 1
        self.num_features = num_features
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.batch_size = batch_size

        self.encoder = EncoderModel(num_features=num_features,
                                    hidden_size=hidden_size, layer=self.layer,
                                    seq_len=input_seq_len)
        self.decoder = DecoderModel(num_input_channels=input_seq_len+1,
                                    batch_size=batch_size,
                                    pred_seq_len=pred_seq_len, out_num=1053,
                                    dropout_p=0.5)

    def forward(self, z_in, z_tar, device):
        z_in = z_in.view(int(z_in.shape[0] / self.input_seq_len),
                         self.input_seq_len, -1)

        for i in range(self.input_seq_len):
            # out, encoder_hidden = self.encoder(z_in[:, self.input_seq_len-1-i, :]
            #                                    .unsqueeze(1))
            out, encoder_hidden = self.encoder(z_in[:, i, :].unsqueeze(1),
                                               device)

        # decoder_input = torch.zeros(self.batch_size, 1, self.num_features,
        #                             device=torch.device(device))
        # decoder_input = z_in[:,-1,:].unsqueeze(1)


        # decoder_hidden = encoder_hidden
        encoder_input = torch.cat((z_in, encoder_hidden.squeeze(0).unsqueeze(1))
                                  ,dim=1)

        pred_out = self.decoder(encoder_input.unsqueeze(2))

        # if use_teacher_forcing:
        #     for i in range(self.pred_seq_len):
        #         decoder_output, decoder_hidden = self.decoder(decoder_input,
        #                                                        decoder_hidden)
        #         decoder_input = z_tar[:, i, :].unsqueeze(1)
        #         pred_out.append(decoder_output.squeeze(1))
        #
        # else:
        #     for i in range(self.pred_seq_len):
        #         decoder_output, decoder_hidden = self.decoder(decoder_input,
        #                                                        decoder_hidden)
        #         decoder_input = decoder_output
        #         pred_out.append(decoder_output.squeeze(1))

        return pred_out

# Model with batch normalization
class EncoderModel(nn.Module):
    def __init__(self, num_features, hidden_size, layer, seq_len):
        super(EncoderModel, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.gru = nn.GRU(num_features,hidden_size, layer,batch_first=True)
        self.init_(self.gru)
        self.fc = nn.Linear(hidden_size, num_features)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, z, device, channel_first=False, apply_softmax=False):

        z, hn = self.gru(z)
        z = F.relu(z)
        hn_trans = torch.zeros(hn.shape[0], hn.shape[1], self.num_features,
                               device=torch.device(device))
        for i in range(hn.shape[0]):
            # hn_trans[i] = self.fc(hn[i])
            hn_trans[i] = F.relu(self.fc(hn[i]))

        return z, hn_trans

    def init_(self, model):
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DecoderModel(nn.Module):
    def __init__(self, num_input_channels, batch_size, pred_seq_len, out_num,
                 dropout_p):
        super(DecoderModel, self).__init__()

        self.batch_size = batch_size
        self.pred_seq_len = pred_seq_len
        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=5, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.dropout = nn.Dropout(dropout_p)

        self.fc3 = nn.Linear(64*65, 5120)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.fc4 = nn.Linear(5120, out_num * self.pred_seq_len)
        nn.init.xavier_uniform_(self.fc4.weight)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(64)

    def forward(self, z, channel_first=False, apply_softmax=False):

        # Rearrange input so num_input_channels is in dim 1 (N, C, L)
        # if not channel_first:
        #     x = x.transpose(1, 2)
        # Conv outputs
        z = F.relu(self.conv_bn1(self.conv1(z)))
        z = F.max_pool2d(z, 4)

        z = F.relu(self.conv_bn2(self.conv2(z)))
        z = F.max_pool2d(z, 4)

        z = self.dropout(z)

        # FC layer
        z = z.view(-1, self.num_flat_features(z))
        # z = self.fc1(z) #relu?
        # z = self.fc2(z)
        z = self.fc3(z)
        y_pred = self.fc4(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred.reshape(self.batch_size, self.pred_seq_len, 1053)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def main():

    input_data = torch.randn(16*8, 1053)
    input_data = input_data.to('cuda')
    output_data = torch.randn(16, 4, 1053)
    output_data = output_data.to('cuda')
    input_seq_len = 8
    pred_seq_len=4

    model = Seq2seq_new(num_features=1053, hidden_size=512,
                    input_seq_len=input_seq_len, pred_seq_len=pred_seq_len,
                    batch_size=16)
    model = model.to('cuda')
    model(input_data, z_tar=output_data,device='cuda')


if __name__ == '__main__':

    main()