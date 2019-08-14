import torch.nn as nn
import torch.nn.functional as F
import torch

class Seq2seq_mlp(nn.Module):
    def __init__(self, num_features, input_seq_len, pred_seq_len, batch_size,
                 device):
        super(Seq2seq_mlp, self).__init__()
        self.layer = 1
        self.input_seq_len = input_seq_len
        self.batch_size = batch_size
        self.hidden_size = 512

        self.encoder = EncoderModel(num_features=num_features,
                                    hidden_size=self.hidden_size,
                                    layer=self.layer,
                                    device=device
                                    )

        self.decoder = DecoderModel(pred_seq_len=pred_seq_len,
                                    num_features=num_features,
                                    hidden_size=self.hidden_size,
                                    batch_size=batch_size,
                                    device=device)

    def forward(self, z_in, z_tar,):
        z_in = z_in.view(int(z_in.shape[0] / self.input_seq_len),
                         self.input_seq_len, -1)

        for i in range(self.input_seq_len):
            encoder_out, encoder_hidden = self.encoder(z_in[:, i, :].unsqueeze(1))

        decoder_out = self.decoder(encoder_hidden)

        return decoder_out


# Model with batch normalization
class EncoderModel(nn.Module):
    def __init__(self, num_features, hidden_size, layer, device):
        super(EncoderModel, self).__init__()
        self.device = device

        self.gru = nn.GRU(num_features, hidden_size, layer,batch_first=True)
        self.init_(self.gru)

        self.fc = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, z, channel_first=False, apply_softmax=False):

        z, hn = self.gru(z)
        z = F.relu(z)
        hn_trans = torch.zeros(hn.shape, device=torch.device(self.device))
        for i in range(hn.shape[0]):
            hn_trans[i] = self.fc(hn[i])

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
    def __init__(self, pred_seq_len, num_features, hidden_size,
                 batch_size, device):
        super(DecoderModel, self).__init__()
        self.device = device
        self.pred_seq_len = pred_seq_len
        self.batch_size = batch_size
        self.num_features = num_features

        self.embedding = nn.Embedding(pred_seq_len, hidden_size)

        self.fc_global = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.fc_global.weight)

        self.fc_local = nn.Linear(hidden_size * 3, num_features)
        nn.init.xavier_uniform_(self.fc_local.weight)

    def forward(self, encoder_hidden):

        # embed_input
        embed_input = torch.zeros(self.batch_size, self.pred_seq_len + 1,
                                  encoder_hidden.shape[2],
                                  device=torch.device(self.device))

        for i in range(self.pred_seq_len):
            embed_input[:, i, :] = self.embedding(
                (torch.zeros([self.batch_size]) + i).long().to(self.device))
        embed_input[:, -1, :] = encoder_hidden.squeeze()

        # global calculation
        global_info = self.fc_global(embed_input)
        ca = global_info[:, -1, :]

        # local calculation
        out = torch.zeros(self.batch_size, self.pred_seq_len, self.num_features,
                          device=torch.device(self.device))

        for i in range(self.pred_seq_len):
            x = torch.cat((global_info[:, i, :], embed_input[:, i, :], ca),
                          dim=1)

            out[:, i, :] = self.fc_local(x)

        return out



def main():
    # CUDA_LAUNCH_BLOCKING = 1
    device = 'cuda'
    input_data = torch.randn(16*8, 1053)
    input_data = input_data.to(device)
    output_data = torch.randn(16, 4, 1053)
    output_data = output_data.to(device)
    input_seq_len = 8
    pred_seq_len=4

    model = Seq2seq_mlp(num_features=1053,
                    input_seq_len=input_seq_len, pred_seq_len=pred_seq_len,
                    batch_size=16, device=device)
    model = model.to(device)
    model(input_data, z_tar=output_data)


if __name__ == '__main__':
    main()
