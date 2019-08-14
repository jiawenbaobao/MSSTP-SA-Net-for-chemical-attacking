import torch.nn as nn
import torch.nn.functional as F
import torch

class Seq2seq_attn(nn.Module):
    def __init__(self, num_features, input_seq_len, pred_seq_len
                 , batch_size, dropout):
        super(Seq2seq_attn, self).__init__()
        self.encoder = Encoder_atten(num_features, hidden_size=256,
                                     seq_len=input_seq_len)
        self.decoder = Decoder_atten(num_features, input_seq_len, dropout)
        self.num_features = num_features
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.batch_size = batch_size

    def forward(self, z_in, z_tar, device, use_teacher_forcing):
        z_in = z_in.view(int(z_in.shape[0] / self.input_seq_len),
                         self.input_seq_len, -1)

        encoder_outputs = torch.zeros(self.batch_size, self.input_seq_len,
                                      self.encoder.num_features, device=device)
        for i in range(self.input_seq_len):
            out, encoder_hidden = self.encoder(z_in[:, i, :].unsqueeze(1))
            encoder_outputs[:,i] = out[:, 0]

        decoder_input = torch.zeros(self.batch_size, 1, self.num_features,
                                    device=torch.device(device))
        decoder_hidden = encoder_hidden.unsqueeze(0)

        pred_out = []
        if use_teacher_forcing:
            for i in range(self.pred_seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden,
                                                              encoder_outputs)
                decoder_input = z_tar[:, i, :].unsqueeze(1)
                pred_out.append(decoder_output.squeeze(1))

        else:
            for i in range(self.pred_seq_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden,
                                                              encoder_outputs)
                decoder_input = decoder_output
                pred_out.append(decoder_output.squeeze(1))

        return torch.stack(pred_out, dim=1)

class Encoder_atten(nn.Module):
    def __init__(self, num_features, hidden_size, seq_len):
        super(Encoder_atten, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.gru = nn.GRU(num_features, hidden_size, batch_first=True)
        self.init_(self.gru)
        self.fc = nn.Linear(hidden_size, num_features)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, z, channel_first=False, apply_softmax=False):

        z, hn = self.gru(z)
        z = F.relu(self.fc(z))
        hn = self.fc(hn[0])

        return z, hn

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


class Decoder_atten(nn.Module):
    def __init__(self, num_features, input_len_seq, dropout):
        super(Decoder_atten, self).__init__()
        self.gru = nn.GRU(num_features, num_features, batch_first=True)
        self.init_(self.gru)

        self.attn = nn.Linear(num_features*2, input_len_seq)
        nn.init.xavier_uniform_(self.attn.weight)

        self.attn_combine = nn.Linear(num_features*2, num_features)
        nn.init.xavier_uniform_(self.attn_combine.weight)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, hidden, encoder_outputs,
                channel_first=False, apply_softmax=False):
        attn_weights = torch.tanh(
            self.attn(torch.cat((z[:,0,...], hidden[0]), 1))
        )
        atten_applied = torch.bmm(attn_weights.unsqueeze(1),
                                  encoder_outputs)
        output = torch.cat((z.squeeze(1), atten_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hn = self.gru(output, hidden)

        return output, hn

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

def main():

    input_data = torch.randn(16*8, 1053)
    input_data = input_data.to('cuda')
    output_data = torch.randn(16, 4, 1053)
    output_data = output_data.to('cuda')
    input_seq_len = 8
    pred_seq_len=4

    model = Seq2seq_attn(num_features=1053,
                    input_seq_len=input_seq_len, pred_seq_len=pred_seq_len,
                    batch_size=16, dropout=0.5)
    model = model.to('cuda')
    model(input_data, z_tar=output_data,device='cuda', use_teacher_forcing=False)


if __name__ == '__main__':

    main()