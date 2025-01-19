import torch

feature_dim = 50
seq_len = 10
epsilon = 1e-8




source_seq = torch.randn(seq_len, feature_dim)
padding_seq = torch.zeros(1, feature_dim)
label_seq = torch.concat((source_seq[1:, :], padding_seq))


class selfAttention(torch.nn.Module):
    def __init__(self, feature_dim):
        super(selfAttention, self).__init__()
        self.W_Q = torch.nn.Linear(feature_dim, feature_dim)
        self.W_K = torch.nn.Linear(feature_dim, feature_dim) 
        self.W_V = torch.nn.Linear(feature_dim, feature_dim)
        self.W_O = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attention_score = torch.matmul(Q, K.T) / torch.sqrt(feature_dim)
        attention_score = torch.softmax(attention_score, dim=1)
        output_seq = torch.matmul(attention_score, V)
        output_seq = self.W_O(output_seq)
        return output_seq

def rms_norm(source_seq):
    return source_seq / (torch.sqrt(torch.sum(source_seq ** 2, dim=1, keepdim=True)) + epsilon)

class FCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

def tranfromer(source_seq):
    self_attention = selfAttention(feature_dim)
    temp_seq = rms_norm(source_seq)
    output_seq = self_attention.forward(temp_seq)
    output_seq += source_seq
    temp_seq = output_seq.copy()
    fully_connected_net = FCN(feature_dim, feature_dim)
    for index in range(seq_len):
        output_seq[index] = fully_connected_net.forward(output_seq[index])
    output_seq = torch.softmax(output_seq, dim=1)
    output_seq += temp_seq
    return output_seq
   
   