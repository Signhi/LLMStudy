import torch

feature_dim = 50
seq_len = 10
epsilon = 1e-8




source_seq = torch.randn(seq_len, feature_dim)
padding_seq = torch.zeros(1, feature_dim)
label_seq = torch.concat((source_seq[1:, :], padding_seq))


class selfMultiHeadAttention(torch.nn.Module):
    def __init__(self, feature_dim, head_num):
        super(selfMultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.per_head_dim = feature_dim // head_num
        self.W_O = torch.nn.Linear(feature_dim, feature_dim)
        self.W_Q = []
        self.W_K = []
        self.W_V = []
        for ii in range(head_num):
            self.W_Q.append(torch.nn.Linear(self.per_head_dim, self.per_head_dim))
            self.W_K.append(torch.nn.Linear(self.per_head_dim, self.per_head_dim))
            self.W_V.append(torch.nn.Linear(self.per_head_dim, self.per_head_dim))
    def forward(self, x):
        multi_x = [x[:, i*self.per_head_dim:(i+1)*self.per_head_dim] for i in range(self.head_num)]
        attention_score = []
        
        for ii in range(self.head_num):
            Q = self.W_Q[ii](multi_x[ii])
            K = self.W_K[ii](multi_x[ii])
            V = self.W_V[ii](multi_x[ii])
            temp_output = torch.matmul(Q[ii], K[ii].T) / torch.sqrt(self.per_head_dim)
            temp_output = torch.softmax(temp_output, dim=1)
            temp_output = torch.matmul(temp_output, V[ii])
            attention_score.append(temp_output)
        attention_score = torch.cat(attention_score, dim=1)
        output = self.W_O(attention_score)
        return output


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
    self_multi_head_attention = selfMultiHeadAttention(feature_dim)
    temp_seq = rms_norm(source_seq)
    output_seq = self_multi_head_attention.forward(temp_seq)
    output_seq += source_seq
    fully_connected_net = FCN(feature_dim, feature_dim)
    for index in range(seq_len):
        output_seq[index] = fully_connected_net.forward(output_seq[index])
    output_seq = torch.softmax(output_seq, dim=1)
    return output_seq
   
   