import torch
import numpy as np
import torch.nn as nn
"""
src_len = 2560 #source length enc_input max sequence length
tgt_len = 6 #target length dec_input(=dec_output) max sequence length
#transformer parameters
d_model = 512 # embedding size
d_ff = 2048 #前馈神经网络的中间维度，基本上是512-2048+ReLU 然后 2048-512 最后还要转回到512的长度
d_k = d_v = 64 #k=q，v 的维度都是64
n_layers = 6 #Encoder 和 decoder的层数
n_heads = 8 #多头注意力机制的头数
batch_size = 36
"""

class Transformer(nn.Module):                               #------10-------
    def __init__(self):
        super(Transformer,self).__init__()
        d_model = 512
        d_k = d_v = 64
        n_heads = 8
        d_ff = 2048
        #第1层encoder
        self.W_Q_1 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_1 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_1 = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_1 = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln_1 = nn.LayerNorm(d_model)

        self.fc_P_1 = nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False)
        )
        self.ln_p_1 = nn.LayerNorm(d_model)


        # 第2层encoder
        self.W_Q_2 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_2 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_2 = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_2 = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln_2 = nn.LayerNorm(d_model)
        self.fc_P_2 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln_p_2 = nn.LayerNorm(d_model)
        # 第3层encoder
        self.W_Q_3 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_3 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_3 = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_3 = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln_3 = nn.LayerNorm(d_model)
        self.fc_P_3 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln_p_3 = nn.LayerNorm(d_model)
        # 第4层encoder
        self.W_Q_4 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_4 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_4 = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_4 = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln_4 = nn.LayerNorm(d_model)
        self.fc_P_4 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln_p_4 = nn.LayerNorm(d_model)
        # 第5层encoder
        self.W_Q_5 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_5 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_5 = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_5 = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln_5 = nn.LayerNorm(d_model)
        self.fc_P_5 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln_p_5 = nn.LayerNorm(d_model)
        # 第6层encoder
        self.W_Q_6 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K_6 = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V_6 = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc_6 = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ln_6 = nn.LayerNorm(d_model)
        self.fc_P_6 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.ln_p_6 = nn.LayerNorm(d_model)
        #全连接层和softmax层
        self.projection = nn.Linear(2560, 6, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,enc_inputs):
        #超参数设置
        d_k = d_v =64
        n_heads = 8
        batch_size = 36
        enc_outputs = enc_inputs.view(36, 5, 512)  # 把原来[36,2560]的数据变成[36,5,512]的数据
        # 获得掩码矩阵
        #seq_k = torch.rand(36, 5)
        seq_k = torch.rand(36, 5).to(enc_inputs.device)
        # batch_size和句子长度
        # 将有2的地方填充掩码
        pad_attn_mask = seq_k.data.eq(2).unsqueeze(1)  # [batch_size,1,len_k],False is masked
        enc_self_attn_mask = pad_attn_mask.expand(36, 5, 5)
        enc_self_attns = []
        enc_self_attn_mask = enc_self_attn_mask
        #第1层encoder
        residual_1 = enc_outputs
        Q_1 = self.W_Q_1(enc_outputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q:[batch_size,n_heads,len_q,d_k]
        K_1 = self.W_K_1(enc_outputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K:[batch_size,n_heads,len_k,d_k]
        V_1 = self.W_V_1(enc_outputs).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V:[batch_size,n_heads,len_v(=len_k),d_v]
        attn_mask_1 = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask:[batch_size,n_heads,seq_len,seq_len]

        scores_1 = torch.matmul(Q_1, K_1.transpose(-1, -2)) / np.sqrt(d_k)
        scores_1.masked_fill_(attn_mask_1, -1e9)
        attn_1 = nn.Softmax(dim=-1)(scores_1)
        context_1 = torch.matmul(attn_1, V_1)

        context_1 = context_1.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output_1 = self.fc_1(context_1)  # [batch_size,len_q,d_model]

        enc_outputs_1_1 = self.ln_1(output_1 + residual_1)

        enc_self_attn_1 = attn_1
        #前馈神经网络MLP加resnet
        residual_p = enc_outputs_1_1
        output_p = self.fc_P_1(enc_outputs_1_1)
        enc_outputs_1_2 = self.ln_p_1(output_p + residual_p)

        enc_self_attns.append(enc_self_attn_1)
        # 第2层encoder
        residual_2 = enc_outputs_1_2
        Q_2 = self.W_Q_2(enc_outputs_1_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                 2)  # Q:[batch_size,n_heads,len_q,d_k]
        K_2 = self.W_K_2(enc_outputs_1_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                 2)  # K:[batch_size,n_heads,len_k,d_k]
        V_2 = self.W_V_2(enc_outputs_1_2).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                                 2)  # V:[batch_size,n_heads,len_v(=len_k),d_v]
        attn_mask_2 = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                             1)  # attn_mask:[batch_size,n_heads,seq_len,seq_len]

        scores_2 = torch.matmul(Q_2, K_2.transpose(-1, -2)) / np.sqrt(d_k)
        scores_2.masked_fill_(attn_mask_2, -1e9)
        attn_2 = nn.Softmax(dim=-1)(scores_2)
        context_2 = torch.matmul(attn_2, V_2)

        context_2 = context_2.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output_2 = self.fc_2(context_2)  # [batch_size,len_q,d_model]
        enc_outputs_2_1 = self.ln_2(output_2 + residual_2)
        enc_self_attn_2 = attn_2
        residual_p = enc_outputs_2_1
        output_p = self.fc_P_2(enc_outputs_2_1)
        enc_outputs_2_2 = self.ln_p_2(output_p + residual_p)

        enc_self_attns.append(enc_self_attn_2)
        # 第3层encoder
        residual_3 = enc_outputs_2_2
        Q_3 = self.W_Q_3(enc_outputs_2_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # Q:[batch_size,n_heads,len_q,d_k]
        K_3 = self.W_K_3(enc_outputs_2_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # K:[batch_size,n_heads,len_k,d_k]
        V_3 = self.W_V_3(enc_outputs_2_2).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                                   2)  # V:[batch_size,n_heads,len_v(=len_k),d_v]
        attn_mask_3 = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                             1)  # attn_mask:[batch_size,n_heads,seq_len,seq_len]

        scores_3 = torch.matmul(Q_3, K_3.transpose(-1, -2)) / np.sqrt(d_k)
        scores_3.masked_fill_(attn_mask_3, -1e9)
        attn_3 = nn.Softmax(dim=-1)(scores_3)
        context_3 = torch.matmul(attn_3, V_3)

        context_3 = context_3.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output_3 = self.fc_3(context_3)  # [batch_size,len_q,d_model]
        enc_outputs_3_1 = self.ln_3(output_3 + residual_3)
        enc_self_attn_3 = attn_3


        residual_p = enc_outputs_3_1
        output_p = self.fc_P_3(enc_outputs_3_1)
        enc_outputs_3_2 = self.ln_p_3(output_p + residual_p)
        enc_self_attns.append(enc_self_attn_3)
        # 第4层encoder
        residual_4 = enc_outputs_3_2
        Q_4 = self.W_Q_4(enc_outputs_3_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # Q:[batch_size,n_heads,len_q,d_k]
        K_4 = self.W_K_4(enc_outputs_3_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # K:[batch_size,n_heads,len_k,d_k]
        V_4 = self.W_V_4(enc_outputs_3_2).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                                   2)  # V:[batch_size,n_heads,len_v(=len_k),d_v]
        attn_mask_4 = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                             1)  # attn_mask:[batch_size,n_heads,seq_len,seq_len]

        scores_4 = torch.matmul(Q_4, K_4.transpose(-1, -2)) / np.sqrt(d_k)
        scores_4.masked_fill_(attn_mask_4, -1e9)
        attn_4 = nn.Softmax(dim=-1)(scores_4)
        context_4 = torch.matmul(attn_4, V_4)

        context_4 = context_4.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output_4 = self.fc_4(context_4)  # [batch_size,len_q,d_model]
        enc_outputs_4_1 = self.ln_4(output_4 + residual_4)
        enc_self_attn_4 = attn_4


        residual_p = enc_outputs_4_1
        output_p = self.fc_P_4(enc_outputs_4_1)
        enc_outputs_4_2 = self.ln_p_4(output_p + residual_p)
        enc_self_attns.append(enc_self_attn_4)
        # 第5层encoder
        residual_5 = enc_outputs_4_2
        Q_5 = self.W_Q_5(enc_outputs_4_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # Q:[batch_size,n_heads,len_q,d_k]
        K_5 = self.W_K_5(enc_outputs_4_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # K:[batch_size,n_heads,len_k,d_k]
        V_5 = self.W_V_5(enc_outputs_4_2).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                                   2)  # V:[batch_size,n_heads,len_v(=len_k),d_v]
        attn_mask_5 = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                             1)  # attn_mask:[batch_size,n_heads,seq_len,seq_len]

        scores_5 = torch.matmul(Q_5, K_5.transpose(-1, -2)) / np.sqrt(d_k)
        scores_5.masked_fill_(attn_mask_5, -1e9)
        attn_5 = nn.Softmax(dim=-1)(scores_5)
        context_5 = torch.matmul(attn_5, V_5)

        context_5 = context_5.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output_5 = self.fc_5(context_5)  # [batch_size,len_q,d_model]
        enc_outputs_5_1 = self.ln_5(output_5 + residual_5)
        enc_self_attn_5 = attn_5


        residual_p = enc_outputs_5_1
        output_p = self.fc_P_5(enc_outputs_5_1)
        enc_outputs_5_2 = self.ln_p_5(output_p + residual_p)
        enc_self_attns.append(enc_self_attn_5)
        # 第6层encoder
        residual_6 = enc_outputs_5_2
        Q_6 = self.W_Q_6(enc_outputs_5_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # Q:[batch_size,n_heads,len_q,d_k]
        K_6 = self.W_K_6(enc_outputs_5_2).view(batch_size, -1, n_heads, d_k).transpose(1,
                                                                                   2)  # K:[batch_size,n_heads,len_k,d_k]
        V_6 = self.W_V_6(enc_outputs_5_2).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                                   2)  # V:[batch_size,n_heads,len_v(=len_k),d_v]
        attn_mask_6 = enc_self_attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                             1)  # attn_mask:[batch_size,n_heads,seq_len,seq_len]

        scores_6 = torch.matmul(Q_6, K_6.transpose(-1, -2)) / np.sqrt(d_k)
        scores_6.masked_fill_(attn_mask_6, -1e9)
        attn_6 = nn.Softmax(dim=-1)(scores_6)
        context_6 = torch.matmul(attn_6, V_6)

        context_6 = context_6.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output_6 = self.fc_6(context_6)  # [batch_size,len_q,d_model]
        enc_outputs_6_1 = self.ln_6(output_6 + residual_6)
        enc_self_attn_6 = attn_6

        residual_p = enc_outputs_6_1
        output_p = self.fc_P_6(enc_outputs_6_1)
        enc_outputs_6_2 = self.ln_p_6(output_p + residual_p)
        enc_self_attns.append(enc_self_attn_6)
        #全连接层
        enc_outputs_7 = enc_outputs_6_2.view(36,2560)
        dec_logits = self.projection(enc_outputs_7)#dec_logits:[batch_size,tgt_len,tgt_vocab_size]
        output = self.softmax(dec_logits)
        return enc_outputs_1_1,enc_outputs_1_2,enc_outputs_2_1,enc_outputs_2_2,enc_outputs_3_1,enc_outputs_3_2,enc_outputs_4_1,enc_outputs_4_2,enc_outputs_5_1,enc_outputs_5_2,enc_outputs_6_1,enc_outputs_6_2,enc_outputs_7,output,dec_logits
