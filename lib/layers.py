import torch.nn as nn
import torch
from torch.nn.modules import RNN
import math


class ConstantPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(ConstantPositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.shape[1],:]

class Module(nn.Module):
    '''
        Wraper to extend functions of models.
    '''
    def __init__(self):
        super().__init__()

    def count_parameters(self):
        param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {param:,} trainable parameters')


class RNN(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                 activation='linear',
                 return_hidden=False
                ):

        super().__init__()
        
        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.hidden_ff = nn.Linear(hid_dim,hid_dim)
        self.output_ff = nn.Linear(hid_dim, output_dim)
        if activation == 'linear':
            self.activation = nn.Identity()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise Exception("Uknow actication type")
        self.hid_dim = hid_dim
        self.return_hidden = return_hidden
    def forward(self, x, initial_hidden=None):
        
        #src = [batch size, input len, input dim]
        length = x.shape[1]
        batch_size = x.shape[0]

        hidden = []
        # Initial hidden state
        if initial_hidden is None:
            hidden.append(torch.zeros(batch_size, 1, self.hid_dim, dtype=x.dtype, device=x.device))
        else:
            hidden.append(initial_hidden)

        # input mapping
        x = self.input_ff(x)

        # recurrent relation
        for i in range(length):
            h_next = self.activation(x[:,i:i+1,:] + self.hidden_ff(hidden[i]))
            hidden.append(h_next)

        # Convert all hidden into a tensor
        hidden = torch.cat(hidden[1:], dim=1)

        # output mapping
        out = self.output_ff(hidden)

        if self.return_hidden:
            return out, hidden
        return out
    
class DCN(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                 kernel_size,
                 num_layers,
                 activation='linear'
                ):

        super().__init__()
        
        self.conv_layers = nn.ModuleList([nn.Conv1d(hid_dim, 
                                                    hid_dim,
                                                    kernel_size,
                                                    padding=(kernel_size-1)*(kernel_size**i),
                                                    dilation=kernel_size**i,
                                                    bias=False) for i in range(num_layers)])

        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.output_ff = nn.Linear(hid_dim, output_dim)

        if activation == 'linear':
            self.activation = nn.Identity()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise Exception("Uknow actication type")

     

    def forward(self, x):
        
        #src = [batch size, input len, input dim]
        length = x.shape[1]
        x = self.input_ff(x)

        x = x.permute(0,2,1)

        for layer in self.conv_layers:
            x = x + self.activation(layer(x)[:,:,:length])
       
        x = x.permute(0,2,1)

        y = self.activation(self.output_ff(x))
        return y

class PositionwiseFeedforwardLayer(Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class EncoderLayer(Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, head_dim=None):

        super().__init__()
        

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src,pos_embed, x, param):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        
        _src, attention = self.self_attention(x, pos_embed, src, param)

        src = self.self_attn_layer_norm(src + self.dropout1(_src))

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout2(_src))
        

        #src = [batch size, src len, hid dim]
        
        return src,attention
    
class Encoder(Module):
    r"""
    The main part of the transformer encoder code is taken from 
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
    """
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100,
                 pos_embed = False, pos_encoding_dim = 0, head_dim=None, param='full',
                 pos_embed_scale=1, input_embed_scale=1):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.device = device

        self.input_dim = input_dim
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_embed = pos_embed
        # pos_encoding_dim is the dimension in data which is position information.
        self.param = param

        self.input_embedding = nn.Linear(input_dim + pos_encoding_dim, hid_dim)
        self.input_embedding.bias.data = self.input_embedding.bias.data*input_embed_scale
        self.input_embedding.weight.data = self.input_embedding.weight.data*input_embed_scale



        if pos_embed:
            # self.pos_embedding = nn.Embedding(max_length, hid_dim)
            pos = torch.nn.init.xavier_normal_(torch.zeros(1, max_length, hid_dim))*pos_embed_scale
            self.pos_embedding = nn.Parameter(pos)

        self.out_fc = nn.Linear(hid_dim, output_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, head_dim=head_dim) 
                                     for _ in range(n_layers)])

        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        # self.init_weights()
        # self.initialize_weights()

    def init_weights(self):
        initrange = 0.1    
        self.out_fc.bias.data.zero_()
        self.out_fc.weight.data.uniform_(-initrange, initrange)



    def forward(self, src):

        #src = [batch size, src len]
        
        if self.pos_embed:
            x = self.input_embedding(src)    # Transformed input 
            pos_embed = self.pos_embedding + torch.zeros_like(x) # match pos_embedding to x shape
            src = self.dropout(( x * self.scale)+pos_embed) # x with pos_embed
        else:
            src = self.input_embedding(src) 
            raise Exception
        #src = [batch size, src len, hid dim]

        # x is input without pos 

        for layer in self.layers:
            src, attention= layer(src,pos_embed,x, param=self.param)
        
        output = self.out_fc(src)

        #src = [batch size, src len, hid dim]
            
        return output , attention

    def initialize_weights(self):
        def _initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(_initialize_weights)

class MultiHeadAttentionLayer(Module):
    def __init__(self, hid_dim, n_heads, dropout, bias = True):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.fc_k = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.fc_v = nn.Linear(hid_dim, hid_dim, bias=bias)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, pos,src,param):
        
        batch_size = x.shape[0]
        

        if param == 'full':
            query = src
            key = src
            value = src
        elif param == 'pos':
            query = pos
            key = pos
            value = src
        elif param == 'query_pos':
            query = pos
            key = src
            value = src
        else:
            raise Exception
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]

        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class EncDec(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                 output_len,
                 activation='linear'
                ):

        super().__init__()
        self.encoder = RNN(input_dim,output_dim, hid_dim, activation, return_hidden=True)
        self.decoder = RNN(input_dim,output_dim, hid_dim, activation)
        self.out_len = output_len
    def forward(self, x):
        _, context = self.encoder(x)
        context = context[:,-2:-1,:]
        batch_size = x.shape[0]
        decoder_input_pad = torch.zeros(batch_size,self.out_len,x.shape[-1], dtype=x.dtype, device=x.device)

        y = self.decoder(decoder_input_pad, context)

        return y