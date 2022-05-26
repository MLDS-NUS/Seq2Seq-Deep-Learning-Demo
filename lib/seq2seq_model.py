import pytorch_lightning as pl
import torch
from torch import nn
from scipy.special import softmax
import os
from lib.tcn import TemporalConvNet
from lib.layers import ConstantPositionalEncoding
import numpy as np
from torch.utils.data import TensorDataset

from ml_collections import FrozenConfigDict
CONFIG = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                                NUM = 20,
                                                SHIFT = 30),
                            'convo': dict(LENGTH = 100,
                                                NUM = 20,
                                                FILTER = [0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002]),
                                'lorenz': dict(NUM = 10, 
                                                K=1, J=10, 
                                                LENGTH=32 ), 'train_size':49500, 'valid_size': 500})


class ShiftDataset(torch.utils.data.Dataset):

    def __init__(self,size, seq_len,shift, dtype=torch.float32):
        input = []
        output = []
        for _ in range(size):
            data = self._generate_gaussian(seq_len)
            input.append(data)
            output.append(np.concatenate((np.zeros(shift), data[:-shift])))

        input = np.array(input)
        output = np.array(output)
        self.X = torch.tensor(input, dtype=dtype).unsqueeze(-1)
        self.Y = torch.tensor(output, dtype=dtype).unsqueeze(-1)

    def __len__(self):
        return len(self.X)                

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def _generate_gaussian(self, seq_length):
        def rbf_kernel(x1, x2, variance = 1):
            from math import exp
            return exp(-1 * ((x1-x2) ** 2) / (2*variance))
        def gram_matrix(xs):
            return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]
        xs = np.arange(seq_length)*0.1
        mean = [0 for _ in xs]
        gram = gram_matrix(xs)
        ys = np.random.multivariate_normal(mean, gram)
        return ys


class Seq2SeqModel(pl.LightningModule):
    def __init__(self):
        super().__init__() 
        
        self.save_hyperparameters()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "train_loss_epoch"
                    }
        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx, loss=nn.MSELoss()):
        x, y = batch
        y_hat = self(x)
        trainloss = loss(y_hat, y)
        self.log("train_loss", trainloss, on_epoch=True, prog_bar=True, logger=True)
        return trainloss

    def validation_step(self, batch, batch_idx, loss=nn.MSELoss()):
        x, y = batch
        y_hat = self(x)
        validloss = loss(y_hat, y)
        self.log("valid_loss", validloss, prog_bar=True, logger=True)
        return validloss

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self(x)
        return pred.detach().cpu().numpy()


class RNNModel(Seq2SeqModel):
    def __init__(self, hid_dim, num_layers, input_dim, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        y = self.rnn(x)[0]
        y = nn.Tanh()(self.dense(y))
        output = y
        return output

class TCNModel(Seq2SeqModel):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        if len(x.shape) ==2:
            x = x.unsqueeze(0)
        x = x.permute(0,2,1)
        y1 = self.tcn(x)
        y1 = y1.permute(0,2,1)
        return self.linear(y1)

class TransformerModel(Seq2SeqModel):
    def __init__(self, input_dim, output_dim, num_layers, hid_dim, nhead, src_length, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.output_ff =  nn.Linear(hid_dim, output_dim)
        transformerlayer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformerlayer, num_layers=num_layers)
        self.pos_encoder = ConstantPositionalEncoding(hid_dim, max_len=src_length)
        mask = self._generate_square_subsequent_mask(src_length)
        self.register_buffer('mask', mask)

    def forward(self, x):
        x = self.input_ff(x)
        x = self.pos_encoder(x)
        y = self.transformer(x, self.mask)
        output = self.output_ff(y)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



class TransformerTextGeneration(Seq2SeqModel):
    def __init__(self, num_layers, hid_dim, nhead, dropout=0.1, load_data=False):
        super().__init__() 
        self.load_data = load_data
        input_dim = output_dim = self.data_setup()
        src_length = self.maxlen
        self.input_ff = nn.Linear(input_dim, hid_dim)
        self.output_ff =  nn.Linear(hid_dim, output_dim)
        transformerlayer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformerlayer, num_layers=num_layers)
        self.pos_encoder = ConstantPositionalEncoding(hid_dim, max_len=src_length)
        mask = self._generate_square_subsequent_mask(src_length)
        self.register_buffer('mask', mask)
        self.save_hyperparameters()


    def data_setup(self):
        data_name = 'wiki'
        with open(f'resources/data/text/{data_name}.txt', encoding='utf-8') as f:
            self.text = f.read().lower()

        self.chars = sorted(list(set(self.text)))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}
        
        self.maxlen = 50
        if self.data_setup:
            
            step = 5
            sentences = []
            next_chars = []
            for i in range(0, len(self.text) - self.maxlen, step):
                sentences.append(self.text[i: i + self.maxlen])
                next_chars.append(self.text[i + self.maxlen])

            x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=bool)
            y = np.zeros(len(sentences))
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    x[i, t, self.char_indices[char]] = 1
                y[i] = self.char_indices[next_chars[i]]

            self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.LongTensor(y))
        return len(self.chars)

    def forward(self, x, predicting=False):

        x = self.input_ff(x)
        x = self.pos_encoder(x)

        # During prediction the mask may have different shape.
        if not predicting:
            mask = self.mask
        else:
            mask = self._generate_square_subsequent_mask(x.shape[1])

        y = self.transformer(x, mask)[:,-1,:]
        output = self.output_ff(y)
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx, nn.CrossEntropyLoss())

    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx, nn.CrossEntropyLoss())

    def predict(self, sentence=None, start_index=None, length=400, diversity = 0.5):
        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        
        generated = ''
        if start_index is None and sentence is None:
            start_index = np.random.randint(0, len(self.text)-self.maxlen)
        if sentence is None:
            sentence = self.text[start_index: start_index + self.maxlen]
        sentence=sentence.lower()
        generated += sentence
        for _ in range(length):
            x_pred = np.zeros((1, len(sentence), len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self(torch.tensor(x_pred, dtype=torch.float32), predicting=True)[0].detach().cpu().numpy()
            # preds = softmax(preds)
        
            # next_index = sample(preds, diversity)
            next_index = np.argmax(preds)

            
            next_char = self.indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
        return generated


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=128,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class RNNTextGeneration(Seq2SeqModel):
    def __init__(self, hid_dim, num_layers, load_data=False):
        super().__init__() 
        self.load_data = load_data
        input_dim = output_dim = self.data_setup()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hid_dim, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hid_dim, output_dim)
        self.init_weight(self.rnn)
        # self.load_weight()
        self.save_hyperparameters()

    def init_weight(self, cell, gain=1):
        for layers in cell.all_weights:
            ih, hh, ih_b, hh_b = layers
            for i in range(0, hh.size(0), cell.hidden_size):
                torch.nn.init.orthogonal_(hh[i:i + cell.hidden_size], gain=gain)
            l = len(ih_b)
            ih_b[l // 4:l // 2].data.fill_(1.0)
            hh_b[l // 4:l // 2].data.fill_(1.0)

    def load_weight(self):
        import pickle
        from torch import from_numpy
        with open('weight.pkl', 'rb') as f:
            d = pickle.load(f)
        self.rnn._parameters['weight_ih_l0'].data = from_numpy(d['ih'])
        self.rnn._parameters['weight_hh_l0'].data = from_numpy(d['hh'])
        self.rnn._parameters['bias_ih_l0'].data = from_numpy(d['bias_ih'])
        self.rnn._parameters['bias_hh_l0'].data = from_numpy(d['bias_hh'])
        self.dense._parameters['weight'].data = from_numpy(d['dense_weight'])
        self.dense._parameters['bias'].data = from_numpy(d['dense_bias'])

    def data_setup(self):
        data_name = 'shakespeare'
        with open(f'resources/data/text/{data_name}.txt', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}

        if self.load_data:
            self.maxlen = 40
            step = 3
            sentences = []
            next_chars = []
            for i in range(0, len(self.text) - self.maxlen, step):
                sentences.append(self.text[i: i + self.maxlen])
                next_chars.append(self.text[i + self.maxlen])

            x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=bool)
            y = np.zeros(len(sentences))
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    x[i, t, self.char_indices[char]] = 1
                y[i] = self.char_indices[next_chars[i]]

            self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.LongTensor(y))
        return len(self.chars)

    def forward(self, x):
        y = self.rnn(x)[0][:,-1,:]
        y = self.dense(y)
        output = y
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-2)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx, nn.CrossEntropyLoss())

    def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx, nn.CrossEntropyLoss())

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.01)
        scheduler = {
                        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": "train_loss_epoch"
                    }
        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def predict(self, sentence=None, start_index=None, length=400, diversity = 0.5):
        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        
        generated = ''
        if start_index is None and sentence is None:
            start_index = np.random.randint(0, len(self.text)-self.maxlen)
        if sentence is None:
            sentence = self.text[start_index: start_index + self.maxlen]
        generated += sentence
        for i in range(length):
            x_pred = np.zeros((1, len(sentence), len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self(torch.tensor(x_pred, dtype=torch.float32))[0].detach().cpu().numpy()
            preds = softmax(preds)+1e-7
            next_index = sample(preds, diversity)
            
            
            next_char = self.indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
        return generated


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=128,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)


class RNNWordGeneration(RNNTextGeneration):
    def __init__(self, hid_dim, num_layers, load_data=True):
        super().__init__(hid_dim, num_layers, load_data)

    def data_setup(self):
        from collections import Counter, defaultdict
        import numpy as np
        data_name = 'shakespeare'
        with open(f'resources/data/text/{data_name}.txt', encoding='utf-8') as f:
            text = f.read().lower()
        text = list(filter(lambda x: x!='', text.split(' ')))
        self.chars = {word for word, count in Counter(text).items() if count >=10}
        # print(f'Have {len(self.chars)} of words')
        self.char_indices = defaultdict(lambda:0, {c: i for i, c in enumerate(sorted(self.chars),1)})
        self.indices_char = defaultdict(lambda:'', {c: i for i, c in self.char_indices.items()})
        maxlen = 40
        step = 3
        sentences = []
        next_word = []

        if self.load_data:
            for i in range(0, len(text) - maxlen, step):
                sentences.append(text[i: i + maxlen])
                next_word.append(text[i + maxlen])
            x = np.zeros((len(sentences)+1, maxlen+1, len(self.chars)+1),dtype=bool)
            y = np.zeros(len(sentences)+1)
            for i, sentence in enumerate(sentences):
                for t, word in enumerate(sentence):
                    x[i, t, self.char_indices[word]] = 1
                y[i] = self.char_indices[next_word[i]]
            
            self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.LongTensor(y))
        return len(self.chars)+1

    def predict(self, sentence=None, length=100, diversity = 0.5):
        def sample(preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        
        generated = ''
        sentence=sentence.lower()
        generated  = generated + sentence +' '
        for i in range(length):
            x_pred = np.zeros((1, len(sentence), len(self.chars)+1))
            for t, char in enumerate(sentence.split(' ')):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self(torch.tensor(x_pred, dtype=torch.float32))[0].detach().cpu().numpy()
            preds = softmax(preds)
            next_index = sample(preds, diversity)

            
            next_char = self.indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated  = generated + next_char +' '
        return generated