from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pickle
from lib.seq2seq_model import TCNModel, RNNModel, TransformerModel, TransformerTextGeneration, RNNTextGeneration,RNNWordGeneration

from math import floor
from datetime import datetime
from ml_collections import FrozenConfigDict
CONFIG = FrozenConfigDict({'shift': dict(LENGTH = 100,
                                                NUM = 20,
                                                SHIFT = 30),
                            'convo': dict(LENGTH = 100,
                                                NUM = 20,
                                                FILTER = [0.002, 0.022, 0.097, 0.159, 0.097, 0.022, 0.002]),
                                'lorenz': dict(NUM = 10, 
                                                K=1, J=10, 
                                                LENGTH=32 ), 'train_size':9500, 'valid_size': 500})


def train_model(name, model, input, output, train_test_split, epochs=300, batch_size=128, check_point_monitor='valid_loss', devices=4):
    """_summary_

    Args:
        name (str): Name of this run
        model (Model):The model
        input (ndarray): input array
        output (ndarray): output array
        train_test_split (float): ratio of train test split
    """
    if input is not None:
    # If input not provided then skip this part
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(input, output)
        total = len(dataset)
        train_size = floor(total*train_test_split)
        test_size = total - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    else:
        train_loader = None
        valid_loader = None

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    now = datetime.now().strftime("%H:%M:%S__%m-%d")
    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{now}", 
                                        save_top_k=4, 
                                        monitor=check_point_monitor,
                                        filename=name + "-{epoch:02d}-{valid_loss:.2e}") 
    
    if devices == 1:
        trainer = Trainer(accelerator="gpu", 
                    devices=1,
                    max_epochs=epochs,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=name),
                    callbacks=[checkpoint_callback, lr_monitor])
    else:
        trainer = Trainer(accelerator="gpu", 
                    devices=devices,
                    strategy=DDPStrategy(find_unused_parameters=False),
                    max_epochs=epochs,
                    precision=32,
                    logger=TensorBoardLogger("runs", name=name),
                    callbacks=[checkpoint_callback, lr_monitor])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)






def train_rnn_lorenz():

    model = RNNModel(hid_dim=256, num_layers=2, input_dim=1, output_dim=1)

    with open('resources/data/lorenz/lorenz_1_10_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Lorenz-RNN', model, input, output, 0.8, epochs=5000)

def train_rnn_shift():

    model = RNNModel(hid_dim=128, num_layers=1, input_dim=1, output_dim=1)


    with open('resources/data/shift/shift_32_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Shift-TCN', model, input, output, 0.8, epochs=5000, devices=1)

def train_TCN_shift():

    model = TCNModel(input_size=1, output_size=1,num_channels=[10]*7, kernel_size=4, dropout=0.1)

    with open('resources/data/shift/shift_32_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Shift-TCN', model, input, output, 0.8, epochs=5000)


def train_TCN_lorenz():

    model = TCNModel(input_size=1, output_size=1,num_channels=[30]*7, kernel_size=4, dropout=0.1)

    with open('resources/data/lorenz/lorenz_1_10_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Lorenz-TCN', model, input, output, 0.8, epochs=5000)



def train_transformer_shift():

    model = TransformerModel(input_dim=1, output_dim=1, num_layers=5,hid_dim=32,nhead=8,src_length=128)


    with open('resources/data/shift/shift_32_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Shift-Transformer', model, input, output, 0.8, epochs=5000)


def train_transformer_lorenz():

    model = TransformerModel(input_dim=1, output_dim=1, num_layers=5,hid_dim=32,nhead=8,src_length=128)


    with open('resources/data/lorenz/lorenz_1_10_128.pkl', 'rb') as f:
        input, output = pickle.load(f)

    train_model('Lorenz-Transformer', model, input, output, 0.8, epochs=5000)


def train_transformer_text():

    model = TransformerTextGeneration(num_layers=12,hid_dim=720,nhead=12).load_from_checkpoint('checkpoints/17:26:23__05-19/Text-Transformer-epoch=45-valid_loss=0.00e+00.ckpt')


    train_model('Text-Transformer', model, None, None, 0.8, epochs=5000, check_point_monitor='train_loss_epoch')

def train_rnn_text():
    model = RNNTextGeneration(hid_dim=128, num_layers=1, load_data=True)
    train_model('Text-RNN', model, None, None, None, epochs=5000, check_point_monitor='train_loss_epoch', devices=4)



def train_rnn_word():
    model = RNNWordGeneration(hid_dim=128, num_layers=1, load_data=True)
    train_model('Word-RNN', model, None, None, None, epochs=5000, check_point_monitor='train_loss_epoch', devices=4)

