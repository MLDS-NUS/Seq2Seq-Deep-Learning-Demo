import random
import numpy as np
import torch
from torch import nn
from lib.seq2seq_model import RNNModel, TCNModel, TransformerModel
from lib.plot import ShiftPlotter, ConvoPlotter, LorenzPlotter, TextGenerator, LorenzEvaluation, ShiftEvaluation


SEED = 1234
DTYPE = torch.float32
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark=True
