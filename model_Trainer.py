import torch.nn as nn
from models.models import *
from models.models_config import get_model_config
from models.pre_train_test_split import trainer
import torch
from torch.utils.data import DataLoader
from utils import *
device = torch.device('cpu')

# loading data for training and testing, specifiying the parameters and running the model 
my_data = torch.load('/content/drive/My Drive/data/1s_11c_1024_snr0_D0.5_os.pt')

##################################### Training Batch Size ##############

train_dl = DataLoader(MyDataset(my_data['train_data'], 
                                my_data['train_labels']), batch_size=256, shuffle=True, drop_last=True)

#################################### Testing Batch Size ################

test_dl = DataLoader(MyDataset(my_data['test_data'], 
                               my_data['test_labels']), batch_size = 1, shuffle=False, drop_last=False)

#################################### Hidden Dimension ##################

model = CNN_1D(1,1024,0.5).to(device)

################################### Epochs and Learning Rate ###########

params = {'pretrain_epoch': 1000,'lr': 1e-3}

########################################################################
config = get_model_config('CNN')

trained_model=trainer(model, train_dl, test_dl,'SHM_C', config, params) 

# Saving the model for inference (Learned Paramters only)
torch.save(trained_model.state_dict(),'1s_11c_model')
#######################################################################
