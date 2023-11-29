# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from torchviz import make_dot
import os, fnmatch
import torchaudio
import sounddevice as sd
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import array
import torch.fft as fft
from CustomDataloader import CustomDataloaderCreator,DataConfig
from Training import Trainer
from tqdm import tqdm
import wandb
import sys

from plottingHelper import compareTwoAudios


class SimpleModel(nn.Module):
    def __init__(self, input_feature_dim, output_feature_dim,hiddenSize,numHiddenLayers,dtype = torch.float64):
        super(SimpleModel, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim

        layers = []

        for i in range(numHiddenLayers):
            if(i==0):
                layers.append(nn.Linear(input_feature_dim, hiddenSize,dtype=dtype))
                layers.append(nn.ReLU())

            elif(i==numHiddenLayers-1):
                layers.append(nn.Linear(hiddenSize, output_feature_dim,dtype=dtype))

            else:
                layers.append(nn.Linear(hiddenSize, hiddenSize,dtype=dtype))
                layers.append(nn.ReLU())
            

        self.neuralnet = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(-1)
        # print("Input dtype: ", x.dtype)
        output = self.neuralnet(x)
        # output = fft.irfft(output)
        return output

def sweep(config,sweeping,useWandB):
    print("Sweeping: ", sweeping)
    print("Use WandB: ", useWandB)

    if(not sweeping): print("Default Config:")
    print(config)


    if sweeping:
        validRun = (config.stride_length < config.frameSize) or (config.frameSize//config.stride_length > 4)
        if(not validRun):
            wandb.run.finish(exit_code=1)
            print("Invalid run: FrameSize and stride_length not compatible")
            sys.exit(0)
  

    # Set up torch and cuda
    dtype = torch.float32
    # dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    print(device)
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    # ### # Defining parameter
    noisyPath = '/home/ubuntu/OticonStuff/dataset/train'
    cleanPath = '/home/ubuntu/OticonStuff/dataset/y_train'
    noisy_files_list = fnmatch.filter(os.listdir(noisyPath), '*.wav')
    clean_files_list = fnmatch.filter(os.listdir(cleanPath), '*.wav')

    # print("Number of noisy files: ", len(noisy_files_list))
    # print("Number of clean files: ", len(clean_files_list))
    # print("Noisy file: ", noisy_files_list[1])
    # print("Clean file: ", clean_files_list[1])

    #Corresponds to 512 -> 32ms
    frameSize = config.frameSize
    #Corresponds to 32 -> 2ms
    stride_length = config.stride_length
    speechSampleSize = 48000  
    sampleRate = 16000

    # Model Input Buffer ( How many frames of size "frameSize" to be fed to the model)
    modelBufferFrames = config.modelBufferFrames

    # Forward pass buffer
    fft_freq_bins = frameSize // 2 + 1

    print("Speech Sample Size: ", speechSampleSize)
    print("Sample Rate: ", sampleRate)
    print("FFT Frequency Bins: ", fft_freq_bins)

    #Split into train and temp ( 70-15-15 split for now)
    X_train, X_temp, y_train, y_temp = train_test_split(noisy_files_list, clean_files_list, test_size=0.3, random_state=42)

    #Splitting the temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # print(f'shape of numpy X_train: {np.array(X_train).shape}')
    # # print(f'shape of numpy y_train: {np.array(y_train).shape}')
    # print(f'shape of numpy X_test: {np.array(X_test).shape}')
    # # print(f'shape of numpy y_test: {np.array(y_test).shape}')

    # print(f'shape of numpy X_val: {np.array(X_val).shape}')
    # # print(f'shape of numpy y_test: {np.array(y_test).shape}')

    
    #All defaults in dataconfig
    dataConfig = DataConfig(
        batchSize= config.batchSize,
        dtype= dtype,
        device = device,
        learningRate = config.learningRate,
        frameSize = frameSize, 
        stride_length = stride_length,
        sample_rate = sampleRate,
        duration = config.duration,
        n_fft = frameSize,
        modelBufferFrames = modelBufferFrames,
        shuffle = config.shuffle,
        noisyPath = noisyPath,
        cleanPath = cleanPath
    )

    dataloaderCreator = CustomDataloaderCreator(X_train, y_train,X_test,y_test,X_val,y_val,dataconfig=dataConfig)
    dataloaderCreator.prepare()

    dataloader = dataloaderCreator.getTrainDataloader2()
    validationDataloader = dataloaderCreator.getValidationDataloader2()
    # testDataloader = dataloaderCreator.getTestDataloader()

    X_train_eg = X_train[0]
    print(X_train_eg)
    y_train_eg = y_train[0]
    print(y_train_eg)

    X_test_eg = X_test[0]
    print(X_test_eg)
    y_test_eg = y_test[0]
    print(y_test_eg)


    num_epochs = config.num_epochs
    fft_freq_bins = dataConfig.frameSize // 2 + 1
    input_feature_dim = dataConfig.modelBufferFrames*fft_freq_bins
    output_feature_dim = fft_freq_bins
    # output_feature_dim = dataConfig.stride_length
    # output_feature_dim = dataConfig.frameSize

    model = SimpleModel(input_feature_dim, output_feature_dim,hiddenSize= config.hiddenSize,numHiddenLayers=config.numHiddenLayers,dtype=dataConfig.dtype).to(device)

    # model = SequenceModeller(input_feature_dim, output_feature_dim,dtype=dataConfig.dtype).to(device)

    # model = StatelessLSTM(  input_feature_dim=input_feature_dim,
    #                         conv_filters=32,
    #                         kernel_size=5,
    #                         conv_stride=3,
    #                         stride_length=dataConfig.stride_length,
    #                         lstm_hidden_units= 5,
    #                         dtype=dataConfig.dtype
    #                         ).to(device)

    # Define your loss function (e.g., mean squared error)
    loss_fn = nn.MSELoss()

    # Define your optimizer (e.g., Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=dataConfig.learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8, verbose=False)


    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        loss_func = loss_fn,
        num_epochs = num_epochs,
    )

    if(useWandB):
        if(sweeping):
            runName = wandb.run.name
        else:
            runName = f'NormalRun:{dataConfig.frameSize}_stride:{dataConfig.stride_length}_MBF:{dataConfig.modelBufferFrames}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_shuffle:{dataConfig.shuffle}'
    
    else:
        runName = "Doesnt matter here"
    
    trainer.train2(
        train_dataloader = dataloader,
        val_dataloader = validationDataloader,
        dataConfig = dataConfig,
        modelSaveDir = "/home/ubuntu/OticonStuff/models",
        wandbName = runName,
        wavFileTesting = "19-198-0003.wav",
        debugFlag = False,
        useWandB=useWandB,
        sweeping = sweeping
    )

# ------------------------------------------------------------------------------------------------------------------------------------------
# Actual code execution starts here

sweep_config = {
    'method': 'bayes',
    'name': 'Initial_Sweep(pesqScore)',
    'metric': {
      'name': 'pesqScore',
      'goal': 'maximize'   
    },
    'parameters': {
        'frameSize': {
            'values': [512,512*2,512*3,512*4,512*5,512*8,512*12]
        },
        'stride_length': {
            'values': [32,64,128,256,512*2,512*3,512*4,512*5,512*8,512*12]
        },
        'modelBufferFrames': {
            'min': 1,'max': 15
        },
        'learningRate': {
            'values': [0.01,0.001]
        },
        'batchSize': {
            'values': [64,128,256]
        },
        'num_epochs': {
            'values': [250]
        },
        'shuffle': {
            'values': [True,False]
        },
        'duration': {
            'values': [3,5]
        },
        'hiddenSize': {
             'min': 10,'max': 512
        },
        'numHiddenLayers': {
           'min': 3,'max': 20
        }
    }
}

class DefaultConfig():
    def __init__(self):
        self.frameSize = 512
        self.stride_length = 256
        self.modelBufferFrames = 1
        self.learningRate = 0.01
        self.batchSize = 256
        self.num_epochs = 150
        self.shuffle = True
        self.duration = 3
        self.hiddenSize = 512
        self.numHiddenLayers = 3
    
    def printMembers(self):
        print("frameSize: ", self.frameSize)
        print("stride_length: ", self.stride_length)
        print("modelBufferFrames: ", self.modelBufferFrames)
        print("learningRate: ", self.learningRate)
        print("batchSize: ", self.batchSize)
        print("num_epochs: ", self.num_epochs)
        print("shuffle: ", self.shuffle)
        print("duration: ", self.duration)
        print("hiddenSize: ", self.hiddenSize)
        print("numHiddenLayers: ", self.numHiddenLayers)


def main():
    if sweeping:
        with wandb.init(project='Shobhit_SEM9') as run:
            run.name = f'Sweep:{wandb.config.frameSize}_stride:{wandb.config.stride_length}_MBF:{wandb.config.modelBufferFrames}_LR:{wandb.config.learningRate}_BS:{wandb.config.batchSize}_Epochs:{wandb.config.num_epochs}_shuffle:{wandb.config.shuffle}'
            sweep(wandb.config,sweeping,useWandB)
        


# ----Main Controls-----
useWandB = True
sweeping = True
resume = False
# -----------------------

if sweeping:
    if(resume):
        #TODO: Change this line later
        agent = wandb.agent("ricenet/Shobhit_SEM9/lvbaqd19", function=main, count=500)
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project='Shobhit_SEM9') 
        agent = wandb.agent(sweep_id, function=main, count=500)

else:
    sweep(DefaultConfig(),sweeping,useWandB)

  