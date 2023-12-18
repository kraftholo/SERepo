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

from plottingHelper import compareTwoAudios
import torch.nn.functional as F

# %%
# Set up torch and cuda
dtype = torch.float32
# dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

print(device)
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class DefaultConfig():
    def __init__(self):
        self.frameSize = 512
        self.stride_length = 256
        self.learningRate = 0.01
        self.batchSize = 256
        self.num_epochs = 500
        self.shuffle = True
        self.duration = 6
        self.hiddenSize = 512*4
        self.numHiddenLayers = 4
        self.filesToUse = 150 * 5
    
    def printMembers(self):
        print("frameSize: ", self.frameSize)
        print("stride_length: ", self.stride_length)
        print("learningRate: ", self.learningRate)
        print("batchSize: ", self.batchSize)
        print("num_epochs: ", self.num_epochs)
        print("shuffle: ", self.shuffle)
        print("duration: ", self.duration)
        print("hiddenSize: ", self.hiddenSize)
        print("numHiddenLayers: ", self.numHiddenLayers)

class SimpleLinearModel(nn.Module):
    def __init__(self, input_feature_dim,hiddenSize,numHiddenLayers,dtype = torch.float64):
        super(SimpleLinearModel, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.flattenedDims = input_feature_dim[0]*input_feature_dim[1]
        layers = []
        layers.append(nn.Flatten())

        for i in range(numHiddenLayers):
            if(i==0):
                layers.append(nn.Linear(self.flattenedDims, hiddenSize,dtype=dtype))
                layers.append(nn.ReLU())

            elif(i==numHiddenLayers-1):
                layers.append(nn.Linear(hiddenSize, self.flattenedDims,dtype=dtype))
                layers.append(nn.Sigmoid())                                                 # Sigmoid last activation

            else:
                layers.append(nn.Linear(hiddenSize, hiddenSize,dtype=dtype))
                layers.append(nn.ReLU())
            

        self.neuralnet = nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(-1)
        # print("Input dtype: ", x.dtype)
        output = self.neuralnet(x)
        op = output.view(-1,*self.input_feature_dim)
        # output = fft.irfft(output)
        return op
    
class BasicAutoEncoder(nn.Module):
    def __init__(self, input_feature_dim,latentSize,dtype = torch.float64):
        super(BasicAutoEncoder, self).__init__()
        
        self.input_feature_dim = input_feature_dim
        self.flattenedDims = input_feature_dim[0]*input_feature_dim[1]
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.flattenedDims, latentSize*4,dtype=dtype))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(latentSize*4, latentSize*2,dtype=dtype))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(latentSize*2, latentSize,dtype=dtype))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(latentSize, latentSize*2,dtype=dtype))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(latentSize*2, latentSize*4,dtype=dtype))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(latentSize*4, self.flattenedDims,dtype=dtype))
        layers.append(nn.Sigmoid())                                                # Sigmoid last activation

        self.neuralnet = nn.Sequential(*layers)

    def forward(self, x):
        # print(f'Input shape: {x.shape}')            #([BatchSize,257, 376])
        op = self.neuralnet(x)
        # print(f'Model output shape: {op.shape}')
        op = op.view(-1,*self.input_feature_dim)
        # print(f'Final output shape: {op.shape}')    # ([BatchSize,257, 376])
        return op

# https://medium.com/@polanitzer/building-a-cnn-based-autoencoder-with-denoising-in-python-on-gray-scale-images-of-hand-drawn-digits-61131ec492e4
class ConvAutoEncoder(nn.Module):
    def __init__(self,dtype = torch.float64):
        super(ConvAutoEncoder, self).__init__()
        # Expected input = (batch_size, inputFFTFrames, numFFTBins)

        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size= (3,3), stride= (1,1), padding= (1,1),dtype=dtype)
        self.maxpool1 = nn.MaxPool2d(kernel_size= (2,2), stride= (2,2))
        
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= (3,3), stride= (1,1), padding= (1,1),dtype=dtype)
        self.maxpool2 = nn.MaxPool2d(kernel_size= (2,2), stride= (2,2))

        self.t_conv1 = nn.ConvTranspose2d(in_channels= 64, out_channels= 64, kernel_size= (3,3), padding=1,dtype=dtype)
        self.upSample1 = nn.Upsample(scale_factor= 2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels= 64, out_channels= 32, kernel_size= (3,3), padding=1,dtype=dtype)
        self.upSample2 = nn.Upsample(scale_factor= 2)
        self.t_conv3 = nn.Conv2d(in_channels= 32, out_channels= 1, kernel_size= (3,3), padding=1,dtype=dtype)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

    def forward(self,x):
        # print("Input shape: ", x.shape) # ([BatchSize,257, x])
        inputShape = np.copy(x.shape)

        reshapedInput = x.unsqueeze(-1) # ([BatchSize, 257, x, 1])
        x = reshapedInput.permute(0, 3, 1, 2) # ([BatchSize, 1, 256, x]) One channel image
        # print("After permute shape: ", x.shape)
        x = self.conv1(x)
        # print("After conv1 shape: ", x.shape)
        x = self.relu(x)
        x = self.maxpool1(x)
        # print("After maxpool1 shape: ", x.shape)
        x = self.conv2(x)
        # print("After conv2 shape: ", x.shape)
        x = self.relu(x)
        x = self.maxpool2(x)
        # print("After maxpool2 shape: ", x.shape)
        x = self.t_conv1(x)
        # print("After t_conv1 shape: ", x.shape)
        x = self.relu(x)
        x = self.upSample1(x)
        # print("After upSample1 shape: ", x.shape)
        x = self.t_conv2(x)
        # print("After t_conv2 shape: ", x.shape)
        x = self.relu(x)
        x = self.upSample2(x)
        # print("After upSample2 shape: ", x.shape)
        x = self.t_conv3(x)
        # print("After t_conv3 shape: ", x.shape)
        x = self.sigmoid(x)                                                         # Sigmoid last activation   
        # print("After sigmoid shape: ", x.shape)

        masksGenerated = x.permute(0, 2, 3, 1) # ([BatchSize,256, x, 1])
        masksGenerated = masksGenerated.squeeze(-1) # ([BatchSize,256, x])
        # print("masksGenerated shape: ", masksGenerated.shape)

        padded_masks = torch.zeros((inputShape[0], inputShape[1], inputShape[2]))
        padded_masks[:masksGenerated.shape[0], :masksGenerated.shape[1], :masksGenerated.shape[2]] = masksGenerated

        # print("Final output shape: ", masksGenerated.shape)
        return padded_masks


    
# model = ConvAutoEncoder(dtype)
# print(model)

# %%
noisyPath = '/home/ubuntu/OticonStuff/dataset/train'
cleanPath = '/home/ubuntu/OticonStuff/dataset/y_train'
noisy_files_list = fnmatch.filter(os.listdir(noisyPath), '*.wav')
clean_files_list = fnmatch.filter(os.listdir(cleanPath), '*.wav')
config = DefaultConfig()

# print("Number of noisy files: ", len(noisy_files_list))
# print("Number of clean files: ", len(clean_files_list))
# print("Noisy file: ", noisy_files_list[1])
# print("Clean file: ", clean_files_list[1])

#Corresponds to 512 -> 32ms
frameSize = config.frameSize
#Corresponds to 32 -> 2ms
stride_length = config.stride_length 
sampleRate = 16000
speechSampleSize = config.duration * sampleRate 


# %%
X_train, X_temp, y_train, y_temp = train_test_split(noisy_files_list, clean_files_list, test_size=0.3, shuffle=True)

#Splitting the temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,shuffle=True)

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
    modelBufferFrames= 0,                   #Doesn't matter for non-RT
    duration = config.duration,
    n_fft = frameSize,
    shuffle = config.shuffle,
    noisyPath = noisyPath,
    cleanPath = cleanPath
)

dataloaderCreator = CustomDataloaderCreator(X_train, y_train,X_test,y_test,X_val,y_val,dataconfig=dataConfig,filesToUse=config.filesToUse)
dataloaderCreator.prepareNonRT()

dataloader = dataloaderCreator.getTrainDataloaderNonRT()
validationDataloader = dataloaderCreator.getValidationDataloaderNonRT()

numOfFrames = dataloaderCreator.getNumOfFrames()
numOfFreqBins = dataloaderCreator.getNumOfFreqBins()
dataConfig.numOfFreqBins = numOfFreqBins
dataConfig.numOfFrames = numOfFrames

selectedTrainExampleFile = X_train[4]
print("Selected train example file: ", selectedTrainExampleFile)

# %%
print("Number of frames: ", numOfFrames)
print("Number of freq bins: ", numOfFreqBins)

# Model config
num_epochs = config.num_epochs
useScheduler = False
useModelNo = 1


if(useModelNo == 0):
    model = ConvAutoEncoder(dataConfig.dtype)
    runName = f'CNonRT:{dataConfig.frameSize}_stride:{dataConfig.frameSize/2}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}_shuffle:{dataConfig.shuffle}_ConvAE'
    # runName = f'CNonRT:{dataConfig.frameSize}_stride:{dataConfig.frameSize/2}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}_shuffle:{dataConfig.shuffle}_ConvAE_NoAct'
    # runName = f'CNonRT(TDLOSS):{dataConfig.frameSize}_stride:{dataConfig.frameSize/2}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}_shuffle:{dataConfig.shuffle}_ConvAE'
elif(useModelNo == 1):
    latentsize = 16*8
    model = BasicAutoEncoder((numOfFreqBins,numOfFrames),latentSize=latentsize,dtype=dataConfig.dtype) #([BatchSize,257, 376])
    runName = f'CNonRT:{dataConfig.frameSize}_LatentSize:{latentsize}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}__shuffle:{dataConfig.shuffle}_BasicAE(lessData)'
    # runName = f'CNonRT:{dataConfig.frameSize}_LatentSize:{latentsize}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}__shuffle:{dataConfig.shuffle}_BasicAE_NoAct'
    # runName = f'CNonRT(TDLOSS):{dataConfig.frameSize}_LatentSize:{latentsize}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}__shuffle:{dataConfig.shuffle}_BasicAE'
else:
    model = SimpleLinearModel((numOfFreqBins,numOfFrames),config.hiddenSize,config.numHiddenLayers,dtype=dataConfig.dtype)
    runName = f'CNonRT:{dataConfig.frameSize}_LatentSize:{config.hiddenSize}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}_shuffle:{dataConfig.shuffle}_SimpleLinearModel'
    # runName = f'CNonRT:{dataConfig.frameSize}_LatentSize:{config.hiddenSize}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}_shuffle:{dataConfig.shuffle}_SimpleLinearModel_NoAct'
    # runName = f'CNonRT(TDLOSS):{dataConfig.frameSize}_LatentSize:{config.hiddenSize}_LR:{dataConfig.learningRate}_BS:{dataConfig.batchSize}_Epochs:{config.num_epochs}_decayingLR:{useScheduler}_shuffle:{dataConfig.shuffle}_SimpleLinearModel'

print("Model: ", model)
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


# Wandb Controls====================================
useWandB = True
sweeping = False
# ====================================

# Loss in the frequency domain 
trainer.trainNonRT(
    train_dataloader = dataloader,
    val_dataloader = validationDataloader,
    dataConfig = dataConfig,
    modelSaveDir = "/home/ubuntu/OticonStuff/models",
    wandbName = runName,
    wavFileTesting = f'{selectedTrainExampleFile}',
    debugFlag = False,
    useWandB=useWandB,
    sweeping = sweeping,
    useScheduler = useScheduler,
    scaleFactor= 1
)


# Loss in the time domain
# trainer.trainNonRTLossTD(
#         train_dataloader = dataloader,
#         val_dataloader = validationDataloader,
#         dataConfig = dataConfig,
#         modelSaveDir = "/home/ubuntu/OticonStuff/models",
#         wandbName = runName,
#         wavFileTesting = f'{selectedTrainExampleFile}',
#         debugFlag = False,
#         useWandB=useWandB,
#         sweeping = sweeping,
#         useScheduler = useScheduler,
#         scaleFactor= 1
#     )



