# %%
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
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import array
import torch.fft as fft
from CustomDataloader import CustomDataloaderCreator,DataConfig
import tqdm
from collections import OrderedDict
import wandb
import copy
from plottingHelper import compareTwoAudios
from trainingMetricHelper import returnPesqScore,makeFileReady, makeFileReadyNonRT,runInferenceWithModel,runInferenceWithModel2,runInferenceNonRT,runInferenceNonRTMag

class Trainer():

    def __init__(self,model,optimizer,scheduler,loss_func,num_epochs):
        self.model = model
        self.modelcopy = copy.deepcopy(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.num_epochs = num_epochs

        self.best_vals = {'psnr': 0.0, 'loss': 1e8,'pesq':0.0}
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
        self.best_pesqmodel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
        self.modelParameterCount = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'modelParameterCount = {self.modelParameterCount}')

    def train(self,train_dataloader,val_dataloader,dataConfig,modelSaveDir,wandbName,debugFlag = False,useWandB = True):    
        name = wandbName
        #Initializing wandb  

        if(useWandB):
            wandb.init(
                # set the wandb project where this run will be logged
                project="Shobhit_SEM9",
                name= name,
                config={
                    "epochs": self.num_epochs,
                    "learning_rate": dataConfig.learningRate,
                    "batch_size": dataConfig.batchSize,
                    "stride_length": dataConfig.stride_length,
                    "frame_size": dataConfig.frameSize,
                    "sample_rate": dataConfig.sample_rate,
                    "duration": dataConfig.duration,
                    "n_fft": dataConfig.n_fft,
                    "modelBufferFrames": dataConfig.modelBufferFrames,
                    "shuffle": dataConfig.shuffle,
                    "dtype": dataConfig.dtype,
                },
            )
            wandb.log({'model parameters' : self.modelParameterCount})

        modelPath = f'modelSaveDir/{name}'
        fft_freq_bins = int(dataConfig.n_fft/2) + 1

        
        #Start training loop
        with tqdm.trange(self.num_epochs, ncols=100) as t:
            for i in t:
                # <Inside an epoch>    
                #Make sure gradient tracking is on, and do a pass over the data
                self.model.train(True)

                # Update model
                self.optimizer.zero_grad()

                running_trainloss = 0.0
                #Training loop
                randomSelectedBatchNum = np.random.randint(0,len(train_dataloader))
                

                for batchNum,data in enumerate(train_dataloader):
                    # <Inside a batch>  
                    modelInputs, targets = data
                    randomSelectedTrainingPoint = np.random.randint(0,targets.shape[0])
                    # print(f'modelInputs.dtype = {modelInputs.dtype}')

                    #ModelInputs here is of type complex64
                    if(dataConfig.dtype == torch.float32):
                        modelInputs = torch.abs(modelInputs).float()
                    else:
                        modelInputs = torch.abs(modelInputs).double()
                   
                    # print(f'modelInputs.dtype = {modelInputs.dtype}')
                    #Idk if this is required now
                    if(batchNum == len(train_dataloader)):
                        break

                    reshaped_input = modelInputs.view(modelInputs.shape[0], fft_freq_bins*dataConfig.modelBufferFrames)
                    #Model input is (Batchsize, 257*10) :: batch of 10 frames of 257 FFT bins
                    ifftedOutputs = self.model(reshaped_input)

                    #Model output is (Batchsize, 512) :: batch of single IFFT-ed frame of 257 FFT bins
                    if(debugFlag): 
                        print(f'ifftedOutputs.shape = {ifftedOutputs.shape}')     

                    #Taking the first 32 samples from the ifft output
                    # firstSamples = ifftedOutputs[:,:dataConfig.stride_length] 
                    firstSamples = ifftedOutputs

                    if(debugFlag):
                        print(f'IFFT of model output shape = {ifftedOutputs.shape}')
                        print(f'IFFT of model output first {dataConfig.stride_length} samples shape = {firstSamples.shape}')   
                        print(f'targets.shape = {targets.shape}')

                    loss = self.loss_func(firstSamples, targets)


                    if(batchNum == randomSelectedBatchNum and i%10 ==0):
                        compareTwoAudios(firstSamples[randomSelectedTrainingPoint][:dataConfig.stride_length],targets[randomSelectedTrainingPoint][:dataConfig.stride_length],i,randomSelectedBatchNum,logInWandb = useWandB)
                    
                    running_trainloss += loss
                    loss.backward()
                    self.optimizer.step()

                # <After an epoch> 
                avg_trainloss = running_trainloss / len(train_dataloader)
                # Check for validation loss!
                running_vloss = 0.0
                # Set the model to evaluation mode
                self.model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():

                    for i,data in enumerate(val_dataloader):
                        
                        val_modelInputs, val_targets = data

                         #val_modelInputs here is of type complex64
                        if(dataConfig.dtype == torch.float32):
                            val_modelInputs = torch.abs(val_modelInputs).float()
                        else:
                            val_modelInputs = torch.abs(val_modelInputs).double()
                   

                        #Idk if this is required now
                        if(i == len(val_dataloader)):
                            break

                        val_reshaped_input = val_modelInputs.view(val_modelInputs.shape[0], fft_freq_bins*dataConfig.modelBufferFrames)
                        #Model input is (Batchsize, 257*10) :: batch of 10 frames of 257 FFT bins
                        val_ifftedOutputs = self.model(val_reshaped_input)

                        #Model output is (Batchsize, 512) :: batch of single IFFT-ed frame of 257 FFT bins
                        if(debugFlag): 
                            print(f'ifftedOutputs.shape = {val_ifftedOutputs.shape}')     

                        #Taking the first 32 samples from the ifft output
                        # firstSamples = ifftedOutputs[:,:dataConfig.stride_length] 
                        val_firstSamples = val_ifftedOutputs

                        if(debugFlag):
                            print(f'IFFT of model output shape = {val_ifftedOutputs.shape}')
                            print(f'IFFT of model output first {dataConfig.stride_length} samples shape = {val_firstSamples.shape}')   

                        val_loss = self.loss_func(val_firstSamples, val_targets)

                        # if(i == len(val_dataloader)/2):
                            # compareTwoAudios(val_firstSamples[5],val_targets[5])
                        #     # printQualityScores(val_targets[5],val_firstSamples[5],dataConfig.sample_rate)

                        running_vloss += val_loss

                # Calculate average val loss
                avg_vloss = running_vloss / len(val_dataloader)
                print('LOSS train {} valid {}'.format(avg_trainloss, avg_vloss))

                # Update the scheduler
                self.scheduler.step(avg_vloss)

                if(useWandB):
                    # Log results to W&B
                    wandb.log({
                        'trainLoss': avg_trainloss,
                        'valLoss': avg_vloss,
                    })
                
                #Save the model if the validation loss is good
                if avg_vloss < self.best_vals['loss']:
                    self.best_vals['loss'] = avg_vloss
                    self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                    torch.save(self.best_model, f'{wandbName}.pt')
        
        if(useWandB):
            wandb.finish() 


    #Training spectral masking
    def train2(self,train_dataloader,val_dataloader,dataConfig,modelSaveDir,wavFileTesting,wandbName,sweeping,debugFlag = False,useWandB = True):
            name = wandbName
            #Initializing wandb  
            exampleModelInputs, examplePhaseInfos, exampleCleanSpeech, exampleNoisySpeech = makeFileReady(wavFileTesting,dataConfig)
            pesqToBeat = returnPesqScore(exampleCleanSpeech,exampleNoisySpeech,dataConfig.sample_rate)
            print(f'pesqToBeat = {pesqToBeat}')

            if(useWandB):
                if(not sweeping):
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project="Shobhit_SEM9",
                        name= name,
                        config={
                            "epochs": self.num_epochs,
                            "learning_rate": dataConfig.learningRate,
                            "batch_size": dataConfig.batchSize,
                            "stride_length": dataConfig.stride_length,
                            "frame_size": dataConfig.frameSize,
                            "sample_rate": dataConfig.sample_rate,
                            "duration": dataConfig.duration,
                            "n_fft": dataConfig.n_fft,
                            "modelBufferFrames": dataConfig.modelBufferFrames,
                            "shuffle": dataConfig.shuffle,
                            "dtype": dataConfig.dtype,
                        },
                    )
                wandb.log({'model parameters' : self.modelParameterCount,
                           'testAudio': wandb.Audio(exampleCleanSpeech, caption="Clean Speech", sample_rate= dataConfig.sample_rate),
                           'psnrToBeat': pesqToBeat})
                

            modelPath = f'modelSaveDir/{name}'
            fft_freq_bins = int(dataConfig.n_fft/2) + 1
            
            #Start training loop
            with tqdm.trange(self.num_epochs, ncols=100) as t:
                for i in t:
                    # <Inside an epoch>    
                    #Make sure gradient tracking is on, and do a pass over the data
                    self.model.train(True)

                    # Update model
                    self.optimizer.zero_grad()

                    running_trainloss = 0.0
                    #Training loop
                    randomSelectedBatchNum = np.random.randint(0,len(train_dataloader))
                    

                    for batchNum,data in enumerate(train_dataloader):
                        # <Inside a batch>  
                        combined_tensor, targets = data
                        randomSelectedTrainingPoint = np.random.randint(0,targets.shape[0])
                        # print(f'modelInputs.dtype = {modelInputs.dtype}')
                        # print(f'modelInputs.shape = {modelInputs.shape}')

                        # print("combined_tensor.shape",combined_tensor.shape)

                        length = len(targets)
                        bufferlen = dataConfig.modelBufferFrames
                        framesize = fft_freq_bins
                        # Split the tensor back into two tensors
                        modelInputs = combined_tensor[:, :bufferlen * framesize].reshape(length, bufferlen, framesize)
                        phaseInfos = combined_tensor[:, bufferlen * framesize:]
                        # print("modelInputs shape ", {modelInputs.shape})
                        # print("phaseInfos shape ", {phaseInfos.shape})


                        # print(f'batchInput.shape = {batchInput.shape}')
                        # print("batchInput['input'].shape",batchInput['input'].shape)
                        # print("batchInput['phaseInfo'].shape",batchInput['phaseInfo'].shape)
                        # print(f'targets.shape = {targets.shape}')

                        #ModelInputs here is of type complex64
                        if(dataConfig.dtype == torch.float32):
                            modelInputsMag = torch.abs(modelInputs).float()
                            targetsMag = torch.abs(targets).float()
                        else:
                            modelInputsMag = torch.abs(modelInputs).double()
                            targetsMag = torch.abs(targets).double()
                    
                        # print(f'modelInputs.dtype = {modelInputs.dtype}')
                        #Idk if this is required now
                        if(batchNum == len(train_dataloader)):
                            break
                        
                        lastFrameInBuffer = modelInputsMag[:,dataConfig.modelBufferFrames-1,:]
                        # print(f'lastFrameInBuffer.shape = {lastFrameInBuffer.shape}')

                        reshaped_input = modelInputsMag.view(modelInputsMag.shape[0], fft_freq_bins*dataConfig.modelBufferFrames)
                        #Model input is (Batchsize, 257*10) :: batch of 10 frames of 257 FFT bins
                        # ifftedOutputs = self.model(reshaped_input)
                        masksGenerated = self.model(reshaped_input)
                        # print(f'masksGenerated.shape = {masksGenerated.shape}')
                        outputs = lastFrameInBuffer*masksGenerated
                        
                        #Model output is (Batchsize, 512) :: batch of single IFFT-ed frame of 257 FFT bins
                        if(debugFlag): 
                            print(f'outputs.shape = {outputs.shape}')     

                        #Taking the first 32 samples from the ifft output
                        # firstSamples = ifftedOutputs[:,:dataConfig.stride_length] 
                        
                        fftFrames = outputs * torch.exp(1j*phaseInfos)
                        # print(f'fftFrames.shape = {fftFrames.shape}')
                        

                        loss = self.loss_func(outputs, targetsMag)


                        # if(batchNum == randomSelectedBatchNum and i%10 ==0):
                        #     compareTwoAudios(firstSamples[randomSelectedTrainingPoint][:dataConfig.stride_length],targets[randomSelectedTrainingPoint][:dataConfig.stride_length],i,randomSelectedBatchNum,logInWandb = useWandB)
                        
                        running_trainloss += loss
                        loss.backward()
                        self.optimizer.step()

                    # <After an epoch> 
                    avg_trainloss = running_trainloss / len(train_dataloader)
                    # Check for validation loss!
                    running_vloss = 0.0
                    # Set the model to evaluation mode
                    self.model.eval()

                    # Disable gradient computation and reduce memory consumption.
                    with torch.no_grad():

                        for k,data in enumerate(val_dataloader):
                            
                            val_combined_tensor, val_targets = data

                            # print("val_combined_tensor.shape",combined_tensor.shape)

                            length = len(val_targets)
                            bufferlen = dataConfig.modelBufferFrames
                            framesize = fft_freq_bins
                            # Split the tensor back into two tensors
                            # Split the tensor back into two tensors
                            val_modelInputs = val_combined_tensor[:, :bufferlen * framesize].reshape(length, bufferlen, framesize)
                            val_phaseInfos = val_combined_tensor[:, bufferlen * framesize:]
                            # print("val_modelInputs shape ", {val_modelInputs.shape})
                            # print("val_phaseInfos shape ", {val_phaseInfos.shape})


                            # print(f'batchInput.shape = {batchInput.shape}')
                            # print("batchInput['input'].shape",batchInput['input'].shape)
                            # print("batchInput['phaseInfo'].shape",batchInput['phaseInfo'].shape)
                            # print(f'val_targets.shape = {val_targets.shape}')

                            #ModelInputs here is of type complex64
                            if(dataConfig.dtype == torch.float32):
                                val_modelInputsMag = torch.abs(val_modelInputs).float()
                                val_targetsMag = torch.abs(val_targets).float()
                            else:
                                val_modelInputsMag = torch.abs(val_modelInputs).double()
                                val_targetsMag = torch.abs(val_targets).double()
                        
                            # print(f'modelInputs.dtype = {modelInputs.dtype}')
                            #Idk if this is required now
                            if(batchNum == len(train_dataloader)):
                                break
                            
                            val_lastFrameInBuffer = val_modelInputsMag[:,dataConfig.modelBufferFrames-1,:]
                            # print(f'lastFrameInBuffer.shape = {val_lastFrameInBuffer.shape}')

                            val_reshaped_input = val_modelInputsMag.view(val_modelInputsMag.shape[0], fft_freq_bins*dataConfig.modelBufferFrames)
                            #Model input is (Batchsize, 257*10) :: batch of 10 frames of 257 FFT bins
                            # ifftedOutputs = self.model(reshaped_input)
                            val_masksGenerated = self.model(val_reshaped_input)
                            # print(f'val_masksGenerated.shape = {val_masksGenerated.shape}')
                            val_outputs = val_lastFrameInBuffer*val_masksGenerated
                            
                            #Model output is (Batchsize, 512) :: batch of single IFFT-ed frame of 257 FFT bins
                            if(debugFlag): 
                                print(f'val_outputs.shape = {val_outputs.shape}')     

                            #Taking the first 32 samples from the ifft output
                            # firstSamples = ifftedOutputs[:,:dataConfig.stride_length] 

                            val_fftFrames = val_outputs * torch.exp(1j*val_phaseInfos)
                            # print(f'val_fftFrames.shape = {val_fftFrames.shape}')
                            

                            val_loss = self.loss_func(val_outputs, val_targetsMag)

                            # if(i == len(val_dataloader)/2):
                                # compareTwoAudios(val_firstSamples[5],val_targets[5])
                            #     # printQualityScores(val_targets[5],val_firstSamples[5],dataConfig.sample_rate)

                        running_vloss += val_loss


                    # Calculate average val loss
                    avg_vloss = running_vloss / len(val_dataloader)
                    print('LOSS train {} valid {}'.format(avg_trainloss, avg_vloss))

                    # Update the scheduler
                    self.scheduler.step(avg_vloss)

                    if(useWandB):
                        # Log results to W&B
                        wandb.log({
                            'trainLoss': avg_trainloss,
                            'valLoss': avg_vloss,
                        },step = i)
                    
                    #Save the model if the validation loss is good
                    if avg_vloss < self.best_vals['loss']:
                        self.best_vals['loss'] = avg_vloss
                        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                        torch.save(self.best_model, f'{wandbName}.pt')
            
                    # Check PESQ at EVERY EPOCH and log it in wandb
                    currentModel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                    reconstructedAudio,_ = runInferenceWithModel2(currentModel,self.modelcopy,dataConfig,exampleModelInputs,examplePhaseInfos,exampleCleanSpeech)
                    pesqScore = returnPesqScore(exampleCleanSpeech,reconstructedAudio,dataConfig.sample_rate)
                    print(f'pesqScore = {pesqScore}')
                    if(useWandB):
                        wandb.log({
                            'pesqScore': pesqScore,
                        },step = i)

                    # Store a model for best pesq score ( also upload audio to wandb)
                    if(pesqScore > self.best_vals['pesq']):
                        self.best_vals['pesq'] = pesqScore
                        self.best_pesqmodel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                        torch.save(self.best_pesqmodel, f'{wandbName}PESQ.pt')
                        if(useWandB):
                            wandb.log({'testAudioReconstructed': wandb.Audio(reconstructedAudio, caption="Reconstructed Speech", sample_rate= dataConfig.sample_rate)},step = i)

            if(useWandB):
                wandb.finish()

    # Training non realtime spectral masking
    def trainNonRT(self,train_dataloader,val_dataloader,dataConfig,modelSaveDir,wavFileTesting,wandbName,sweeping,useScheduler,debugFlag = False,useWandB = True,scaleFactor = 2):
        print("Training non realtime:",wandbName)
        name = wandbName
        exampleNoisySTFTData,exampleCleanSpeech, exampleNoisySpeech = makeFileReadyNonRT(wavFileTesting,dataConfig)
        pesqToBeat = returnPesqScore(exampleCleanSpeech,exampleNoisySpeech,dataConfig.sample_rate)
        print(f'pesqToBeat = {pesqToBeat}')

        if(useWandB):
                if(not sweeping):
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project="Shobhit_SEM9",
                        name= name,
                        config={
                            "epochs": self.num_epochs,
                            "learning_rate": dataConfig.learningRate,
                            "batch_size": dataConfig.batchSize,
                            # "stride_length": dataConfig.stride_length,
                            "frame_size": dataConfig.frameSize,
                            "sample_rate": dataConfig.sample_rate,
                            "duration": dataConfig.duration,
                            "n_fft": dataConfig.n_fft,
                            "shuffle": dataConfig.shuffle,
                            "dtype": dataConfig.dtype,
                        },
                    )
                wandb.log({'model parameters' : self.modelParameterCount,
                           'testCleanAudio': wandb.Audio(exampleCleanSpeech, caption="Clean Speech", sample_rate= dataConfig.sample_rate),
                           'testNoisyAudio': wandb.Audio(exampleNoisySpeech, caption="Noisy Speech", sample_rate= dataConfig.sample_rate),
                           'pesqToBeat': pesqToBeat})
                
        fft_freq_bins = dataConfig.numOfFreqBins

        with tqdm.trange(self.num_epochs, ncols=100) as t:
                for i in t:
                    # <Inside an epoch>
                    self.model.train(True)
                    self.optimizer.zero_grad()

                    running_trainloss = 0.0
                    randomSelectedBatch = np.random.randint(0,len(train_dataloader))

                    for batchNum,data in enumerate(train_dataloader):
                        # <Inside a batch>
                        modelInputs,targets = data
                        # print(f'modelInputs.dtype = {modelInputs.dtype}')
                        # print(f'modelInputs.shape = {modelInputs.shape}')
                        # print(f'targets.shape = {targets.shape}')
                    
                        angleInfo = torch.angle(modelInputs)
                        # print(f'angleInfo.shape = {angleInfo.shape}')

                        if(dataConfig.dtype == torch.float32):
                            modelInputsMag = torch.abs(modelInputs).float()
                            targetsMag = torch.abs(targets).float()
                        else:
                            modelInputsMag = torch.abs(modelInputs).double()
                            targetsMag = torch.abs(targets).double()
                            

                        if(batchNum == len(train_dataloader)):
                            break
                        
                        # ModelInputs = ([BatchSize, 257, 376])
                    
                        masksGenerated = self.model(modelInputsMag) 
                   
                        if(i%20 == 0 and batchNum == randomSelectedBatch):
                            uploadExampleMasks = []
                            printMask = True
                            for index,mask in enumerate(masksGenerated): 
                                status,toSkip = Trainer.tensorStatus(mask)
                                # print(f'Epoch_{i},Batch_{batchNum}: Mask_{index+1} status = {status}')
                                if(toSkip): continue

                                if(printMask):
                                    print(f'Batch_{batchNum}: Mask_{index+1} mask values = {mask}')
                                    printMask = False

                        outputs = modelInputsMag*masksGenerated
                        # outputs = torch.abs(masksGenerated)

                        # print(f'outputs.shape = {outputs.shape}')

                        # Have to take MSE loss in the REAL Frequency domain (not complex)
                        loss = self.loss_func(outputs, targetsMag)

                        reconstructed_spectrograms = torch.mul(outputs, torch.exp(1j * angleInfo))

                        reconstructedAudios = scaleFactor * torch.istft(reconstructed_spectrograms, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                        targetAudios =  scaleFactor * torch.istft(targets, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                        noisyAudios  = scaleFactor * torch.istft(modelInputs, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                        
                        #Check to see if the loss is being compared of okay audios
                        if(useWandB and i%50 == 0):
                            wandb.log({'InsideTrainingReconsAudio': wandb.Audio(reconstructedAudios[0].cpu().detach().numpy(), caption="Reconstructed Speech(insideTrainloop)", sample_rate= dataConfig.sample_rate)},step = i)
                            wandb.log({'InsideTrainingCleanAudio': wandb.Audio(targetAudios[0].cpu().detach().numpy(), caption="Target Speech(insideTrainloop)", sample_rate= dataConfig.sample_rate)},step = i)
                            wandb.log({'InsideTrainingNoisyAudio': wandb.Audio(noisyAudios[0].cpu().detach().numpy(), caption="Noisy Speech(insideTrainloop)", sample_rate= dataConfig.sample_rate)},step = i)


                        running_trainloss += loss
                        loss.backward()
                        self.optimizer.step()

                    # <After an epoch>
                    avg_trainloss = running_trainloss / len(train_dataloader)
                    # Check for validation loss!
                    running_vloss = 0.0
                    # Set the model to evaluation mode
                    self.model.eval()

                    with torch.no_grad():
                        for k,data in enumerate(val_dataloader):
                            val_modelInputs,val_targets = data
                            val_angleInfo = torch.angle(val_modelInputs)

                            if(dataConfig.dtype == torch.float32):
                                val_modelInputsMag = torch.abs(val_modelInputs).float()
                                val_targetsMag = torch.abs(val_targets).float()
                            else:
                                val_modelInputsMag = torch.abs(val_modelInputs).double()
                                val_targetsMag = torch.abs(val_targets).double()

                            if(batchNum == len(val_dataloader)):
                                break

                            val_masksGenerated = self.model(val_modelInputsMag) 
                            # print(f'val_masksGenerated.shape = {val_masksGenerated.shape}')

                            val_outputs = val_modelInputsMag*val_masksGenerated
                            # val_outputs = torch.abs(val_masksGenerated)

                            val_loss = self.loss_func(val_outputs, val_targetsMag)

                            val_reconstructed_spectrogram = torch.mul(val_outputs, torch.exp(1j * val_angleInfo))

                        running_vloss += val_loss

                    # Calculate average val loss
                    avg_vloss = running_vloss / len(val_dataloader)
                    print('LOSS train {} valid {}'.format(avg_trainloss, avg_vloss))

                    # Update the scheduler
                    if(useScheduler): self.scheduler.step(avg_vloss)

                    if(useWandB):
                        # Log results to W&B
                        wandb.log({
                            'trainLoss': avg_trainloss,
                            'valLoss': avg_vloss,
                        },step = i)
                    
                    #Save the model if the validation loss is good
                    if avg_vloss < self.best_vals['loss']:
                        self.best_vals['loss'] = avg_vloss
                        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                        torch.save(self.best_model, f'{wandbName}.pt')
            
                    # Check PESQ at EVERY EPOCH and log it in wandb
                    currentModel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                    reconstructedAudio = runInferenceNonRT(currentModel,self.modelcopy,dataConfig,exampleNoisySTFTData)
                    pesqScore = returnPesqScore(exampleCleanSpeech,reconstructedAudio,dataConfig.sample_rate)
                    
                    if(useWandB):
                        wandb.log({
                            'pesqScore': pesqScore,
                        },step = i)

                    # Store a model for best pesq score ( also upload audio to wandb)
                    if(pesqScore > self.best_vals['pesq']):
                        self.best_vals['pesq'] = pesqScore
                        self.best_pesqmodel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                        torch.save(self.best_pesqmodel, f'{wandbName}PESQ.pt')

                    if(useWandB and i % 20 == 0 and i != 0):
                        print(f'runningPesqScore = {pesqScore}')
                        wandb.log({'testAudioReconstructed': wandb.Audio(reconstructedAudio, caption="Reconstructed Speech", sample_rate= dataConfig.sample_rate)},step = i)

        if(useWandB):
            wandb.finish()

    def trainNonRTLossTD(self,train_dataloader,val_dataloader,dataConfig,modelSaveDir,wavFileTesting,wandbName,sweeping,useScheduler,debugFlag = False,useWandB = True,scaleFactor = 2):
        print("Training non realtime:",wandbName)
        name = wandbName
        exampleNoisySTFTData,exampleCleanSpeech, exampleNoisySpeech = makeFileReadyNonRT(wavFileTesting,dataConfig)
        pesqToBeat = returnPesqScore(exampleCleanSpeech,exampleNoisySpeech,dataConfig.sample_rate)
        print(f'pesqToBeat = {pesqToBeat}')

        if(useWandB):
                if(not sweeping):
                    wandb.init(
                        # set the wandb project where this run will be logged
                        project="Shobhit_SEM9",
                        name= name,
                        config={
                            "epochs": self.num_epochs,
                            "learning_rate": dataConfig.learningRate,
                            "batch_size": dataConfig.batchSize,
                            # "stride_length": dataConfig.stride_length,
                            "frame_size": dataConfig.frameSize,
                            "sample_rate": dataConfig.sample_rate,
                            "duration": dataConfig.duration,
                            "n_fft": dataConfig.n_fft,
                            "shuffle": dataConfig.shuffle,
                            "dtype": dataConfig.dtype,
                        },
                    )
                wandb.log({'model parameters' : self.modelParameterCount,
                           'testCleanAudio': wandb.Audio(exampleCleanSpeech, caption="Clean Speech", sample_rate= dataConfig.sample_rate),
                           'testNoisyAudio': wandb.Audio(exampleNoisySpeech, caption="Noisy Speech", sample_rate= dataConfig.sample_rate),
                           'pesqToBeat': pesqToBeat})
                
        fft_freq_bins = dataConfig.numOfFreqBins

        with tqdm.trange(self.num_epochs, ncols=100) as t:
                for i in t:
                    # <Inside an epoch>
                    self.model.train(True)
                    self.optimizer.zero_grad()

                    running_trainloss = 0.0
                    randomSelectedBatch = np.random.randint(0,len(train_dataloader))

                    for batchNum,data in enumerate(train_dataloader):
                        # <Inside a batch>
                        modelInputs,targets = data
                        # print(f'modelInputs.dtype = {modelInputs.dtype}')
                        # print(f'modelInputs.shape = {modelInputs.shape}')
                        # print(f'targets.shape = {targets.shape}')
                        

                        angleInfo = torch.angle(modelInputs)
                        # print(f'angleInfo.shape = {angleInfo.shape}')

                        if(dataConfig.dtype == torch.float32):
                            modelInputsMag = torch.abs(modelInputs).float()
                            targetsMag = torch.abs(targets).float()
                        else:
                            modelInputsMag = torch.abs(modelInputs).double()
                            targetsMag = torch.abs(targets).double()
                            

                        if(batchNum == len(train_dataloader)):
                            break
                        
                        # ModelInputs = ([BatchSize, 257, 376])
        
                        masksGenerated = self.model(modelInputsMag) 
                        # print(f'masksGenerated.shape = {masksGenerated.shape}')
                    
                        if(i%20 == 0 and batchNum == randomSelectedBatch):
                            uploadExampleMasks = []
                            printMask = True
                            for index,mask in enumerate(masksGenerated): 
                                status,toSkip = Trainer.tensorStatus(mask)
                                # print(f'Epoch_{i},Batch_{batchNum}: Mask_{index+1} status = {status}')
                                if(toSkip): continue

                                if(printMask):
                                    print(f'Batch_{batchNum}: Mask_{index+1} mask values = {mask}')
                                    printMask = False


                        outputs = modelInputsMag*masksGenerated
                        # outputs = torch.abs(masksGenerated)
                        reconstructed_spectrograms = torch.mul(outputs, torch.exp(1j * angleInfo))

                        reconstructedAudios = scaleFactor * torch.istft(reconstructed_spectrograms, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                        targetAudios =  scaleFactor * torch.istft(targets, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                        noisyAudios  = scaleFactor * torch.istft(modelInputs, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                        #Check to see if the loss is being compared of okay audios
                        if(useWandB and i%50 == 0):
                            wandb.log({'InsideTrainingReconsAudio': wandb.Audio(reconstructedAudios[0].cpu().detach().numpy(), caption="Reconstructed Speech(insideTrainloop)", sample_rate= dataConfig.sample_rate)},step = i)
                            wandb.log({'InsideTrainingCleanAudio': wandb.Audio(targetAudios[0].cpu().detach().numpy(), caption="Target Speech(insideTrainloop)", sample_rate= dataConfig.sample_rate)},step = i)
                            wandb.log({'InsideTrainingNoisyAudio': wandb.Audio(noisyAudios[0].cpu().detach().numpy(), caption="Noisy Speech(insideTrainloop)", sample_rate= dataConfig.sample_rate)},step = i)

                        loss = self.loss_func(reconstructedAudios, targetAudios)

                        running_trainloss += loss
                        loss.backward()
                        self.optimizer.step()

                    # <After an epoch>
                    avg_trainloss = running_trainloss / len(train_dataloader)
                    # Check for validation loss!
                    running_vloss = 0.0
                    # Set the model to evaluation mode
                    self.model.eval()

                    with torch.no_grad():
                        for k,data in enumerate(val_dataloader):
                            val_modelInputs,val_targets = data
                            val_angleInfo = torch.angle(val_modelInputs)

                            if(dataConfig.dtype == torch.float32):
                                val_modelInputsMag = torch.abs(val_modelInputs).float()
                                val_targetsMag = torch.abs(val_targets).float()
                            else:
                                val_modelInputsMag = torch.abs(val_modelInputs).double()
                                val_targetsMag = torch.abs(val_targets).double()

                            if(batchNum == len(val_dataloader)):
                                break

                            val_masksGenerated = self.model(val_modelInputsMag) 
                            # print(f'val_masksGenerated.shape = {val_masksGenerated.shape}')

                            val_outputs = val_modelInputsMag*val_masksGenerated
                            # val_outputs = torch.abs(val_masksGenerated)
                            val_reconstructed_spectrograms = torch.mul(val_outputs, torch.exp(1j * val_angleInfo))

                            val_reconstructedAudios = scaleFactor * torch.istft(val_reconstructed_spectrograms, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))
                            val_targetAudios = scaleFactor * torch.istft(val_targets, n_fft=dataConfig.n_fft, hop_length=dataConfig.frameSize//2, win_length=dataConfig.frameSize, window=torch.hann_window(dataConfig.frameSize))

                            val_loss = self.loss_func(val_reconstructedAudios, val_targetAudios)

                        running_vloss += val_loss

                    # Calculate average val loss
                    avg_vloss = running_vloss / len(val_dataloader)
                    print('LOSS train {} valid {}'.format(avg_trainloss, avg_vloss))

                    # Update the scheduler
                    if(useScheduler): self.scheduler.step(avg_vloss)

                    if(useWandB):
                        # Log results to W&B
                        wandb.log({
                            'trainLoss': avg_trainloss,
                            'valLoss': avg_vloss,
                        },step = i)
                    
                    #Save the model if the validation loss is good
                    if avg_vloss < self.best_vals['loss']:
                        self.best_vals['loss'] = avg_vloss
                        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                        torch.save(self.best_model, f'{wandbName}.pt')
            
                    # Check PESQ at EVERY EPOCH and log it in wandb
                    currentModel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                    reconstructedAudio = runInferenceNonRT(currentModel,self.modelcopy,dataConfig,exampleNoisySTFTData)
                    pesqScore = returnPesqScore(exampleCleanSpeech,reconstructedAudio,dataConfig.sample_rate)
                    
                    if(useWandB):
                        wandb.log({
                            'pesqScore': pesqScore,
                        },step = i)

                    # Store a model for best pesq score ( also upload audio to wandb)
                    if(pesqScore > self.best_vals['pesq']):
                        self.best_vals['pesq'] = pesqScore
                        self.best_pesqmodel = OrderedDict((k, v.detach().clone()) for k, v in self.model.state_dict().items())
                        torch.save(self.best_pesqmodel, f'{wandbName}PESQ.pt')

                    if(useWandB and i % 20 == 0 and i != 0):
                        print(f'runningPesqScore = {pesqScore}')
                        wandb.log({'testAudioReconstructed': wandb.Audio(reconstructedAudio, caption="Reconstructed Speech", sample_rate= dataConfig.sample_rate)},step = i)

        if(useWandB):
            wandb.finish()


    def tensorStatus(tensor):
        if tensor.numel() == 0:
            return "Tensor is empty",True
        elif (torch.all(tensor == 0)):
            return "Tensor is filled with zeros", True
        elif (torch.all(tensor == 1)):
            return "Tensor is filled with ones", True
        else:
            return "Tensor is not filled with zeros or ones" , False
                        


                












