# %%
import torch
from torch.utils.data import Dataset
import soundfile as sf
import os, fnmatch
import numpy as np
import torch.nn as nn
import torch.fft as fft
from sklearn.model_selection import train_test_split
from scipy.signal.windows import hann
from scipy.signal import stft
from tqdm import tqdm

#Data Normalization
def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return wav

class DataConfig():
    def __init__(self, 
                 frameSize = 512, 
                 stride_length = 32,
                 sample_rate = 16000,
                 duration = 3,
                 n_fft = 512,
                 modelBufferFrames = 10,
                 batchSize = 32,
                 shuffle = True,
                 noisyPath = 'dataset/train/',
                 cleanPath = 'dataset/y_train/',
                 dtype = torch.float64,
                 device = 'cpu',
                 learningRate = 0.001,
                ):
        
        self.frameSize = frameSize
        self.stride_length = stride_length
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.modelBufferFrames = modelBufferFrames
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.noisyPath = noisyPath
        self.cleanPath = cleanPath
        self.dtype = dtype
        self.device = device
        self.learningRate = learningRate

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target_data = self.targets[idx]
        return input_data, target_data
    

# class CustomDataset2(Dataset):
#     def __init__(self, inputs, targets):
#         self.inputs = inputs
#         self.targets = targets

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         input_data = self.inputs[idx]
#         target_data = self.targets[idx]
#         return input_data, target_data

class CustomDataloaderCreator():
    def __init__(self, 
                 noisy_files_list, 
                 clean_files_list,
                 test_noisy_files_list,
                 test_clean_files_list,
                 val_noisy_files_list,
                 val_clean_files_list,
                 dataconfig,
                 filesToUse = 500
                 ):
        
        self.debugFlag = True

        #Training Dataset
        self.noisy_files_list = noisy_files_list
        self.clean_files_list = clean_files_list
        self.trainNoisyDataset = []
        self.trainCleanDataset = []
        self.trainModelInputBuffers = []
        self.targets = []

        self.trainSpectrogramData = []
        self.targetsNonRT = []

        #Test Dataset
        self.test_noisy_files_list = test_noisy_files_list
        self.test_clean_files_list = test_clean_files_list
        self.testNoisyDataset = []
        self.testCleanDataset = []
        self.test_modelInputBuffers = []
        self.trainPhaseInfo = []
        self.test_targets = []

        self.testSpectrogramData = []
        self.test_targetsNonRT = []
        
        #Validation Dataset
        self.val_noisy_files_list = val_noisy_files_list
        self.val_clean_files_list = val_clean_files_list
        self.validationNoisyDataset = []
        self.validationCleanDataset = []
        self.val_modelInputBuffers = []
        self.val_phaseInfo = []
        self.val_targets = []

        self.val_spectrogramData = []
        self.val_targetsNonRT = []

        self.filesToUse = filesToUse
        
        self.dataconfig = dataconfig

        if(dataconfig.dtype == torch.float64) :
            self.dtype = np.float64
            self.complexDtype = np.complex128
        
        elif(dataconfig.dtype == torch.float32) :
            self.dtype = np.float32
            self.complexDtype = np.complex64
        
    #Creates the specified duration audio clips from the noisy and clean files, 
    # defines the trainNoisyDataset,trainCleanDataset,validationNoisyDataset,validationCleanDataset
    def createAudioClips(self):
        print("CustomDataLoader.createAudioClips()")
        speechSampleSize = self.dataconfig.duration * self.dataconfig.sample_rate

        listIter = [self.noisy_files_list, self.val_noisy_files_list]
        datasetIter = [(self.trainNoisyDataset, self.trainCleanDataset), (self.validationNoisyDataset, self.validationCleanDataset)]
        NOISY = 0
        CLEAN = 1
        TEST = 2    #Not used yet

        #Create the training and then validation dataset
        for index,currlist in enumerate(listIter):
            for idx,filename in enumerate(currlist):
                if idx == self.filesToUse:
                    break

                noisySpeech,_ = sf.read(os.path.join(self.dataconfig.noisyPath, filename))
                cleanSpeech,_ = sf.read(os.path.join(self.dataconfig.cleanPath, filename))

                #Normalize
                noisySpeech = minMaxNorm(noisySpeech)
                cleanSpeech = minMaxNorm(cleanSpeech)

                numSubSamples = int(len(noisySpeech)/speechSampleSize)
                for i in range(numSubSamples):
                    datasetIter[index][NOISY].append(noisySpeech[i*speechSampleSize:(i+1)*speechSampleSize])
                    datasetIter[index][CLEAN].append(cleanSpeech[i*speechSampleSize:(i+1)*speechSampleSize])
        
    #This function creates train and validation inputs and targets
    # Input : Frequency Domain 10 frame buffer (size 10*framesize)
    # Target : Time Domain 2 ms clean speech (size strideLength)
    def createModelBufferInputs(self):
        print("CustomDataLoader.createModelBufferInputs()")
       
        fft_freq_bins = int(self.dataconfig.n_fft/2) + 1

        datasetIter = [(self.trainNoisyDataset, self.trainCleanDataset), (self.validationNoisyDataset, self.validationCleanDataset)]
        modelBufferFramesIter = [self.trainModelInputBuffers, self.val_modelInputBuffers]
        targetsIter = [self.targets, self.val_targets]
        
        NOISY = 0
        CLEAN = 1
       
        #Create the training and validation inputs and targets
        for index,data in enumerate(datasetIter):
            currNoisyDataset = data[NOISY]
            corrCleanDataset = data[CLEAN]
            print(f'xFrames (expectedFrames) per audio clip = {len(currNoisyDataset[0])//self.dataconfig.stride_length}')
            for idx, currNoisySample in enumerate(tqdm(currNoisyDataset)):
                modelInputBuffer = np.zeros((self.dataconfig.modelBufferFrames,fft_freq_bins)).astype(self.complexDtype)
                inbuffer = np.zeros((self.dataconfig.frameSize)).astype(self.dtype)

                for i in range(0, len(currNoisySample),self.dataconfig.stride_length):      

                    if(i+self.dataconfig.stride_length > len(currNoisySample)-1):
                        break

                    #inbuffer is moved : [__s1__+++++++++++++__s2__] -> [+++++++++++++__s2__]
                    inbuffer[:-self.dataconfig.stride_length] = inbuffer[self.dataconfig.stride_length:] 
                    #inbuffer is filled with new data: [+++++++++++++__s2__] -> [+++++++++++++----]
                    inbuffer[-self.dataconfig.stride_length:] = currNoisySample[i : i + self.dataconfig.stride_length]

                    #Start up time
                    if i < self.dataconfig.frameSize:
                        continue

                    buffer_array = np.array(inbuffer)
                    windowed_buffer = buffer_array * hann(len(buffer_array), sym=False)

                    # Taking the real-valued FFT
                    frame = np.fft.rfft(windowed_buffer)    

                    # if(self.debugFlag):
                    #     print(f'frame.shape = {frame.shape}')
            
                    # Shift the modelInputBuffer
                    modelInputBuffer[:-1, :] = modelInputBuffer[1:, :]

                    # Fill the last row of modelInputBuffer with the new spectrogram values
                    modelInputBuffer[-1, :] = frame
                    modelBufferFramesIter[index].append(np.array(modelInputBuffer))
                    targetsIter[index].append(np.array(corrCleanDataset[idx][i:i+self.dataconfig.stride_length]))
            
            # #Shuffle up the dataset if required
            # if self.dataconfig.shuffle:
            #     self.indices = np.random.permutation(len(modelBufferFramesIter[index]))
            # else:
            #     self.indices = np.arange(len(modelBufferFramesIter[index]))
            # print("CustomDataLoader.createModelBufferInputs(): modelinputbuffers size = ", len(self.trainModelInputBuffers))
            # print(f'gotten frames per modelinputbuffer = {len(self.trainModelInputBuffers)//len(self.trainNoisyDataset)}')
            # print("CustomDataLoader.createModelBufferInputs(): targets size = ", len(self.targets))

    #This function creates train and validation inputs and targets
    # Input : Frequency Domain 10 frame buffer (size 10*framesize)
    # Target : Frequency Domain 1 frame buffer (size framesize)
    def createModelBufferInputs2(self):
        print("CustomDataLoader.createModelBufferInputs2()")
       
        fft_freq_bins = int(self.dataconfig.n_fft/2) + 1

        datasetIter = [(self.trainNoisyDataset, self.trainCleanDataset), (self.validationNoisyDataset, self.validationCleanDataset)]
        modelBufferFramesIter = [self.trainModelInputBuffers, self.val_modelInputBuffers]
        targetsIter = [self.targets, self.val_targets]
        
        NOISY = 0
        CLEAN = 1
       
        #Create the training and validation inputs and targets
        for index,data in enumerate(datasetIter):
            currNoisyDataset = data[NOISY]
            corrCleanDataset = data[CLEAN]
            print(f'xFrames (expectedFrames) per audio clip = {len(currNoisyDataset[0])//self.dataconfig.stride_length}')
            for idx, currNoisySample in enumerate(tqdm(currNoisyDataset)):
                modelInputBuffer = np.zeros((self.dataconfig.modelBufferFrames,fft_freq_bins)).astype(self.complexDtype)
                inbuffer = np.zeros((self.dataconfig.frameSize)).astype(self.dtype)
                inbufferClean = np.zeros((self.dataconfig.frameSize)).astype(self.dtype)
            
                for i in range(0, len(currNoisySample),self.dataconfig.stride_length):      
                
                    if(i+self.dataconfig.stride_length > len(currNoisySample)-1):
                        break

                    #inbuffer is moved : [__s1__+++++++++++++__s2__] -> [+++++++++++++__s2__]
                    inbuffer[:-self.dataconfig.stride_length] = inbuffer[self.dataconfig.stride_length:] 
                    #inbuffer is filled with new data: [+++++++++++++__s2__] -> [+++++++++++++----]
                    inbuffer[-self.dataconfig.stride_length:] = currNoisySample[i : i + self.dataconfig.stride_length]

                    inbufferClean[:-self.dataconfig.stride_length] = inbufferClean[self.dataconfig.stride_length:] 
                    inbufferClean[-self.dataconfig.stride_length:] = corrCleanDataset[idx][i : i + self.dataconfig.stride_length]
                    
                    #Start up time
                    if i < self.dataconfig.frameSize:
                        continue

                    # ModelInput Creation
                    buffer_array = np.array(inbuffer)
                    windowed_buffer = buffer_array * hann(len(buffer_array), sym=False)
                    frame = np.fft.rfft(windowed_buffer)    
                    modelInputBuffer[:-1, :] = modelInputBuffer[1:, :]
                    modelInputBuffer[-1, :] = frame
                    modelBufferFramesIter[index].append(np.array(modelInputBuffer))

                    # Target Creation
                    clean_buffer_array = np.array(inbufferClean)
                    clean_windowed_buffer = buffer_array * hann(len(clean_buffer_array), sym=False)
                    clean_frame = np.fft.rfft(clean_windowed_buffer)
                    clean_iffted_segment = np.fft.irfft(clean_frame)
                    targetsIter[index].append(clean_iffted_segment)

    # This function creates train and validation inputs for spectral masking case, also sends in the phase info
    # Input : Frequency Domain 10 frame buffer (size 10*framesize)
    # Target : Frequency Domain 1 frame buffer's MASK (size framesize)
    def createModelBufferInputs3(self):
        print("CustomDataLoader.createModelBufferInputs3()")
       
        fft_freq_bins = int(self.dataconfig.n_fft/2) + 1

        datasetIter = [(self.trainNoisyDataset, self.trainCleanDataset), (self.validationNoisyDataset, self.validationCleanDataset)]
        modelBufferFramesIter = [self.trainModelInputBuffers, self.val_modelInputBuffers]
        phaseInfoIter = [self.trainPhaseInfo, self.val_phaseInfo]
        targetsIter = [self.targets, self.val_targets]
        
        NOISY = 0
        CLEAN = 1
       
        #Create the training and validation inputs and targets
        for index,data in enumerate(datasetIter):
            currNoisyDataset = data[NOISY]
            corrCleanDataset = data[CLEAN]
            print(f'xFrames (expectedFrames) per audio clip = {len(currNoisyDataset[0])//self.dataconfig.stride_length}')
            for idx, currNoisySample in enumerate(tqdm(currNoisyDataset)):
                modelInputBuffer = np.zeros((self.dataconfig.modelBufferFrames,fft_freq_bins)).astype(self.complexDtype)
                inbuffer = np.zeros((self.dataconfig.frameSize)).astype(self.dtype)
                inbufferClean = np.zeros((self.dataconfig.frameSize)).astype(self.dtype)
            
                for i in range(0, len(currNoisySample),self.dataconfig.stride_length):      
                
                    if(i+self.dataconfig.stride_length > len(currNoisySample)-1):
                        break

                    #inbuffer is moved : [__s1__+++++++++++++__s2__] -> [+++++++++++++__s2__]
                    inbuffer[:-self.dataconfig.stride_length] = inbuffer[self.dataconfig.stride_length:] 
                    #inbuffer is filled with new data: [+++++++++++++__s2__] -> [+++++++++++++----]
                    inbuffer[-self.dataconfig.stride_length:] = currNoisySample[i : i + self.dataconfig.stride_length]

                    inbufferClean[:-self.dataconfig.stride_length] = inbufferClean[self.dataconfig.stride_length:] 
                    inbufferClean[-self.dataconfig.stride_length:] = corrCleanDataset[idx][i : i + self.dataconfig.stride_length]
                    
                    #Start up time
                    if i < self.dataconfig.frameSize:
                        continue

                    # ModelInput Creation
                    buffer_array = np.array(inbuffer)
                    windowed_buffer = buffer_array * hann(len(buffer_array), sym=False)
                    frame = np.fft.rfft(windowed_buffer)    

                    # Phase Info
                    phaseInfo = np.angle(frame)

                    modelInputBuffer[:-1, :] = modelInputBuffer[1:, :]
                    modelInputBuffer[-1, :] = frame
                    modelBufferFramesIter[index].append(np.array(modelInputBuffer))
                    phaseInfoIter[index].append(np.array(phaseInfo))

                    # Target Creation
                    clean_buffer_array = np.array(inbufferClean)
                    clean_windowed_buffer = buffer_array * hann(len(clean_buffer_array), sym=False)
                    clean_frame = np.fft.rfft(clean_windowed_buffer)
                    targetsIter[index].append(clean_frame)

    #TODO: Include stridelength here
    def createDataForNonRT(self):
        print("CustomDataLoader.createDataForNonRT()")

        dataSetIter = [(self.trainNoisyDataset, self.trainCleanDataset), (self.validationNoisyDataset, self.validationCleanDataset)]
        dataIter = [(self.trainSpectrogramData,self.targetsNonRT), (self.val_spectrogramData,self.val_targetsNonRT)]
        
        # Preparing train and validation data for non-RT
        for index in range(len(dataSetIter)):
            currNoisyDataset,currCleanDataset = dataSetIter[index]
            currSpectrogramData,currTargets = dataIter[index]
            for index,noisyAudioSample in enumerate(currNoisyDataset): 

                _,_,noisy_spectrogram = stft(noisyAudioSample, 
                                             fs= self.dataconfig.sample_rate,
                                        nfft=self.dataconfig.n_fft, 
                                        # noverlap=self.dataconfig.stride_length, 
                                        nperseg=self.dataconfig.frameSize, 
                                        window='hann', 
                                        # center=True,
                                        # return_complex=True
                                        )

                _,_,clean_spectrogram = stft(currCleanDataset[index],
                                        fs= self.dataconfig.sample_rate,
                                        nfft=self.dataconfig.n_fft, 
                                        # noverlap=self.dataconfig.stride_length, 
                                        nperseg=self.dataconfig.frameSize, 
                                        window='hann', 
                                        # center=True,
                                        # return_complex=True
                                        )
    
                currSpectrogramData.append(noisy_spectrogram)
                currTargets.append(clean_spectrogram)

    # This function creates the test dataset 
    def createTestDataset(self):
        print("CustomDataLoader.createTestDataset()")
        speechSampleSize = self.dataconfig.duration * self.dataconfig.sample_rate
        for index,filename in enumerate(self.test_noisy_files_list):
            if index == 100:
                break

            noisySpeech,_ = sf.read(os.path.join(self.dataconfig.noisyPath, filename))
            cleanSpeech,_ = sf.read(os.path.join(self.dataconfig.cleanPath, filename))

            #Normalize
            noisySpeech = minMaxNorm(noisySpeech)
            cleanSpeech = minMaxNorm(cleanSpeech)

            numSubSamples = int(len(noisySpeech)/speechSampleSize)
            for i in range(numSubSamples):
                self.testNoisyDataset.append(noisySpeech[i*speechSampleSize:(i+1)*speechSampleSize])
                self.testCleanDataset.append(cleanSpeech[i*speechSampleSize:(i+1)*speechSampleSize])

        print("Test Noisy Dataset Size: ", len(self.testNoisyDataset))
        print("Test Clean Dataset Size: ", len(self.testCleanDataset))

    #Call this function to prepare the dataloader
    def prepare(self):
        self.createAudioClips()
        # self.createModelBufferInputs()
        # self.createModelBufferInputs2()
        self.createModelBufferInputs3()
        self.printMembers()

    def prepareNonRT(self):
        self.createAudioClips()
        self.createDataForNonRT()
        self.printNonRTMembers()

    def getNumOfFrames(self):
        return self.trainSpectrogramData[0].shape[1]
    
    def getNumOfFreqBins(self):
        return self.trainSpectrogramData[0].shape[0]
            
    def getTrainDataloader(self):
        trainingDataset = CustomDataset(
            np.array(self.trainModelInputBuffers).astype(self.complexDtype),
            np.array(self.targets).astype(self.dtype)
            )

        return torch.utils.data.DataLoader(trainingDataset,
            batch_size = self.dataconfig.batchSize,
            shuffle = self.dataconfig.shuffle,
            generator = torch.Generator(device= self.dataconfig.device)
            )
    
    def getValidationDataloader(self):
        validationDataset = CustomDataset(
            np.array(self.val_modelInputBuffers).astype(self.complexDtype),
            np.array(self.val_targets).astype(self.dtype)
            )

        return torch.utils.data.DataLoader(validationDataset,
                        batch_size = self.dataconfig.batchSize, 
                        shuffle = self.dataconfig.shuffle,
                        generator = torch.Generator(device=self.dataconfig.device)
                        )
    

    # Spectral masking loaders ==================================================================================================================
    def getTrainDataloader2(self):

        array1 = np.array(self.trainModelInputBuffers).astype(self.complexDtype)
        len = array1.shape[0]
        array2 = np.array(self.trainPhaseInfo).astype(self.dtype)

        reshaped_array1 = array1.reshape(len, -1)
        combined_array = np.concatenate([reshaped_array1, array2], axis=1)
        print("combined_array shape ", {combined_array.shape})

        trainingDataset = CustomDataset(
            combined_array,
            np.array(self.targets).astype(self.complexDtype)
            )

        return torch.utils.data.DataLoader(trainingDataset,
            batch_size = self.dataconfig.batchSize,
            shuffle = self.dataconfig.shuffle,
            generator = torch.Generator(device= self.dataconfig.device)
            )

    def getValidationDataloader2(self):
        array1 = np.array(self.val_modelInputBuffers).astype(self.complexDtype)
        array2 = np.array(self.val_phaseInfo).astype(self.dtype)
        len = array1.shape[0]
    
        reshaped_array1 = array1.reshape(len, -1)
        combined_array = np.concatenate([reshaped_array1, array2], axis=1)
        print("combined_array shape ", {combined_array.shape})

        # # Create a structured array with two fields: 'field1' and 'field2'
        # val_combined_array_structured = np.empty((len(array1),), dtype=[('input', array1.dtype, (bufferlen, framesize)), ('phaseInfo', array2.dtype, (pi_len,))])

        # # Assign values to the fields
        # val_combined_array_structured['input'] = array1
        # val_combined_array_structured['phaseInfo'] = array2

        # # Check the shape of the structured array
        # print(f'val_combinedInput shape = {val_combined_array_structured.shape}')
        # print("val_combined_array_structured[\'input\'] shape ", {val_combined_array_structured['input'].shape})
        # print("val_combined_array_structured[\'phaseInfo\'] shape ",{val_combined_array_structured['phaseInfo'].shape})

        validationDataset = CustomDataset(
            combined_array,
            np.array(self.val_targets).astype(self.complexDtype)
            )

        return torch.utils.data.DataLoader(validationDataset,
                        batch_size = self.dataconfig.batchSize, 
                        shuffle = self.dataconfig.shuffle,
                        generator = torch.Generator(device=self.dataconfig.device)
                        )
    
    # NonRT dataloaders ==================================================================================================================
    def getTrainDataloaderNonRT(self):
        trainingDataset = CustomDataset(
            np.array(self.trainSpectrogramData).astype(self.complexDtype),
            np.array(self.targetsNonRT).astype(self.complexDtype)
            )

        return torch.utils.data.DataLoader(trainingDataset,
            batch_size = self.dataconfig.batchSize,
            shuffle = self.dataconfig.shuffle,
            generator = torch.Generator(device= self.dataconfig.device)
            )
    
    def getValidationDataloaderNonRT(self):
        validationDataset = CustomDataset(
            np.array(self.val_spectrogramData).astype(self.complexDtype),
            np.array(self.val_targetsNonRT).astype(self.complexDtype)
            )

        return torch.utils.data.DataLoader(validationDataset,
                        batch_size = self.dataconfig.batchSize, 
                        shuffle = self.dataconfig.shuffle,
                        generator = torch.Generator(device=self.dataconfig.device)
                        )

    # Printing helpers ==================================================================================================================
    def printMembers(self):
        print('--------------------------DISPLAY---------------------------------------------')
        print(f'noisy_files_list.shape = {np.array(self.noisy_files_list).shape}')
        print(f'clean_files_list.shape = {np.array(self.clean_files_list).shape}')
        print(f'trainNoisyDataset.shape = {len(self.trainNoisyDataset)}')
        print(f'trainCleanDataset.shape = {len(self.trainCleanDataset)}')
        print(f'trainModelInputBuffers.shape = {len(self.trainModelInputBuffers)},{len(self.trainModelInputBuffers[0])}')
        print(f'targets.shape = {len(self.targets)}')

        print('-----------------------------------------------------------------------')
        print(f'test_noisy_files_list.shape = {np.array(self.test_noisy_files_list).shape}')
        print(f'test_clean_files_list.shape = {np.array(self.test_clean_files_list).shape}')
        print(f'testNoisyDataset.shape = {len(self.testNoisyDataset)}')
        print(f'testCleanDataset.shape = {len(self.testCleanDataset)}')
        print('-----------------------------------------------------------------------')
        print(f'val_noisy_files_list.shape = {np.array(self.val_noisy_files_list).shape}')
        print(f'val_clean_files_list.shape = {np.array(self.val_clean_files_list).shape}')
        print(f'validationNoisyDataset.shape = {len(self.validationNoisyDataset)}')
        print(f'validationCleanDataset.shape = {len(self.validationCleanDataset)}')
        print(f'val_modelInputBuffers.shape = {len(self.val_modelInputBuffers)},{len(self.val_modelInputBuffers[0])}')
        print(f'val_targets.shape = {len(self.val_targets)}')

    def printNonRTMembers(self):
        print('--------------------------DISPLAY---------------------------------------------')
        print(f'noisy_files_list.shape = {np.array(self.noisy_files_list).shape}')
        print(f'clean_files_list.shape = {np.array(self.clean_files_list).shape}')
        print(f'trainNoisyDataset.shape = {len(self.trainNoisyDataset)}')
        print(f'trainCleanDataset.shape = {len(self.trainCleanDataset)}')
        print(f'trainSpectrogramData.shape = {np.array(self.trainSpectrogramData).shape}')
        print(f'trainTargets.shape = {np.array(self.targetsNonRT).shape}')
    
        print('-----------------------------------------------------------------------')
        print(f'val_noisy_files_list.shape = {np.array(self.val_noisy_files_list).shape}')
        print(f'val_clean_files_list.shape = {np.array(self.val_clean_files_list).shape}')
        print(f'validationNoisyDataset.shape = {len(self.validationNoisyDataset)}')
        print(f'validationCleanDataset.shape = {len(self.validationCleanDataset)}')
        print(f'val_spectrogramData.shape = {np.array(self.val_spectrogramData).shape}')
        print(f'val_targets.shape = {np.array(self.val_targetsNonRT).shape}')
