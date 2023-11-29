import torch
import torch
import numpy as np
import soundfile as sf
import torch.fft as fft
import os
from tqdm import tqdm
from pesq import pesq
from scipy.signal.windows import hann
from scipy.signal import stft,istft

def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return wav

    # print("Calculating PESQ Score")
def returnPesqScore(waveform_clean_clip,waveform_noisy_clip, sample_rate):
    waveform_clean_clip = minMaxNorm(waveform_clean_clip)
    waveform_noisy_clip = minMaxNorm(waveform_noisy_clip)
    pesq_score = pesq(sample_rate, waveform_clean_clip, waveform_noisy_clip, 'nb')

    return pesq_score


def makeFileReady(wavFileName,dataConfig):
    # print("makeFileReady")
    cleanInputFullAudio = []
    noisyInputFullAudio = []
    fft_freq_bins = dataConfig.frameSize // 2 + 1

    filename = wavFileName
    speechSampleSize = dataConfig.duration * dataConfig.sample_rate
    # print("Speech Sample Size: ", speechSampleSize)
    noisySpeech,_ = sf.read(os.path.join(dataConfig.noisyPath, filename))
    cleanSpeech,_ = sf.read(os.path.join(dataConfig.cleanPath, filename))

    #Normalize
    noisySpeech = minMaxNorm(noisySpeech)
    cleanSpeech = minMaxNorm(cleanSpeech)
    # print("Noisy Speech Shape: ", noisySpeech.shape)
    # print("Clean Speech Shape: ", cleanSpeech.shape)

    numSubSamples = int(len(noisySpeech)/speechSampleSize)
    for i in range(numSubSamples):
        noisyInputFullAudio.append(noisySpeech[i*speechSampleSize:(i+1)*speechSampleSize])
        cleanInputFullAudio.append(cleanSpeech[i*speechSampleSize:(i+1)*speechSampleSize])

    # print("Number of Subsamples: ", numSubSamples)

    if(dataConfig.dtype == torch.float64) :
        dtype = np.float64
        complexDtype = np.complex128

    elif(dataConfig.dtype == torch.float32) :
        dtype = np.float32
        complexDtype = np.complex64

    modelInputs = []
    phaseInfos = []
    for idx, currNoisySample in enumerate(tqdm(noisyInputFullAudio)):
                    modelInputBuffer = np.zeros((dataConfig.modelBufferFrames,fft_freq_bins)).astype(complexDtype)
                    inbuffer = np.zeros((dataConfig.frameSize)).astype(dtype)
                    inbufferClean = np.zeros((dataConfig.frameSize)).astype(dtype)
                
                    for i in range(0, len(currNoisySample),dataConfig.stride_length):      
                    
                        if(i+dataConfig.stride_length > len(currNoisySample)-1):
                            break

                        #inbuffer is moved : [__s1__+++++++++++++__s2__] -> [+++++++++++++__s2__]
                        inbuffer[:-dataConfig.stride_length] = inbuffer[dataConfig.stride_length:] 
                        #inbuffer is filled with new data: [+++++++++++++__s2__] -> [+++++++++++++----]
                        inbuffer[-dataConfig.stride_length:] = currNoisySample[i : i + dataConfig.stride_length]

                        inbufferClean[:-dataConfig.stride_length] = inbufferClean[dataConfig.stride_length:] 
                        inbufferClean[-dataConfig.stride_length:] = noisyInputFullAudio[idx][i : i + dataConfig.stride_length]
                        
                        #Start up time
                        if i < dataConfig.frameSize:
                            continue
                    
                        # ModelInput Creation
                        buffer_array = np.array(inbuffer)
                        windowed_buffer = buffer_array * hann(len(buffer_array), sym=False)
                        frame = np.fft.rfft(windowed_buffer)    

                        phaseInfo = np.angle(frame)

                        modelInputBuffer[:-1, :] = modelInputBuffer[1:, :]
                        modelInputBuffer[-1, :] = frame
                        modelInputs.append(np.array(modelInputBuffer).astype(complexDtype))
                        phaseInfos.append(np.array(phaseInfo))

    return modelInputs,phaseInfos,cleanSpeech,noisySpeech

def runInferenceWithModel(ordDict,model,dataConfig,modelInputs,phaseInfos,cleanSpeech):
    # print("Running Inference with Model")
    fft_freq_bins = dataConfig.frameSize // 2 + 1
    model.load_state_dict(ordDict)
    model.eval()
    with torch.no_grad():
        ifftedOutputs = []
        for idx,modelInput in enumerate(modelInputs):
            modelInput = torch.from_numpy(modelInput).to(dataConfig.device)
            phaseInfo = torch.from_numpy(phaseInfos[idx]).to(dataConfig.device)
            if(dataConfig.dtype == torch.float32):
                modelInput = torch.abs(modelInput).float()
                # phaseInfo = phaseInfo.float()
            else:
                modelInput = torch.abs(modelInput).double()
                # phaseInfo = phaseInfo.double()
            
            # print("Input shape: ", modelInput.shape)
            reshapedInput = modelInput.view(fft_freq_bins*dataConfig.modelBufferFrames)
            # print("Reshaped Input shape: ", reshapedInput.shape)
            outputMag = model(reshapedInput)

            outputFFTFrame = outputMag * torch.exp(1j*phaseInfo)
            ifftedFrame = fft.irfft(outputFFTFrame)

            # print("Output shape: ", ifftedOutput.shape)
            ifftedOutputs.append(ifftedFrame.cpu().detach().numpy().reshape(dataConfig.frameSize))
         
    myRealtimeAudioSimulator = []

    reconstructedAudioTester = np.zeros_like(cleanSpeech)

    for idx,segment in enumerate(ifftedOutputs):
        addAtIdx = idx*dataConfig.stride_length
        addUntil = addAtIdx + dataConfig.frameSize

        # print(f'For idx = {idx}, np.add.at(reconstructedAudio2, range({addAtIdx}, {addUntil}), segment.lenth = {len(segment)})')
        np.add.at(reconstructedAudioTester, range(addAtIdx, addUntil), segment)
        
        #These indices will not have any dependency on future frames
        if(idx == len(ifftedOutputs) -1):
            myRealtimeAudioSimulator.extend(reconstructedAudioTester[addAtIdx:])
        else:
            myRealtimeAudioSimulator.extend(reconstructedAudioTester[addAtIdx:addAtIdx+dataConfig.stride_length])

    np_realtimeaudio = np.array(myRealtimeAudioSimulator)
    return np_realtimeaudio,cleanSpeech

def runInferenceWithModel2(ordDict,model,dataConfig,modelInputs,phaseInfos,cleanSpeech):
    # print("Running Inference with Model")
    fft_freq_bins = dataConfig.frameSize // 2 + 1
    model.load_state_dict(ordDict)
    model.eval()
    with torch.no_grad():
        ifftedOutputs = []
        for idx,modelInput in enumerate(modelInputs):
            modelInput = torch.from_numpy(modelInput).to(dataConfig.device)
            phaseInfo = torch.from_numpy(phaseInfos[idx]).to(dataConfig.device)
            if(dataConfig.dtype == torch.float32):
                modelInput = torch.abs(modelInput).float()
                # phaseInfo = phaseInfo.float()
            else:
                modelInput = torch.abs(modelInput).double()
                # phaseInfo = phaseInfo.double()
            
            # print("Input shape: ", modelInput.shape)
            reshapedInput = modelInput.view(1,fft_freq_bins*dataConfig.modelBufferFrames)
            # print("Reshaped Input shape: ", reshapedInput.shape)
            outputMag = model(reshapedInput)

            outputFFTFrame = outputMag * torch.exp(1j*phaseInfo)
            ifftedFrame = fft.irfft(outputFFTFrame)

            # print("Output shape: ", ifftedOutput.shape)
            ifftedOutputs.append(ifftedFrame.cpu().detach().numpy().reshape(dataConfig.frameSize))
         
    myRealtimeAudioSimulator = []

    reconstructedAudioTester = np.zeros_like(cleanSpeech)

    for idx,segment in enumerate(ifftedOutputs):
        addAtIdx = idx*dataConfig.stride_length
        addUntil = addAtIdx + dataConfig.frameSize

        # print(f'For idx = {idx}, np.add.at(reconstructedAudio2, range({addAtIdx}, {addUntil}), segment.lenth = {len(segment)})')
        np.add.at(reconstructedAudioTester, range(addAtIdx, addUntil), segment)
        
        #These indices will not have any dependency on future frames
        if(idx == len(ifftedOutputs) -1):
            myRealtimeAudioSimulator.extend(reconstructedAudioTester[addAtIdx:])
        else:
            myRealtimeAudioSimulator.extend(reconstructedAudioTester[addAtIdx:addAtIdx+dataConfig.stride_length])

    np_realtimeaudio = np.array(myRealtimeAudioSimulator)
    return np_realtimeaudio,cleanSpeech


def makeFileReadyNonRT(wavFileName,dataConfig):
    filename = wavFileName
    speechSampleSize = dataConfig.duration * dataConfig.sample_rate
    # print("Speech Sample Size: ", speechSampleSize)

    dtype = 'float64'
    if(dataConfig.dtype == torch.float32) :
            dtype = 'float32'

    noisySpeech,_ = sf.read(os.path.join(dataConfig.noisyPath, filename),dtype= dtype)
    cleanSpeech,_ = sf.read(os.path.join(dataConfig.cleanPath, filename),dtype= dtype)

    #Normalize
    noisySpeech = minMaxNorm(noisySpeech)
    cleanSpeech = minMaxNorm(cleanSpeech)

    noisySpeech = noisySpeech[:speechSampleSize]
    cleanSpeech = cleanSpeech[:speechSampleSize]

    _,_,noisy_spectrogram = stft(noisySpeech, 
                                             fs= dataConfig.sample_rate,
                                        nfft=dataConfig.n_fft, 
                                        # noverlap=self.dataconfig.stride_length, 
                                        nperseg=dataConfig.frameSize, 
                                        window='hann', 
                                        # center=True,
                                        # return_complex=True
                                        )
    
    # TODO: Problem when odd number of freq bins (later in convolutions)
    if(noisy_spectrogram.shape[0] % 2 != 0):
        noisy_spectrogram = noisy_spectrogram[:-1,:]
    
    return noisy_spectrogram,cleanSpeech,noisySpeech

    

# def runInferenceNonRT(ordDict,model,dataConfig,exampleNoisySTFTData):
#     print("runInferenceNonRT()")
#     model.load_state_dict(ordDict)
#     model.eval()

#     mag = np.abs(exampleNoisySTFTData)
#     angle = np.angle(exampleNoisySTFTData)

#     magTensor = torch.from_numpy(mag).to(dataConfig.device)
#     magTensor = magTensor.unsqueeze(-1) # ([257, 376, 1])
#     magTensor = magTensor.permute(2, 0, 1) # ([1, 257, 376])
#     print(f'magTensor.shape = {magTensor.shape}')

#     maskGenerated = model(magTensor)
#     print(f'masksGenerated.shape = {maskGenerated.shape}')

#     permutedMask = maskGenerated.permute(1, 2, 0) # ([257, 376, 1])
#     permutedMask = permutedMask.squeeze(-1) # ([257, 376])
#     print(f'permutedMask.shape = {permutedMask.shape}')

#     output = magTensor*maskGenerated
#     print(f'output.shape = {output.shape}')

#     reconstructed_spectrogram = torch.mul(output, torch.exp(1j * angle))
#     reconstructed_spectrogram = reconstructed_spectrogram.numpy()
#     print(f'reconstructed_spectrogram.shape = {reconstructed_spectrogram.shape}')

#     _,denoisedSignal = istft(Zxx= reconstructed_spectrogram,
#                              fs= dataConfig.sample_rate,
#                             nfft= dataConfig.n_fft, 
#                             # noverlap=self.dataconfig.stride_length, 
#                             nperseg= dataConfig.frameSize, 
#                             window='hann', 
#                             # center=True,
#                             # return_complex=True
#                             )
#     return denoisedSignal


def runInferenceNonRT(ordDict,model,dataConfig,exampleNoisySTFTData):
    # print("runInferenceNonRT()")
    model.load_state_dict(ordDict)
    model.eval()

    exampleNoisySTFTData = np.expand_dims(exampleNoisySTFTData, axis=0)
    modelInputs = torch.from_numpy(exampleNoisySTFTData).to(dataConfig.device)
    # print(f'runInferenceNonRT(): exampleNoisySTFTData.shape = {exampleNoisySTFTData.shape}')

    angleInfo = torch.angle(modelInputs)
    # print(f'angleInfo.shape = {angleInfo.shape}')

    if(dataConfig.dtype == torch.float32):
        modelInputsMag = torch.abs(modelInputs).float()
    else:
        modelInputsMag = torch.abs(modelInputs).double()

    masksGenerated = model(modelInputsMag) 
    
    outputs = modelInputsMag*masksGenerated

    reconstructed_spectrogram = torch.mul(outputs, torch.exp(1j * angleInfo))
    reconstructed_spectrogram = reconstructed_spectrogram.cpu().detach().numpy()
    # print(f'runInferenceNonRT(): reconstructed_spectrogram.shape = {reconstructed_spectrogram.shape}')

    reconstructed_spectrogram = reconstructed_spectrogram.squeeze(axis = 0)

    _,denoisedSignal = istft(Zxx= reconstructed_spectrogram,
                             fs= dataConfig.sample_rate,
                            nfft= dataConfig.n_fft, 
                            # noverlap=self.dataconfig.stride_length, 
                            nperseg= dataConfig.frameSize, 
                            window='hann', 
                            # center=True,
                            # return_complex=True
                            )
    return denoisedSignal