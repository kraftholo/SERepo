import sounddevice as sd
import IPython.display as ipd
import numpy as np
# from pesq import pesq
import matplotlib.pyplot as plt
# from pystoi import stoi


# import torch
# import torch.fft as fft
from scipy.signal import correlate

# import librosa
import librosa.display
from scipy.signal.windows import hann
import wandb

# %% [markdown]
# ### # <font color="green">Helper functions</font>

# %%
# print(f'torchaudio backend : {torchaudio.get_audio_backend()}')

# def minMaxNorm(wav, eps=1e-8):
#     max = np.max(abs(wav))
#     min = np.min(abs(wav))
#     wav = (wav - min) / (max - min + eps)
#     return wav

# def plot_waveform(waveform, sample_rate):
#     waveform = waveform.numpy()

#     num_channels, num_frames = waveform.shape
#     time_axis = torch.arange(0, num_frames) / sample_rate

#     figure, axes = plt.subplots(num_channels, 1)
#     if num_channels == 1:
#         axes = [axes]
#     for c in range(num_channels):
#         axes[c].plot(time_axis, waveform[c], linewidth=1)
#         axes[c].grid(True)
#         if num_channels > 1:
#             axes[c].set_ylabel(f"Channel {c+1}")
#     figure.suptitle("waveform")
#     plt.show(block=False)

# def printQualityScores(waveform_clean_clip,waveform_noisy_clip, sample_rate,epoch,batch):
#     waveform_clean_clip = waveform_clean_clip.detach().numpy()
#     waveform_noisy_clip = waveform_noisy_clip.detach().numpy()
#     print(f'printQualityScores():waveform_clean_clip shape = {waveform_clean_clip.shape}')
#     print(f'printQualityScores():waveform_noisy_clip shape = {waveform_noisy_clip.shape}')
#     pesq_score_test = pesq(sample_rate, waveform_clean_clip, waveform_clean_clip, 'nb')
#     pesq_score_noisy = pesq(sample_rate, waveform_clean_clip, waveform_noisy_clip, 'nb')

#     stoi_score_test = stoi(waveform_clean_clip, waveform_clean_clip, sample_rate, extended=False)
#     stoi_score_noisy = stoi(waveform_clean_clip,  waveform_noisy_clip, sample_rate, extended=False)

#     print(f'PESQ score for clean (baseline) = {pesq_score_test}')
#     print(f'PESQ score for noisy = {pesq_score_noisy}')
#     print("---------------------------------------------------")
#     print(f'STOI score for clean (baseline) = {stoi_score_test}')
#     print(f'STOI score for noisy = {stoi_score_noisy}')


def compareTwoAudios(modelOutput,target,epoch,batch,sampling_rate = 16000,logInWandb = False):
    # print(f'compareTwoAudios():modelOutput shape = {modelOutput.shape}')
    # print(f'compareTwoAudios():target shape = {target.shape}')
    modelOutput = modelOutput.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    time = np.arange(len(modelOutput))

    # Plot the original signal with transparency
    plt.figure(figsize=(12, 6))
    plt.plot(time, modelOutput, label='Model Output', alpha=0.5)

    # Plot the reconstructed signal on top with transparency
    plt.plot(time, target, label='Target', alpha=0.5)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Random datapoint from Epoch: {epoch} and Batch:{batch}')
    plt.legend()


    # # Compute and plot the Fourier transforms
    # input_fft = np.fft.fft(input)
    # reconstructed_fft = np.fft.fft(reconstructed)
    # freq = np.fft.fftfreq(len(input), 1 / sampling_rate)

    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 1, 1)
    # plt.plot(freq, np.abs(input_fft), label='Original Signal')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Fourier Transform of Original Signal')
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(freq, np.abs(reconstructed_fft), label='Reconstructed Signal')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Fourier Transform of Reconstructed Signal')
    # plt.grid()
    # plt.tight_layout()

    
    # mse = (input - reconstructed) ** 2
    # Plot the correlation
    # plt.figure(figsize=(8, 4))
    # plt.plot(time, mse,color='red') 
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Square Error between Original and Reconstructed Signals')
    # plt.grid()
    # plt.show()

    # Calculate the cross-correlation
    correlation = np.correlate(target, modelOutput, mode='full')
    # Create a time array for the correlation
    time = np.arange(-len(target) + 1, len(target))

    # Plot the correlation
    plt.figure(figsize=(12, 6))
    plt.plot(time, correlation)
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Correlation')
    plt.title('Cross-Correlation between Target and Model Output')
    plt.grid()

    if(logInWandb and (epoch%10 == 0)):
        plot_filename = f'Epoch_{epoch}_Batch_{batch}.png'
        plt.savefig(plot_filename)
        wandb.log({f'training_loss_plot_epoch_{epoch}': wandb.Image(plot_filename)}, step=epoch)
    else:
        plt.show()