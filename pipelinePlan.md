# Pipeline Plan
PESQ,MOS,ESTOI,MSE,SNR,STOI,SI-SDR,SI-SNR

## Data Collection
    - [] Choosing a public dataset of clean and noisy speech data

## Preprocessing
    - [] Steps to preprocess the data for training (e.g. normalization)
    - [] Decide on the length of total speech file (3 seconds maybe)
    - [] Split data into training, validation, and test sets

## STFT Feature Extraction
    - [] Framing scheme (e.g. window size, window type, window shift) 
        - Stride = 2ms
        - Window size = 32ms [512 samples]

## Model Architecture and training
    - [] The input layer :
        - Dimensions equivalent to the STFT features for 10 STFT frames

    - [] The output layer:
        - Dimensions equivalent to the STFT features for 1 STFT frame

    - [] Loss
        - Choose loss function to compare clean and enhanced speech
        - Somehow store this 512 sample vector and then keep appending it
        - Finally after the speech is over, we can use the loss function to compare the clean and enhanced speech
        - [] Here decide if you want to compare with FFTs or IFFTs of clean/enhanced speech

## Realtime 
    - [] Buffer starts with zeros and then we append the first 2ms
    - [] Take FFT of the buffer and put it into the model input 10 FRAME buffer
    - [] keep repeating this
