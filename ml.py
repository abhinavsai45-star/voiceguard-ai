"""Shared log-mel feature extraction and CNN inference helpers."""
from pathlib import Path
import io
import math
import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly
import torch
from torch import nn

SAMPLE_RATE=16000
N_MELS=64
FRAMES=300

class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d((8,8)))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(64*8*8,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x): return self.classifier(self.features(x))

def features_from_bytes(raw,filename):
    # Use libsndfile first (WAV, FLAC, OGG and MP3 are supported by current wheels).
    # Keeping this decode path explicit also makes training reproducible.
    try:
        y,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
        y=y.mean(axis=1)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError('This audio codec is not available here. Convert it to WAV, MP3, FLAC, or OGG and try again.') from exc
    y=y[:int(sr*30)]
    if sr!=SAMPLE_RATE:
        divisor=math.gcd(int(sr),SAMPLE_RATE)
        y=resample_poly(y,SAMPLE_RATE//divisor,int(sr)//divisor).astype(np.float32)
    if y.size < SAMPLE_RATE//2: raise ValueError('Audio must contain at least 0.5 seconds of sound.')
    # The CNN consumes log-mel energy features; pre-emphasis reduces DC rumble.
    y=lfilter([1,-.97],[1],y).astype(np.float32)
    n_fft,hop=512,160
    padded=np.pad(y,(n_fft//2,n_fft//2))
    frames=np.lib.stride_tricks.sliding_window_view(padded,n_fft)[::hop]
    spectrum=np.fft.rfft(frames*np.hanning(n_fft).astype(np.float32),n=n_fft,axis=1)
    power=(spectrum.real**2+spectrum.imag**2).astype(np.float32).T
    # Triangular mel filter bank, built directly to avoid decoder/JIT startup cost.
    hz_to_mel=lambda hz: 2595*np.log10(1+hz/700)
    mel_to_hz=lambda mel_value: 700*(10**(mel_value/2595)-1)
    mel_points=mel_to_hz(np.linspace(hz_to_mel(40),hz_to_mel(7600),N_MELS+2))
    fft_freqs=np.linspace(0,SAMPLE_RATE/2,n_fft//2+1)
    basis=np.zeros((N_MELS,len(fft_freqs)),dtype=np.float32)
    for m in range(N_MELS):
        left,center,right=mel_points[m:m+3]
        basis[m]=np.maximum(0,np.minimum((fft_freqs-left)/(center-left),(right-fft_freqs)/(right-center)))
    mel=basis @ power
    logmel=10*np.log10(np.maximum(mel,1e-10))
    logmel-=np.max(logmel)
    logmel=np.clip(logmel,-80,0)/40.0
    if logmel.shape[1]<FRAMES: logmel=np.pad(logmel,((0,0),(0,FRAMES-logmel.shape[1])))
    else: logmel=logmel[:,:FRAMES]
    return torch.tensor(logmel,dtype=torch.float32).unsqueeze(0).unsqueeze(0),len(y)/SAMPLE_RATE

def load_model(path):
    model=AudioCNN()
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    return model

def predict_bytes(raw,filename,model_path):
    x,duration=features_from_bytes(raw,filename)
    model=load_model(model_path)
    with torch.no_grad(): probs=torch.softmax(model(x),dim=1)[0].numpy()
    i=int(np.argmax(probs))
    return ('REAL VOICE' if i==0 else 'AI-GENERATED VOICE',float(probs[i]),duration)
