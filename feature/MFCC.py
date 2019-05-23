'''This script extracts mel-frequency cepstrum coeficient.'''
import numpy as np 
from scipy.io.wavfile import read as wav_read 
from scipy.fftpack import fft,dct
import math

# def trianglefilter(mel_fre_bins,mel_filterbanks)
class mfcc():
    def __init__(self,wavfile,fs,windowlen,slidelen,fft_n=512,mel_n=25,p=13):
        if type(wavfile) == str:
            self.fs,self.signal = wav_read(wavfile)
        else:
            self.fs = fs
            self.signal = np.asarray(wavfile)
        self.windowlen = windowlen
        self.slidelen = slidelen
        self.fft_n = fft_n
        self.mel_n = 25
        self.p = p
    
    def preemphasis(self,a=0.97):
        signal = np.roll(self.signal,1)
        signal = signal-a*signal
        return signal
    
    def frameWindow(self,signal):
        '''using hamming window function'''
        ##the sliding window is half of windowlen.
        N = math.ceil((len(signal)-self.windowlen)/self.slidelen+1)
        L = int((N-1)*self.slidelen+self.windowlen)
        signal = np.pad(signal,(0,L-len(signal)),'constant',constant_values = (0,0))
        frames = np.zeros((N,self.windowlen))
        for i in range(N):
            frames[i,:] = signal[i*self.slidelen:i*self.slidelen+self.windowlen]

        wn = 0.54-0.46*np.cos(2*np.pi*np.arange(self.windowlen)/(self.windowlen-1))
        frames = frames*wn
        return frames
    
    def fft(self,frames):
        spectrum = fft(frames,n=self.fft_n)
        spectrum = spectrum[:,:(self.fft_n//2+1)]
        return spectrum
        
    def melfliterbanks(self,spectrum):
        lowbound = 100
        upbound = self.fs//2
        mel_fre_bins = 2595*np.log10((1+np.arange(1,spectrum.shape[1]+1)/self.fft_n*self.fs/700))
        lowmel = 2595*np.log10(1+lowbound/700)
        upmel = 2595*np.log10(1+upbound/700)
        mel_filterbanks = lowmel+np.arange(self.mel_n+2)*(upmel-lowmel)/(self.mel_n+1)
        #mel_filterbonks round to nearest frequency bins's mel
        filterbanks = 700*(np.power(10,mel_filterbanks/2595)-1)
        index = [math.floor(x/(self.fs/(self.fft_n+1))) for x in filterbanks]
        mel_filterbanks = mel_fre_bins[index]
        ####the triangle filter function
        Hik = np.zeros((self.mel_n,len(mel_fre_bins)))
        for i in range(self.mel_n):
            ### i is the index of mel filterbank,the center melfrequency of i filterbank is round to mel_fre_bins[index[i+1]],and only index[i]~index[i+2]
            ### of mel_fre_bins are not zero.find them and caculate.
            ### mel_fre_bins[s～e] pass through this i filterbank and not zero.
            s = index[i+1-1]
            c = index[i+1]
            e = index[i+2]
            for j in range(s,c):
                Hik[i,j] = (mel_fre_bins[j]-mel_filterbanks[i])/(mel_filterbanks[i+1]-mel_filterbanks[i])
            for j in range(c,e+1):
                Hik[i,j] = (mel_filterbanks[i+2]-mel_fre_bins[j])/(mel_filterbanks[i+2]-mel_filterbanks[i+1])
        logen = np.log10(np.sum(np.expand_dims(np.abs(spectrum),axis=1)*Hik,axis=-1))
        print(logen.shape)
        return logen

    def dct(self,logen):
        dcc = dct(logen)
        coef = dcc[:,:self.p]
        return coef
    
    def extract(self):
        signal = self.preemphasis()
        frames = self.frameWindow(signal)
        spectrum = self.fft(frames)
        logen = self.melfliterbanks(spectrum)
        coef = self.dct(logen)
        return coef
    
if __name__=='__main__':
    wavfile = '/home/lemn/material/dataSet/TIMIT/train/dr1/1/sa1.wav'
    fs = 16000
    windowlen = 400
    m = mfcc(wavfile,fs,windowlen,160)
    fea = m.extract()
    print(fea)
