import sys
import time
import os
import glob
import numpy as np
import pickle
import math
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import matplotlib.pyplot as plt
from scipy import linalg as la
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import utilities

eps = 0.00000001

""" Función para inicializar los bancos de filtros de MFCC """
def mfccInitFilterBanks(fs, nfft):
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    nFiltTotal = numLinFiltTotal + numLogFilt
    freqs = np.zeros(nFiltTotal + 2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal - 1] * logsc ** np.arange(1, numLogFilt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])

    fbank = np.zeros((nFiltTotal, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i + 1]
        highTrFreq = freqs[i + 2]

        lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1, np.floor(cenTrFreq * nfft / fs) + 1, dtype=np.int32)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1, np.floor(highTrFreq * nfft / fs) + 1, dtype=np.int32)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs

""" Cálculo de MFCC para un frame """
def stMFCC(X, fbank, nceps):
    mspec = np.log10(np.dot(X, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps

""" Extracción de características """
def stFeatureExtraction(signal, Fs, Win, Step):
    Win = int(Win)
    Step = int(Step)

    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = np.abs(signal).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)
    curPos = 0
    countFrames = 0
    nFFT = Win // 2

    [fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)
    nceps = 13
    totalNumOfFeatures = nceps

    stFeatures = []
    while curPos + Win - 1 < N:
        countFrames += 1
        x = signal[curPos:curPos + Win]
        curPos = curPos + Step
        X = abs(fft(x))
        X = X[:nFFT]
        X = X / len(X)
        curFV = np.zeros((totalNumOfFeatures, 1))
        curFV[:nceps, 0] = stMFCC(X, fbank, nceps).copy()
        stFeatures.append(curFV)

    stFeatures = np.concatenate(stFeatures, 1)
    return stFeatures.T
