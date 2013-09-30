#! /usr/bin/env python

import os

import cv
import pyaudio
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from skimage.feature import match_template


def detect_surf(image, equalize=False):    
    if equalize: cv.EqualizeHist(image, image)
    keypoints, descriptors = cv.ExtractSURF(image, None, cv.CreateMemStorage(),
                                            (0, 800, 4, 5))
    for keypoint in keypoints:
        x, y = int(keypoint[0][0]), int(keypoint[0][1])
        cv.Circle(image, (x, y), 1, cv.RGB(0, 0, 255), 3, 8, 0)
    return image


def detect_gftt(image, equalize=False, cornerCount=500, qualityLevel=0.005,
                minDistance=30):    
    if equalize: cv.EqualizeHist(image, image)
    eigImage = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1)
    tempImage = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 1)
    cornerMem = cv.GoodFeaturesToTrack(image, eigImage, tempImage,
                                       cornerCount, qualityLevel,
                                       minDistance, None, 2,
                                       False)
    for point in cornerMem:
        x, y = int(point[0]), int(point[1])
        cv.Circle(image, (x, y), 1, cv.RGB(0, 255, 0), 3, 8, 0)
    return image
    

def detect_harris(array):
    image = as_ipl(array)
    cornerMap = cv.CreateMat(image.height, image.width, cv.CV_32FC1)
    cv.CornerHarris(image, cornerMap, 3)
    signature = np.zeros((image.height, image.width)).astype('uint8')
    for y in range(0, image.height):
        for x in range(0, image.width):
            harris = cv.Get2D(cornerMap, y, x)
            if harris[0] > 10e-06:
                cv.Circle(image,(x,y),1,cv.RGB(155, 0, 25), 3,8,0)
                signature[y, x] = 255
    corner_array = np.asarray(signature[:, :])
    return corner_array

def as_ipl(source):
    bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]),
                                  cv.IPL_DEPTH_8U, 1)
    cv.SetData(bitmap, source.tostring(), source.dtype.itemsize*source.shape[1])
    return bitmap    
    
def ipl_as_array(ipl):
    return np.asarray(ipl[:, :])

def show_images(layout, images, show=True):
    fig = plt.figure()
    for i, image in enumerate(images):
        ax = fig.add_subplot(layout*10 + (i+1))
        ax.imshow(image)
    if show: plt.show()
    
################################################################################
    
def spectrogram_from_file(f, channel=0, sample_size=512):
    sampling_rate, audio_data = wavfile.read(f)
    track = np.sum(audio_data, 1) if channel == 'sum' else audio_data[:, channel]
    track = track.astype('float32')
    downsampled = track.reshape((-1, 4)).T.mean(0)
    normalized = downsampled / downsampled.max()
    segmented = normalized.reshape((-1, sample_size))
    spectrum = abs(np.fft.fft(segmented))[:, :sample_size / 2].T
    return spectrum, sampling_rate
    
def spectrogram(data, segment_size=60):
    end = len(data) - len(data) % segment_size
    stacked = data[:end].reshape((-1, segment_size))
    freq_space = np.fft.fft(stacked)
    real = np.abs(freq_space)
    # fft results are mirrored, this trims the excess
    trimmed = real.T[:segment_size/2, :]
    return trimmed


def downsample2d(a, factor):
    e0 = a.shape[0] - a.shape[0] % factor
    e1 = a.shape[1] - a.shape[1] % factor
    shape = a.shape[0] / factor, a.shape[1] / factor
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a[:e0, :e1].reshape(sh).mean(-1).mean(1)



def process_wavfile(filename, store):
    """ Open the given wavfile, downsample it, compute the
    spectrogram, and downsample again. Store the result in
    the given `store` keyed under the filename.
    """
    name = filename.split('/')[-1].split('.')[0]
    sampling_rate, audio = wavfile.read(filename)
    downsampled = audio.reshape((-1, 16)).mean(1)
    spec = spectrogram(downsampled, segment_size=512)
    down_spec = downsample2d(spec, 2)
    store[name] = down_spec

def acquire_audio(seconds=5):
    """ Acquire audio for the given duration.
    """
    rate = 11025
    chunk = 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []
    for _ in range(int(rate / chunk * seconds)):
        frames.append(stream.read(chunk))
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    ary = np.fromstring(b''.join(frames), dtype=np.short)
    ary = ary.reshape((-1, 2)).mean(1)
    return ary

def process_acquired(ary):
    """ Calculate the spectrogram and downsample the
    given audio array.
    """
    spec = spectrogram(ary, segment_size=512)
    down_spec = downsample2d(spec, 2)
    return down_spec
    


spectrum, rate = spectrogram_from_file('adv_time/ep1.wav')
template  = spectrum[:, 5000:15000]


result = match_template(spectrum, template)
score = result.max()
print score