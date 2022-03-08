#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:37:40 2022

@author: sophiegibbins
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


#%%
#paths to data files

dark = '/Users/sophiegibbins/Uni/Year 4/Project/_LASSENA_DATA/20220303'
dark_path_end = input('Enter the dark data file name:')
dark_path = os.path.join(dark, dark_path_end )

data = '/Users/sophiegibbins/Uni/Year 4/Project/_LASSENA_DATA/20220303'
data_path_end = input('Enter the data file name:')
data_path = os.path.join(data, data_path_end)

dirdark = os.listdir(dark_path)
dirdata = os.listdir(data_path)

#%% 
#sort frames by frame number

def sort(lst): 
    convert = lambda x: int(x) if x.isdigit() else x.lower()
    key1 = lambda key: [convert(i) for i in re.split('([0-9]+)', key)]
    return sorted(lst, key=key1)

sorted_dark_files = sort(dirdark)

sorted_data_files = sort(dirdata)

#%%
#open dark frames
dark_stack = []
for i in sorted_dark_files:
    if i[-5:] == '.tiff':
        im = Image.open(os.path.join(dark_path, i))
        imarray = np.array(im)
        imarray = np.float32(imarray)
        dark_stack.append(imarray)
        
#Open light frames
data_stack = []
for i in sorted_data_files:
    if i[-5:] == '.tiff':
        im = Image.open(os.path.join(data_path, i))
        imarray1 = np.array(im)
        imarray1 = np.float32(imarray1)
        data_stack.append(imarray1)

#%%
#mean value for each dark frame 
dark_mean = np.mean(dark_stack, axis = 0)
#mean of all dark frames
mean_dark_mean = np.mean(dark_mean)

#%%
#dark correction 
def dark_correction(lst):
    dark_corrected = []
    for i in range(len(lst)):
        corrected = lst[i] - dark_mean
        dark_corrected.append(corrected)
    return dark_corrected

#mean dark correction
def mean_dark_correction(lst):
    mean_dark_corrected = []
    for i in range(len(lst)):
        mean = np.mean(lst[i])
        mean_dark_corrected.append(mean)
    return mean_dark_corrected

data_corrected = dark_correction(data_stack)

#%%
#mean array from all images
data_mean = np.mean(data_corrected, axis=0)
plt.imshow(data_mean)
plt.show()

#%%
#plot pixel value versus position along one column 

col = int(input('Enter the column in the middle of the collimator:'))
plt.plot(np.arange(0, len(data_mean)), data_mean[:, col])
plt.xlabel('Position')
plt.ylabel('Pixel value')
plt.title('Pixel value vs position')
plt.show()

#%%
#Choose column range to take mean value over

col_min = int(input('Enter first column to take average over:'))
col_max = int(input('Enter last column to take average over:'))

def col_sum(arr, col_min, col_max):
    if col_max-col_min>1: 
        s = 0
        for i in range(col_min, col_max):
            s = s + arr[:, i]
        mean = s/(col_max-col_min)
        return mean

col_mean = col_sum(data_mean, col_min, col_max)

#%%
def ROI_select(arr, row_min, row_max, col_mean):
    x = np.arange(row_min, row_max)
    y = col_mean[row_min:row_max]
    return x, y

row_min = int(input('Enter the first row for the ROI:'))
row_max = int(input('Enter the last row for the ROI:'))
x_roi, y_roi = ROI_select(data_mean, row_min, row_max, col_mean)
plt.plot(x_roi, y_roi)
plt.show()

#%%

def Gauss(x, H, A, x0, sigma):
    return H + abs(A)*np.exp(-1/2*((x-x0)/sigma)**2)

#make a guess for the parameters: peak value, peak position, standard deviation
def guess(x, y):
    mean = (min(x)+max(x))/2
    sigma = (max(x)-mean)/2
    return [min(y), max(y), mean, sigma]

#%%
#Find peaks and valleys for ROI 

def peak_valley_finder(y):
    peaks = find_peaks(y)
    valleys = find_peaks(-y)
    return peaks, valleys

#arrays of indices of peaks and valleys
peak_idx = peak_valley_finder(y_roi)[0][0]
valley_idx = peak_valley_finder(y_roi)[1][0]

def check_indices(pidx, vidx, ydata, peak_thresh, valley_thresh): #set threshold to remove peaks/valleys indices below/above this
    peak = []
    valley = []
    for i in range(0, len(pidx)-1):
        y_peak = ydata[pidx[i]]
        if y_peak >= peak_thresh:
            peak.append(pidx[i])
    
    for j in range(0, len(vidx)-1):
        y_valley = ydata[vidx[j]]
        if y_valley <= peak_thresh and y_valley > valley_thresh:
            valley.append(vidx[j])
            
    return peak, valley

peak_thresh = int(input('Enter threshold value above which to include peaks:'))
valley_thresh = int(input('Enter threshold value above which to include valleys:'))
new_pidx = check_indices(peak_idx, valley_idx, y_roi, peak_thresh, valley_thresh)[0]
new_vidx = check_indices(peak_idx, valley_idx, y_roi, peak_thresh, valley_thresh)[1]

def plot_indices(xdata, ydata, idx):
    x_values = []
    y_values = []
    for i in range(len(idx)):
        x = xdata[idx[i]]
        y = ydata[idx[i]]
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values

valley_x = plot_indices(x_roi, y_roi, new_vidx)[0]
valley_y = plot_indices(x_roi, y_roi, new_vidx)[1]

peak_x = plot_indices(x_roi, y_roi, new_pidx)[0]
peak_y = plot_indices(x_roi, y_roi, new_pidx)[1]


plt.plot(valley_x, valley_y, 'o')
plt.plot(peak_x, peak_y, 'o')
plt.show()

#%%
#find range for each peak and valley for Gaussian fit
def gauss_cutoff(pidx, vidx, xdata):
    peak_range = []
    valley_range = []
    for i in range(1, len(vidx)):
        peak_start = int((pidx[i]+vidx[i-1])/2)
        peak_stop = int((pidx[i]+vidx[i])/2)
        p = xdata[peak_start:peak_stop+1]
        peak_range.append(p)           
        valley_start = int((pidx[i-1]+vidx[i-1])/2)
        valley_stop = int((pidx[i]+vidx[i-1])/2)
        v = xdata[valley_start:valley_stop+1]
        valley_range.append(v)
    return peak_range, valley_range

peak_ranges = gauss_cutoff(new_pidx, new_vidx, x_roi)[0]
valley_ranges = gauss_cutoff(new_pidx, new_vidx, x_roi)[1] 

x = valley_ranges[1] 
y = col_mean[min(x):max(x)+1]
plt.plot(x, y, 'o')
#%%

def peak_points(peaks): #finds data values corresponding to indices
    x_peak = []
    y_peak = []
    for i in range(len(peaks)):
        if len(peaks[i])>=4:
            x = peaks[i]
            y = col_mean[min(x):max(x)+1]
            x_peak.append(x)
            y_peak.append(y)
    return x_peak, y_peak
      
def valley_points(valleys):
    x_valley = []
    y_valley = []
    for i in range(len(valleys)):
        if len(valleys[i])>=4:
            x = valleys[i]
            y = col_mean[min(x):max(x)+1]
            x_valley.append(x)
            y_valley.append(y)
    return x_valley, y_valley

#%%

def peak_fit(peaks, valleys):
    x = peaks[0]
    y = peaks[1]
    x2 = valleys[0]
    y2 = valleys[1]
    opt = []
    valley = []
    for i in range(len(x)):
        popt, pcov = curve_fit(Gauss, x[i], y[i], p0 = guess(x[i], y[i]))
        x_range = np.linspace(min(x[i]), max(x[i]))
        peak_fit = plt.plot(x_range, Gauss(x_range,*popt), label = 'Gaussian fit')
        opt.append(popt[0]+popt[1])
    for j in range(len(x2)):
        popt2, pcov2 = curve_fit(Gauss, x2[j], 1/y2[j], p0 = guess(x2[j], y2[j]))
        x_range2 = np.linspace(min(x2[j]), max(x2[j]))
        fit = Gauss(x_range2, *popt2)
        valley_fit = plt.plot(x_range2, 1/fit, label = 'Gaussian fit')
        v = min(1/fit)
        valley.append(v)
    return peak_fit, valley_fit, opt, valley

peaks = peak_points(peak_ranges)
valleys = valley_points(valley_ranges)

plt.plot(x_roi, y_roi,)

peak_values = peak_fit(peaks, valleys)[2]
valley_values = peak_fit(peaks, valleys)[3]

pvdr = np.mean(peak_values)/np.mean(valley_values)
plt.xlabel('Position')
plt.ylabel('Pixel value')
print('The pvdr is', pvdr)
plt.show()

#%%

