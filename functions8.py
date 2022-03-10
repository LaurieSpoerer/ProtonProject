# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:39:21 2021

@author: lauri
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import ast
import scipy.signal
import copy
from tqdm import tqdm


samcalpath = 'C:/Users/lauri/OneDrive/Uni stuff/Year 4/Proton Detection Project/Code/Preliminary Code/Sam_Data/' #path to the cal data
cal_img_dir = samcalpath+'Images/' #calibration images
cal_dark_dir = samcalpath+'Dark/Dark_Baseline_28ms/' #dark images from calibration
cal_dark_dir_multi = samcalpath+'Dark/Dark multi/'

munpath = 'C:/Users/lauri/OneDrive/Uni stuff/Year 4/Proton Detection Project/Code/Preliminary Code/Munich_MicrobeamShare/' #path to the data
mun_img_dir = munpath+'MunDay2_Steps_100um/user_test/' #path to the images
mun_dark_dir = munpath+'MunDay2Dark4/user_test/' #path to the dark images

lappath = 'C:/Users/lauri/OneDrive/Uni stuff/Year 4/Proton Detection Project/Code/Real Deal/Data/'

extpath = 'E:/_LASSENA_DATA/'

#pathq = False
#while not pathq:
#    direction = input("Are you using Sam's preliminary data (1), data from Laurie's Laptop (2), or data from the SSD (3)?: ")
#    
#    if direction == '1':
#        pathq=True
#    if direction == '2':
#        for i in np.arange(len(os.listdir(lappath))):
#            print(i,":",os.listdir(lappath)[i])
#        lappath_date = lappath+os.listdir(lappath)[int(input("Select the number of the folder desired: "))]+'/'
#        
#        for i in np.arange(len(os.listdir(lappath_date))):
#            print(i,":",os.listdir(lappath_date)[i])
#        img_dir = lappath_date+os.listdir(lappath_date)[int(input("Select the desired image folder name: "))]+'/'
#        dark_dir = lappath_date+os.listdir(lappath_date)[int(input("Select the desired dark folder name: "))]+'/'
#        
#        pathq=True
#        
#    if direction == '3':
#        for i in np.arange(len(os.listdir(extpath))):
#            print(i,":",os.listdir(extpath)[i])
#        extpath_date = extpath+os.listdir(extpath)[int(input("Select the desired folder number: "))]+'/'
#        
#        for i in np.arange(len(os.listdir(extpath_date))):
#            print(i,":",os.listdir(extpath_date)[i])
#        img_dir = extpath_date+os.listdir(extpath_date)[int(input("Select the desired image folder number: "))]+'/'
#        dark_dir = extpath_date+os.listdir(extpath_date)[int(input("Select the desired dark folder number: "))]+'/'
#        
#        pathq=True
#        
#    else:
#        print("Please enter 1, 2 or 3")
        

    





#------------------------------------------------------------------------------------------------------

def sort(list):
    convert = lambda x: int(x) if x.isdigit() else x.lower()
    key1 = lambda key: [convert(i) for i in re.split('([0-9]+)', key)]
    return sorted(list, key=key1)

def get_tiff_time_stamp(im):
    #temp=getImageDescription(im)
    #for x in dump:
    #    print(x,dump[x])
    return float(getImageDescription(im)["tstamp"])*getClockScalingFactor()

def getClockScalingFactor():
    #ImageDescription: {"udp header": "b'\\xaa\\xff'", "row num": "0", "seq num": "44825", "tstamp": "15750083",
     #                  "frame num" : "10", "integration time": "1750000", "pcb_t": "42.125", "fpga_t": "47.625",
      #                 "ib1"       : "0.5", "1b2": "0.5", "ib3": "4.375", "1b4": "0.5", "ib5": "0.5", "1bc": "10.0",
       #                "vb"        : "1.2000000476837158", "cn": "1.399999976158142", "cp": "1.0",
        #               "vp"        : "3.2200000286102295", "vr": "4.5", "shape": [1248, 1204]}
    #int time was 28ms
    return (28.3/1000)/(1750000) #clock scaling factor in seconds
def getImageDescription(im):
    tag = 270 # this is the key for "ImageDescription"
    ID = im.tag[tag] # get the tag corresponding to "ImageDescription"
    #print ID # this will print a unicode dict which is unworkable
    info = ast.literal_eval(ID[0]) # evaluate from unicode to a workable dict
    #print(info)
    return info

############################################################

def getarray(tif_files): #Gets the arrays from an image set folder
    name_array = []
    time_array = []
    for i in (sort(os.listdir(tif_files))):
        if i[-5:] == '.tiff':
            image = (Image.open(tif_files+i))
            try:
                local_time_stamp = get_tiff_time_stamp(image)
            except:
                local_time_stamp = 0
            name_array.append((np.array(image)).astype(np.int32))
            time_array.append(local_time_stamp)
    return name_array, time_array
    

def readframes(tif_files): #Reads the image set into python, no arrays
    name_imgs = []
    for i in tqdm(sort(os.listdir(tif_files))):
        if i[-5:] == '.tiff':
            name_imgs.append(Image.open(tif_files+i))
    return name_imgs

def get_mean_dark_array(dark_dir):
    dark_mean = np.mean(getarray(dark_dir)[0],axis=0)
    return dark_mean

def get_dark_sub(image_arrays, dark_mean): #Finds an array set - dark mean
    dark_sub_array = []
    for i in (image_arrays):
        dark_sub_array.append(i-dark_mean)
    return dark_sub_array

def get_mean_dark_sub_array(img_dir, dark_dir):
    imgs = getarray(img_dir)[0]
    darks = getarray(dark_dir)[0]
    dark_mean = np.mean(darks, axis=0)
    del darks
    mean_dark_sub = np.mean(get_dark_sub(imgs,dark_mean),axis=0)
    del imgs
    del dark_mean
    
    return mean_dark_sub

def get_means_and_stds(arrays): #Finds means and stds of an array set
    array_means = []
    for i in (arrays):
        array_means.append(np.mean(i))
    array_stds = []
    for x in tqdm(arrays):
        array_stds.append(np.std(x))
    return array_means, array_stds

def get_mean_array(arrays): #finds mean array of array set
    x = np.mean(arrays, axis=0)
    return x

def get_array_mean(array): #finds mean value of single array
    x = np.mean(array)
    return x

def get_array_error(array): #finds the error on a single array
    error = (np.std(array))/(np.sqrt((np.shape(array)[0]*np.shape(array)[1])))
    return error

def get_data_mean(arrays): #finds mean value of array set
    x = get_array_mean(get_mean_array(arrays))
    return x

#----------------------------------------------------------------------------------------------------------

def get_calibration_data(img_dir, dark_dir):  #finds the mean of dark subbed values of an entire image set
    dark_mean = np.mean(getarray(dark_dir)[0], axis = 0)
    mean_img_mean = get_data_mean(get_dark_sub(getarray(img_dir)[0], dark_mean))
    del dark_mean
    return mean_img_mean

def get_calibration_data_ROI(img_dir, dark_dir, y1,y2, x1,x2): # ^ but in a chosen ROI, didn't finish as multi is better
    dark_mean = np.mean(getarray(dark_dir)[0], axis = 0)
    mean_img_mean = get_data_mean(get_dark_sub(getarray(img_dir)[0], dark_mean))
    del dark_mean
    return mean_img_mean

def get_calibration_data_multiple(datas, dark_data): #Finds the mean dark subbed value and error of multiple image sets seperately
    mean_datas_values = []
    mean_datas_errors = []
    dark_mean = np.mean(getarray(dark_data)[0], axis = 0)
    for x in sort(os.listdir(datas)):
        mean_datas_values.append(get_data_mean(get_mean_array(get_dark_sub((getarray(datas+x+'/'))[0], dark_mean))))
        mean_datas_errors.append(get_array_error(get_mean_array(get_dark_sub((getarray(datas+x+'/'))[0], dark_mean))))
    del dark_mean
    return mean_datas_values, mean_datas_errors

def get_calibration_data_multiple_ROI(datas, dark_data, y1,y2, x1,x2): # ^ but in a chosen ROI
    mean_datas_values = []
    mean_datas_errors = []
    dark_mean = np.mean(getarray(dark_data)[0], axis = 0)
    for x in tqdm(sort(os.listdir(datas))):
        mean_array = (get_mean_array(get_dark_sub(getarray(datas+x+'/')[0], dark_mean))[y1:y2, x1:x2])
        mean_datas_values.append(get_array_mean(mean_array))
        mean_datas_errors.append(get_array_error(mean_array))
    del dark_mean    
    return mean_datas_values, mean_datas_errors

def get_calibration_data_multiple_2_ROI(datas, dark_datas, y1,y2, x1,x2): # ^ but in a chosen ROI
    mean_datas_values = []
    mean_datas_errors = []
    for x in tqdm(np.arange(len(sort(os.listdir(datas))))):
        dark_mean = np.mean(getarray(cal_dark_dir_multi+os.listdir(dark_datas)[x]+'/')[0], axis = 0)
        mean_array = (get_mean_array(get_dark_sub(getarray(datas+os.listdir(datas)[x]+'/')[0], dark_mean))[y1:y2, x1:x2])
        del dark_mean
        mean_datas_values.append(get_array_mean(mean_array))
        mean_datas_errors.append(get_array_error(mean_array))

    return mean_datas_values, mean_datas_errors

def get_calibration_data_multiple_array(data_arrays, dark_arrays): #Finds the mean dark subbed value and error of multiple image sets seperately
    mean_datas_values = []
    mean_datas_errors = []
    dark_mean = np.mean(dark_arrays[0], axis = 0)
    for x in (data_arrays):
        mean_datas_values.append(get_data_mean(get_mean_array(get_dark_sub(x, dark_mean))))
        mean_datas_errors.append(get_array_error(get_mean_array(get_dark_sub(x, dark_mean))))
    del dark_mean
    return mean_datas_values, mean_datas_errors

def get_calibration_data_multiple_array_ROI(data_arrays, dark_arrays, y1,y2, x1,x2): #Finds the mean dark subbed value and error of multiple image sets seperately
    mean_datas_values = []
    mean_datas_errors = []
    dark_mean = np.mean(dark_arrays[0], axis = 0)
    for x in (data_arrays):
        mean_datas_values.append(get_data_mean(get_mean_array(get_dark_sub(x, dark_mean))[y1:y2, x1:x2]))
        mean_datas_errors.append(get_array_error(get_mean_array(get_dark_sub(x, dark_mean))[y1:y2, x1:x2]))
    del dark_mean
    return mean_datas_values, mean_datas_errors

def get_dose_values(currents, kdose, kcurrent): #Finds the dose values of signals from a known dose at a known current
    prop_const = kdose/kcurrent
    dose_values = [i*prop_const for i in currents]
    
    return dose_values

#----------------------------------------------------------------------------------------------------------

def getlobf(x, y): #finds line of best fit of a curve

# calculate polynomial
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)

# calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    return x_new, y_new, f

def gaussian(x, amp, mean, std, c):
    return (amp * np.exp(-((x - mean) / 4 / std)**2) + c)

def guess(x, y):
    mean = (min(x)+max(x))/2
    sigma = (max(x)-mean)/2
    return [min(y), max(y), mean, sigma]

#----------------------------------------------------------------------------------------------------------

def getarray_ROI(tif_files, x1,x2, y1,y2): #Finds array set of image set in a chosen ROI
    name_array = []
    time_array = []
    for i in sort(os.listdir(tif_files)):
        if i[-5:] == '.tiff':
            image = (Image.open(tif_files+i))
            try:
                local_time_stamp = get_tiff_time_stamp(image)
            except:
                local_time_stamp = 0
            name_array.append(((np.array(image)).astype(np.int32))[y1:y2, x1:x2])
            time_array.append(local_time_stamp)
    return name_array, time_array

#----------------------------------------------------------------------------------------------------------

def get_array_ROI(array, x1,x2, y1,y2): #Finds an ROI array for a given array
    array_ROI = array[y1:y2, x1:x2]
    return array_ROI

def get_arrays_ROI(arrays, x1,x2, y1,y2): #Finds an ROI array set for a given array set
    arrays_ROI = []
    for i in arrays:
        arrays_ROI.append(i[y1:y2, x1:x2])
    return arrays_ROI

#----------------------------------------------------------------------------------------------------------

def col_select(array, colmiddle, plusminus): #finds specific range of columns in an array, and transposes
    cols=np.transpose(array[:,colmiddle-plusminus:colmiddle+plusminus+1])
    return cols
    
def row_select(array, rowmiddle, plusminus): #finds specific rows in an array
    rows=array[rowmiddle-plusminus:rowmiddle+plusminus+1,:]
    return rows

def get_mean_row(rows):
    x = np.mean(rows, axis=0)
    return x

def get_row_ROI(row):
    qq1=False
    while not qq1:
        threshold = int(input("Please enter a threshold: "))
        for i in np.arange(len(row)):
            try:
                if row[i+1] - row[i] > threshold:
                    x1 = i
                    break
                
            
                else:
                    continue
            except IndexError:
                continue
            
            
        for i in np.arange(len(row)):
            try:
                if row[-i-1] - row[-i] > threshold:
                    x2 = len(row) -i
                    break
                else:
                    continue
            except IndexError:
                continue
    
        try:
            row_roi = copy.deepcopy(row[x1:x2])
        except UnboundLocalError:
            print("Please choose a lower threshold")
        else:
            qq1 = True

                
    row_ROI = row[x1:x2+1]
    
    return row_ROI, x1, x2



#----------------------------------------------------------------------------------------------------------

def get_peaks_valleys(row): #finds peaks, valleys, and pdvr
    final_peaks = []
    final_valleys = []
    temp_valleys = []
    peaks = scipy.signal.find_peaks(row)[0]
    valleys = scipy.signal.find_peaks(-row)[0]
    plt.plot(np.arange(len(row)), row)
    for i in tqdm(peaks):
        for j in np.arange(100):
            try:
                if row[i-j] < row[i-j-1]:
                    left_lim_p=j
                    break
                else:
                    continue
            except IndexError:
                continue
            
        for j in np.arange(100):
            try:
                if row[i+j] < row[i+j+1]:
                    right_lim_p=j
                    break
                else:
                    continue
            except IndexError:
                continue
                
        if left_lim_p == 1:
            continue
        if right_lim_p == 1:
            continue
        
        yp = row[i-left_lim_p:i+right_lim_p]
        xp = np.arange(len(row[i-left_lim_p:i+right_lim_p]))
        
        poptp, _ = scipy.optimize.curve_fit(gaussian, xp, yp)
        x_norm_p = np.linspace(0, len(xp), 100)
        #x1 = int((np.argmax(gaussian(x_norm_p, *poptp)) - left_lim_p)/2)
        #x2 = int((np.argmax(gaussian(x_norm_p, *poptp)) - right_lim_p)/2)
        plt.plot(np.linspace(+i-left_lim_p, i+right_lim_p, 100), gaussian(x_norm_p, *poptp), linestyle='dashed')
        final_peaks.append(np.max(gaussian(x_norm_p, *poptp)))
        temp_valleys.append(i+right_lim_p)
    #plt.plot(np.arange(len(row)), row)
    #plt.show()
    
        
    for i in tqdm(valleys):
        for j in np.arange(100):
            try:
                if row[i-j] > row[i-j-1]:
                    left_lim_v=j
                    break
                else:
                    continue
            except IndexError:
                continue
            
        for j in np.arange(100):
            try:
                if row[i+j] > row[i+j+1]:
                    right_lim_v=j
                    break
                else:
                    continue
            except IndexError:
                continue
                
        if left_lim_v == 1:
            continue
        if right_lim_v == 1:
            continue

        yv = row[i-left_lim_v//2:i+right_lim_v//2]
        xv = np.arange(len(row[i-left_lim_v//2:i+right_lim_v//2]))
        
        new_x, new_y, new_func = getlobf(xv,yv)
        x_new = []
        for x in new_x:
            x_new.append(x+i-left_lim_v//2)

        final_valleys.append(np.min(new_y))
        plt.plot(x_new, new_y)
    
    plt.show()

    pvdr = (np.mean(final_peaks))/(np.mean(final_valleys))    
    
    temp_pvdr = np.mean(final_peaks)/np.mean(temp_valleys)
    
    print(f"The PVDR for this row is: {pvdr}")
    print(f"The PVDR could be: {temp_pvdr}")
    return final_peaks, final_valleys


def peak_valley_plotter(row_full, left, right): #finds peaks, valleys, and pdvr
    row = row_full[left:right]
    peaks = scipy.signal.find_peaks(row)[0]
    valleys = scipy.signal.find_peaks(-row)[0]
    plt.plot(np.arange(len(row)),row) 
    for i in tqdm(peaks):
        for j in np.arange(100):
            try:
                if row[i-j] < row[i-j-1]:
                    left_lim_p=j
                    break
                else:
                    continue
            except IndexError:
                continue
            
        for j in np.arange(100):
            try:
                if row[i+j] < row[i+j+1]:
                    right_lim_p=j
                    break
                else:
                    continue
            except IndexError:
                continue
                
        if left_lim_p == 1:
            continue
        if right_lim_p == 1:
            continue
        
        yp = row[i-left_lim_p:i+right_lim_p]
        xp = np.arange(len(row[i-left_lim_p:i+right_lim_p]))
        
        poptp, _ = scipy.optimize.curve_fit(gaussian, xp, yp)
        x_norm_p = np.linspace(0, len(xp), 100)
        
        plt.plot(np.linspace(+i-left_lim_p, i+right_lim_p, 100), gaussian(x_norm_p, *poptp), linestyle='dashed')
        
    for i in tqdm(valleys):
        for j in np.arange(100):
            try:
                if row[i-j] > row[i-j-1]:
                    left_lim_v=j
                    break
                else:
                    continue
            except IndexError:
                continue
            
        for j in np.arange(100):
            try:
                if row[i+j] > row[i+j+1]:
                    right_lim_v=j
                    break
                else:
                    continue
            except IndexError:
                continue
                
        if left_lim_v == 1:
            continue
        if right_lim_v == 1:
            continue


        yv = row[i-left_lim_v//2:i+right_lim_v//2]
        xv = np.arange(len(row[i-left_lim_v//2:i+right_lim_v//2]))
        
        new_x, new_y, new_func = getlobf(xv,yv)
        x_new = []
        for x in new_x:
            x_new.append(x+i-left_lim_v//2)
        plt.plot(x_new, new_y)
       
    plt.show()
        

#---------------------------------------------------------------------------------------------------

def experiment_pvdr(data_dir, dark_dir):
    imgs = getarray(data_dir)[0]
    darks = getarray(dark_dir)[0]
    dark_mean = np.mean(darks,axis=0)
    del darks
    img_dark_sub = get_dark_sub(imgs,dark_mean)
    del imgs
    del dark_mean
    mean_img = np.mean(img_dark_sub,axis=0)
    del img_dark_sub
    plt.imshow(mean_img)
    plt.show()
    rc = input("Are the collimations in rows or columns? (r/c): ")
    
    if rc == 'r':
        mean_img = np.transpose(mean_img)
        plt.imshow(mean_img)
        plt.show()

        
    happy2=False
    while not happy2:
        try:
            row_middle = int(input("What row number in the previous image do you want as the middle?: "))
        except ValueError:
            print("Please enter a number, restarting...")
            happy2=False
            continue
        try:
            plus_minus = int(input("What how many rows either side of the middle do you want?: "))
        except ValueError:
            print("Please enter a number, restarting...")
            happy2=False
            continue
        
        overlay = copy.deepcopy(mean_img)
        overlay[row_middle+plus_minus] = 12000
        overlay[row_middle-plus_minus] = 12000
        plt.imshow(overlay)
        plt.show()
        del overlay
        repeat_y_n2 = input("Are you happy with this? (y/n): ")
        if repeat_y_n2 == 'y':
            happy2 = True
        
    happy1 = False
    while not happy1:
        #try:
        #    threshold = float(input("Please type a threshold number for finding the first and last peaks: "))
        #except ValueError:
        #    print("Please enter a number, restarting...")
        #    happy1=False
        #    continue
        
        mean_row = np.mean(mean_img[row_middle-plus_minus:row_middle+plus_minus],axis=0)
        x1, x2 = get_row_ROI(mean_row)[1:]
        del mean_row
        overlay2 = copy.deepcopy(mean_img)
        overlay2[row_middle-plus_minus:row_middle+plus_minus,x1:x2] = 12000
        plt.imshow(overlay2)
        plt.show()
        del overlay2
        repeat_y_n1 = input("Are you happy with this area? (y/n): ")
        if repeat_y_n1 == 'y':
            happy1 = True
    
    
    mean_img_rows = mean_img[row_middle-plus_minus:row_middle+plus_minus,x1:x2]
    del mean_img

    print("This is the image we are using for the PVDR")
    plt.imshow(mean_img_rows)
    plt.show()
    mean_row = np.mean(mean_img_rows,axis=0)
    del mean_img_rows
    print("This is the average distribution along pixels")
    plt.plot(np.arange(len(mean_row)),mean_row)
    plt.show()
    print("This is the plot of the mean row against position (not pixel number on detector)")
    #plt.plot(np.arange(len(mean_row)),mean_row)
    peaks, valleys = get_peaks_valleys(mean_row)
    #print(f"The PVDR from this image is: {PVDR}")
    plt.show()
    
    
    qq = input("Would you like to look closer at the peaks and valleys? (y/n): ")
    if qq=='y':
        blah=False
        while not blah:
            try:
                left = int(input("What pixel do you want to start from?: "))
            except ValueError:
                print("Enter a number, restarting..")
                continue
            if left == 10000:
                blah=True
                continue
            try:
                right = int(input("What pixel do you want to end?: "))
            except ValueError:
                print("Enter a number, restarting..")
                continue
            if right == 10000:
                blah=True
                continue
            try:
                peak_valley_plotter(mean_row,left,right)
            except ValueError:
                print("Didn't work, not sure why, try again")
                continue
            except UnboundLocalError:
                print("Please change limits")
        
        
            
    return peaks, valleys

def experiment_pvdr_known_ROI(data_dir, dark_dir, row_middle=715, plus_minus=50):
    imgs = getarray(data_dir)[0]
    darks = getarray(dark_dir)[0]
    dark_mean = np.mean(darks,axis=0)
    del darks
    img_dark_sub = get_dark_sub(imgs,dark_mean)
    del imgs
    del dark_mean
    mean_img = np.mean(img_dark_sub,axis=0)
    del img_dark_sub 
    mean_img=np.transpose(mean_img)  
        
    happy1 = False
    while not happy1:
        mean_row = np.mean(mean_img[row_middle-plus_minus:row_middle+plus_minus],axis=0)
        x1, x2 = get_row_ROI(mean_row)[1:]
        del mean_row
        overlay2 = copy.deepcopy(mean_img)
        overlay2[row_middle-plus_minus:row_middle+plus_minus,x1:x2] = 12000
        plt.imshow(overlay2)
        plt.show()
        del overlay2
        repeat_y_n1 = input("Are you happy with this area? (y/n): ")
        if repeat_y_n1 == 'y':
            happy1 = True
    
    
    mean_img_rows = mean_img[row_middle-plus_minus:row_middle+plus_minus,x1:x2]
    mean_img_rows_larger = np.mean(mean_img[row_middle - plus_minus:row_middle + plus_minus, x1-50:x2+50],axis=0)
    del mean_img

    print("This is the image we are using for the PVDR")
    plt.imshow(mean_img_rows)
    plt.show()
    mean_row = np.mean(mean_img_rows,axis=0)
    print("This is the average distribution along pixels")
    plt.plot(np.arange(len(mean_img_rows_larger)),mean_img_rows_larger)
    plt.show()
    del mean_img_rows
    print("This is the plot of the mean row against position (not pixel number on detector)")
    #plt.plot(np.arange(len(mean_row)),mean_row)
    peaks, valleys = get_peaks_valleys(mean_row)
    #print(f"The PVDR from this image is: {PVDR}")
    plt.show()
    
    
    qq = input("Would you like to look closer at the peaks and valleys? (y/n): ")
    if qq=='y':
        blah=False
        while not blah:
            try:
                left = int(input("What pixel do you want to start from?: "))
            except ValueError:
                print("Enter a number, restarting..")
                continue
            try:
                right = int(input("What pixel do you want to end?: "))
            except ValueError:
                print("Enter a number, restarting..")
                continue
            try:
                peak_valley_plotter(mean_row,left,right)
            except ValueError:
                print("Didn't work, not sure why, try again")
                continue
            
        
            
    return peaks, valleys

    
def experiment_cal(cal_data_dir,cal_dark_dir):
    darks = getarray(cal_dark_dir)[0]
    dark_mean = np.mean(darks,axis=0)
    del darks
    mean_dark_subs = []
    ini_cal_data = os.listdir(cal_data_dir)[0]
    mean_dark_subs.append(np.mean(get_dark_sub(getarray(cal_data_dir+ini_cal_data+'/')[0], dark_mean),axis=0))
    del dark_mean
    mean_img = np.mean(mean_dark_subs,axis=0)
    ylen = len(mean_img)
    xlen = len(mean_img[0,:])
    yrange = np.arange((ylen))
    xrange = np.arange((xlen))
    print("This is the first dark subtracted image, use to find ROI:")
    plt.imshow(mean_img)
    plt.show()
    #Going to temporaily get user to enter approx coords
    happy = False
    while not happy:
        crop = copy.deepcopy(mean_img)
        try:
            y = int(input("Please enter approx centre y coord of beam: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue

        try:
            y_w = int(input("Enter a value from 0 to the distance from y coord to closest edge: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        try:
            x = int(input("Please enter approx centre x coord of beam: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        try:
            x_w = int(input("Enter a value from 0 to the distance from x coord to closest edge: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        #try:
        #    threshold = int(input("What threshold shall I use to find the beam?: "))
        #except ValueError:
        #    print("Enter a number, restarting...")
        #    continue
        x1=1
        del x1
        xl, xr = get_row_ROI(np.mean(row_select(mean_img, y, y_w),axis=0))[1:]
        x1=1
        del x1
        crop = copy.deepcopy(mean_img)
        yu, yd = get_row_ROI(np.mean(col_select(mean_img, x, x_w),axis=0))[1:]
        plt.imshow(crop[yu:yd,xl:xr], extent = [xrange[xl], xrange[xr], yrange[yd], yrange[yu]])
        plt.show()
        del crop
        x1=1
        del x1
        check = input("Is this good? (y/n): ")
        if check == 'y':
            happy = True
            new_img = mean_img[yu:yd,xl:xr]
            del mean_img
    
    q2 = input("Would you like to crop it any more? (y/n): ")
    if q2 =='y':
        happy2 = False
        while not happy2:
            try:
                yupper = int(input("Please enter how much you want to crop the top: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
                
            try:
                ylower = int(input("Please enter how much you want to crop the bottom: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
            
            try:
                xleft = int(input("Please enter how much you want to crop the left: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
                
            try:
                xright = int(input("Please enter how much you want to crop the right: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
    
            crop2 = copy.deepcopy(new_img)
            plt.imshow(crop2[yupper:-ylower,xleft:-xright], extent = [xrange[xl+xleft],xrange[xr-xright],yrange[yd-ylower],yrange[yu+yupper]])
            plt.show()
            del crop2
            check2 = input("Are you happy with this? (y/n): ")
            if check2 == 'y':
                happy2 = True
    
    else:
        yupper = 0
        ylower = 0
        xleft = 0
        xright = 0
    
    
    print("This set of coordinates will be used to calculate the means of the calibration images:")
    print(f"x: {xl}+{xleft} to {xr}-{xright}")
    print(f"y: {yu}+{yupper} to {yd}-{ylower}")
    
    
    ree = False
    while not ree:
        cur = input("Please enter the list of currents in order of the directory folder: ")
        rents = cur.split(',')
        try:
            currents = [float(i.strip()) for i in rents]
        except ValueError:
            print("Please make sure all the values are numbers, restarting...")
            ree = False
            continue
        
        if len(currents) == len(os.listdir(cal_data_dir)):
            ree = True
            continue
        
        else:
            print("The number of currents entered does not equal the actual number of currents")
            ree = False
    
    means, means_errs = get_calibration_data_multiple_ROI(cal_data_dir,cal_dark_dir, yu+yupper, yd-ylower, xl+xleft, xr-xright)
    plt.errorbar(currents, means, yerr = means_errs, label='Data', fmt='b.')
    plt.plot(getlobf(currents,means)[0],getlobf(currents,means)[1], label='LOBF', linestyle='dashed')
    plt.xlabel('Current (mA)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
    tee = False
    while not tee:
        try:
            kdose = float(input("Please enter the dose at the known current: "))
        except ValueError:
            print("Please make sure all the values are numbers, restarting...")
            tee = False
            continue
        
        try:
            kcurrent = float(input("Please enter the known current: "))
        except ValueError:
            print("Please make sure all the values are numbers, restarting...")
            tee = False
            continue
        
        test = input(f"The known Dose is {kdose} Gy at {kcurrent} mA. Is this correct? (y/n): ")
        if test =='y':
            tee = True

    
    dose_rate = get_dose_values(currents, kdose, kcurrent)
    
    plt.plot(dose_rate, means, 'bo', label = 'Signal vs Dose Rate Data')
    plt.plot(getlobf(dose_rate,means)[0], getlobf(dose_rate,means)[1],linestyle='dashed',label='LOBF')
    plt.xlabel('Dose Rate (Gy/s)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
    return means, means_errs#, dose_conversion

def experiment_cal_known_ROI(cal_data_dir,cal_dark_dir, yu=149, xl=570, yd=362, xr=781, yupper=50,ylower=50,xleft=50,xright=50,currents=[0.5,1,2.5,2.9,2,3.5,3,4,5],kdose=0.0356,kcurrent=13):
    darks = getarray(cal_dark_dir)[0]
    dark_mean = np.mean(darks,axis=0)
    del darks
    mean_dark_subs = []
    ini_cal_data = os.listdir(cal_data_dir)[0]
    mean_dark_subs.append(np.mean(get_dark_sub(getarray(cal_data_dir+ini_cal_data+'/')[0], dark_mean),axis=0))
    del dark_mean
    mean_img = np.mean(mean_dark_subs,axis=0)
    ylen = len(mean_img)
    xlen = len(mean_img[0,:])
    yrange = np.arange((ylen))
    xrange = np.arange((xlen))
    print("This is the first dark subtracted image, use to find ROI:")
    plt.imshow(mean_img)
    plt.show()
    crop = copy.deepcopy(mean_img)
    plt.imshow(crop[yu:yd,xl:xr], extent = [xrange[xl], xrange[xr], yrange[yd], yrange[yu]])
    plt.show()
    del crop
    x1=1
    del x1

    new_img = mean_img[yu:yd,xl:xr]
    del mean_img
    

    
    q2 = input("Would you like to crop it any more? (y/n): ")
    if q2 =='y':
        happy2 = False
        while not happy2:

            crop2 = copy.deepcopy(new_img)
            plt.imshow(crop2[yupper:-ylower,xleft:-xright], extent = [xrange[xl+xleft],xrange[xr-xright],yrange[yd-ylower],yrange[yu+yupper]])
            plt.show()
            del crop2
            check2 = input("Are you happy with this? (y/n): ")
            if check2 == 'y':
                happy2 = True
            if check2== 'n':
                yupper = 0
                ylower = 0
                xleft = 0
                xright = 0
                happy2=True
    
    else:
        yupper = 0
        ylower = 0
        xleft = 0
        xright = 0
    
    
    print("This set of coordinates will be used to calculate the means of the calibration images:")
    print(f"x: {xl}+{xleft} to {xr}-{xright}")
    print(f"y: {yu}+{yupper} to {yd}-{ylower}")
    
    
    means, means_errs = get_calibration_data_multiple_ROI(cal_data_dir,cal_dark_dir, yu+yupper, yd-ylower, xl+xleft, xr-xright)
    plt.errorbar(currents, means, yerr = means_errs, label='Data', fmt='b.')
    plt.plot(getlobf(currents,means)[0],getlobf(currents,means)[1], label='LOBF', linestyle='dashed')
    plt.xlabel('Current (mA)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()

        
    print(f"The known Dose is {kdose} Gy at {kcurrent} mA")


    
    dose_rate = get_dose_values(currents, kdose, kcurrent)
    
    plt.plot(dose_rate, means, 'bo', label = 'Signal vs Dose Rate Data')
    plt.plot(getlobf(dose_rate,means)[0], getlobf(dose_rate,means)[1],linestyle='dashed',label='LOBF')
    plt.xlabel('Dose Rate (Gy/s)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
    return means, means_errs#, dose_conversion

def experiment_cal_multi_dark(cal_data_dir,cal_dark_dir_multi):
    darks = getarray(cal_dark_dir_multi+os.listdir(cal_dark_dir_multi)[0]+'/')[0]
    dark_mean = np.mean(darks,axis=0)
    del darks
    mean_dark_subs = []
    ini_cal_data = os.listdir(cal_data_dir)[0]
    mean_dark_subs.append(np.mean(get_dark_sub(getarray(cal_data_dir+ini_cal_data+'/')[0], dark_mean),axis=0))
    del dark_mean
    mean_img = np.mean(mean_dark_subs,axis=0)
    ylen = len(mean_img)
    xlen = len(mean_img[0,:])
    yrange = np.arange((ylen))
    xrange = np.arange((xlen))
    print("This is the first dark subtracted image, use to find ROI:")
    plt.imshow(mean_img)
    plt.show()
    #Going to temporaily get user to enter approx coords
    happy = False
    while not happy:
        crop = copy.deepcopy(mean_img)
        try:
            y = int(input("Please enter approx centre y coord of beam: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        try:
            y_w = int(input("Enter a value from 0 to the distance from y coord to closest edge: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        try:
            x = int(input("Please enter approx centre x coord of beam: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        try:
            x_w = int(input("Enter a value from 0 to the distance from x coord to closest edge: "))
        except ValueError:
            print("Enter a number, restarting...")
            continue
            
        #try:
        #    threshold = int(input("Last question, what threshold shall I use to find the beam?: "))
        #except ValueError:
        #    print("Enter a number, restarting...")
        #    continue
        x1=1
        del x1
        xl, xr = get_row_ROI(np.mean(row_select(mean_img, y, y_w),axis=0))[1:]
        x1=1
        del x1
        crop = copy.deepcopy(mean_img)
        yu, yd = get_row_ROI(np.mean(col_select(mean_img, x, x_w),axis=0))[1:]
        plt.imshow(crop[yu:yd,xl:xr], extent = [xrange[xl], xrange[xr], yrange[yd], yrange[yu]])
        plt.show()
        del crop
        x1=1
        del x1
        check = input("Is this good? (y/n): ")
        if check == 'y':
            happy = True
            new_img = mean_img[yu:yd,xl:xr]
            del mean_img
    
    q2 = input("Would you like to crop it any more? (y/n): ")
    if q2 =='y':
        happy2 = False
        while not happy2:
            try:
                yupper = int(input("Please enter how much you want to crop the top: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
                
            try:
                ylower = int(input("Please enter how much you want to crop the bottom: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
            
            try:
                xleft = int(input("Please enter how much you want to crop the left: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
                
            try:
                xright = int(input("Please enter how much you want to crop the right: "))
            except ValueError:
                print("Enter a number, restarting...")
                continue
    
            crop2 = copy.deepcopy(new_img)
            plt.imshow(crop2[yupper:-ylower,xleft:-xright], extent = [xrange[xl+xleft],xrange[xr-xright],yrange[yd-ylower],yrange[yu+yupper]])
            plt.show()
            del crop2
            check2 = input("Are you happy with this? (y/n): ")
            if check2 == 'y':
                happy2 = True
    
    else:
        yupper = 0
        ylower = 0
        xleft = 0
        xright = 0
    
    
    print("This set of coordinates will be used to calculate the means of the calibration images:")
    print(f"x: {xl}+{xleft} to {xr}-{xright}")
    print(f"y: {yu}+{yupper} to {yd}-{ylower}")
    
    
    ree = False
    while not ree:
        cur = input("Please enter the list of currents in order of the directory folder: ")
        rents = cur.split(',')
        try:
            currents = [float(i.strip()) for i in rents]
        except ValueError:
            print("Please make sure all the values are numbers, restarting...")
            ree = False
            continue
        
        if len(currents) == len(os.listdir(cal_data_dir)):
            ree = True
            continue
        
        else:
            print("The number of currents entered does not equal the actual number of currents")
            ree = False
    
    means, means_errs = get_calibration_data_multiple_2_ROI(cal_data_dir,cal_dark_dir_multi, yu+yupper, yd-ylower, xl+xleft, xr-xright)
    plt.errorbar(currents, means, yerr = means_errs, label='Data', fmt='b.')
    plt.plot(getlobf(currents,means)[0],getlobf(currents,means)[1], label='LOBF', linestyle='dashed')
    plt.xlabel('Current (mA)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
    tee = False
    while not tee:
        try:
            kdose = float(input("Please enter the dose at the known current: "))
        except ValueError:
            print("Please make sure all the values are numbers, restarting...")
            tee = False
            continue
        
        try:
            kcurrent = float(input("Please enter the known current: "))
        except ValueError:
            print("Please make sure all the values are numbers, restarting...")
            tee = False
            continue
        
        test = input(f"The known Dose is {kdose} Gy at {kcurrent} mA. Is this correct? (y/n): ")
        if test =='y':
            tee = True

    
    dose_rate = get_dose_values(currents, kdose, kcurrent)
    
    plt.plot(dose_rate, means, 'bo', label = 'Signal vs Dose Rate Data')
    plt.plot(getlobf(dose_rate,means)[0], getlobf(dose_rate,means)[1],linestyle='dashed',label='LOBF')
    plt.xlabel('Dose Rate (Gy/s)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
    return means, means_errs

def experiment_cal_multi_dark_known_ROI(cal_data_dir,multi_cal_dark_dir, yu=149, xl=570, yd=362, xr=781, yupper=50,ylower=50,xleft=50,xright=50,currents=[0.5,1,2.5,2.9,2,3.5,3,4,5],kdose=0.0356,kcurrent=13):
    darks = getarray(cal_dark_dir_multi+os.listdir(cal_dark_dir_multi)[0]+'/')[0]
    dark_mean = np.mean(darks,axis=0)
    del darks
    mean_dark_subs = []
    ini_cal_data = os.listdir(cal_data_dir)[0]
    mean_dark_subs.append(np.mean(get_dark_sub(getarray(cal_data_dir+ini_cal_data+'/')[0], dark_mean),axis=0))
    del dark_mean
    mean_img = np.mean(mean_dark_subs,axis=0)
    ylen = len(mean_img)
    xlen = len(mean_img[0,:])
    yrange = np.arange((ylen))
    xrange = np.arange((xlen))
    print("This is the first dark subtracted image, use to find ROI:")
    plt.imshow(mean_img)
    plt.show()
    crop = copy.deepcopy(mean_img)
    plt.imshow(crop[yu:yd,xl:xr], extent = [xrange[xl], xrange[xr], yrange[yd], yrange[yu]])
    plt.show()
    del crop
    x1=1
    del x1

    new_img = mean_img[yu:yd,xl:xr]
    del mean_img
    

    
    q2 = input("Would you like to crop it any more? (y/n): ")
    if q2 =='y':
        happy2 = False
        while not happy2:

            crop2 = copy.deepcopy(new_img)
            plt.imshow(crop2[yupper:-ylower,xleft:-xright], extent = [xrange[xl+xleft],xrange[xr-xright],yrange[yd-ylower],yrange[yu+yupper]])
            plt.show()
            del crop2
            check2 = input("Are you happy with this? (y/n): ")
            if check2 == 'y':
                happy2 = True
            if check2== 'n':
                yupper = 0
                ylower = 0
                xleft = 0
                xright = 0
                happy2=True
    
    else:
        yupper = 0
        ylower = 0
        xleft = 0
        xright = 0
    
    
    print("This set of coordinates will be used to calculate the means of the calibration images:")
    print(f"x: {xl}+{xleft} to {xr}-{xright}")
    print(f"y: {yu}+{yupper} to {yd}-{ylower}")
    
    
    means, means_errs = get_calibration_data_multiple_2_ROI(cal_data_dir,multi_cal_dark_dir, yu+yupper, yd-ylower, xl+xleft, xr-xright)
    plt.errorbar(currents, means, yerr = means_errs, label='Data', fmt='b.')
    plt.plot(getlobf(currents,means)[0],getlobf(currents,means)[1], label='LOBF', linestyle='dashed')
    plt.xlabel('Current (mA)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()

        
    print(f"The known Dose is {kdose} Gy at {kcurrent} mA")


    
    dose_rate = get_dose_values(currents, kdose, kcurrent)
    
    plt.plot(dose_rate, means, 'bo', label = 'Signal vs Dose Rate Data')
    plt.plot(getlobf(dose_rate,means)[0], getlobf(dose_rate,means)[1],linestyle='dashed',label='LOBF')
    plt.xlabel('Dose Rate (Gy/s)')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    
    return means, means_errs#, dose_conversion

#experiment_pvdr(img_dir,dark_dir)

def experiment():
    pathq = False
    while not pathq:
        direction = input("Are you using Sam's preliminary data (1), data from Laurie's Laptop (2), or data from the SSD (3)?: ")
        
        if direction == '1':
            
            pathq=True
        if direction == '2':
            for i in np.arange(len(os.listdir(lappath))):
                print(i,":",os.listdir(lappath)[i])
            lappath_date = lappath+os.listdir(lappath)[int(input("Select the number of the folder desired: "))]+'/'
            
            for i in np.arange(len(os.listdir(lappath_date))):
                print(i,":",os.listdir(lappath_date)[i])
            img_dir = lappath_date+os.listdir(lappath_date)[int(input("Select the desired image folder name: "))]+'/'
            dark_dir = lappath_date+os.listdir(lappath_date)[int(input("Select the desired dark folder name: "))]+'/'
            
            pathq=True
            
        if direction == '3':
            for i in np.arange(len(os.listdir(extpath))):
                print(i,":",os.listdir(extpath)[i])
            extpath_date = extpath+os.listdir(extpath)[int(input("Select the desired folder number: "))]+'/'
            
            for i in np.arange(len(os.listdir(extpath_date))):
                print(i,":",os.listdir(extpath_date)[i])
            img_dir = extpath_date+os.listdir(extpath_date)[int(input("Select the desired image folder number: "))]+'/'
            dark_dir = extpath_date+os.listdir(extpath_date)[int(input("Select the desired dark folder number: "))]+'/'
            
            pathq=True
            
        else:
            print("Please enter 1, 2 or 3")

    choice_1_q = False
    while not choice_1_q:
        choice_1_a = (input("Would you like to see the PVDR, type {pvdr}, PVDR with known ROI, type {pvdrkr}, calibration, type {cal} , or multi dark calibration, type {multi_cal}? (None = {none}): "))
        if choice_1_a == 'pvdr':
            experiment_pvdr(img_dir,dark_dir)
            choice_1_q = True
        if choice_1_a == 'pvdrkr':
            experiment_pvdr_known_ROI(img_dir,dark_dir)
            choice_1_q = True  
        if choice_1_a == 'cal':
            experiment_cal(img_dir,dark_dir)
            choice_1_q = True
        if choice_1_a == 'calkr':
            experiment_cal_known_ROI(cal_img_dir,cal_dark_dir)
        if choice_1_a == 'multi_cal':
            experiment_cal_multi_dark(img_dir,dark_dir)
            choice_1_q = True
        if choice_1_a == 'multi_calkr':
            experiment_cal_multi_dark_known_ROI(img_dir,dark_dir)
            choice_1_q = True
        if choice_1_a =='none':
            choice_1_q = True


experiment()
'''        
x = np.arange(0, 9)
y = [18.4, 15, 11.6, 8.8, 7.6, 6.3, 5.4, 4.9, 4.2]
plt.plot(x, y, 'o')
x_new=[0,3,5,8]
y_new=[16.0,7.85,5.59,3.63]
plt.plot(x_new,y_new)
plt.show()
'''
