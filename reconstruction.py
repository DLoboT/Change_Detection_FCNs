#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:40:28 2020

@author: daliana
"""

import numpy as np
import cv2
from keras import backend as K
import matplotlib.pyplot as plt
import sys
from skimage.util.shape import view_as_windows
import time

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def ploting(t0, v0, t1, v1, title='figure',
            ylabel = 'metric', xlim=[0, 150], ylim=[0, 100]):
    
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle(title)
    
    ax0.plot(t0, label='trainig')
    ax0.plot(v0, label='validation')
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_xlabel('epochs')
    ax0.set_ylabel(ylabel)
    ax0.set_title('Mean Conv')
    ax0.grid()
    ax0.legend()

    ax1.plot(t1, label='trainig')
    ax1.plot(v1, label='validation')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel(ylabel)
    ax1.set_title('Mean Separable Conv')
    ax1.grid()
    ax1.legend()
       
    plt.show()
    
def gray2rgb(image):
    """
    Funtion to convert classes values from 0,1,3,4 to rgb values
    """
    row,col = image.shape
    image = image.reshape((row*col))
    rgb_output = np.zeros((row*col, 3))
    rgb_map = [[0,0,255],[0,255,0],[0,255,255],[255,255,0],[255,255,255]]
    for j in np.unique(image):
        rgb_output[image==j] = np.array(rgb_map[j])
    
    rgb_output = rgb_output.reshape((row,col,3))  
    rgb_output = cv2.cvtColor(rgb_output.astype('uint8'),cv2.COLOR_BGR2RGB)
    return rgb_output 

class Image_reconstruction(object):

    def __init__ (self, net, output_c_dim, patch_size=512, overlap_percent=0):

        self.net = net
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
   
    def Inference(self, tile):
       
        '''
        Normalize before call this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
       
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((k1*stride, k2*stride, self.output_c_dim))

        for i in range(k1):
            for j in range(k2):
               
                patch = tile_pad[i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size), :]
                patch = patch[np.newaxis,...]
                
                #This is to see the inference time for an image
                #window_shape_array = (self.patch_size, self.patch_size, tile_pad.shape[2])
                #stride_array = (stride, stride, 1)
                #patches_array = np.array(view_as_windows(tile_pad, window_shape_array, step = stride_array))
                #print(patches_array.shape)
                
                #k1, k2, p, row, col, depth = patches_array.shape
                #patches_array = patches_array.reshape(k1*k2,row,col,depth)  
                                
                #print('Start the Inference')
                #start = time.time()
                #infer = self.net.predict(patches_array, verbose=1)
                infer = self.net.predict(patch, verbose=0)
                #predict_probs = obj.Inference(Norm_image)
                #end = time.time()
                #hours, rem = divmod(end-start, 3600)
                #minutes, seconds = divmod(rem, 60)
                #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                
                #sys.exit()

                probs[i*stride : i*stride+stride,
                      j*stride : j*stride+stride, :] = infer[0, overlap//2 : overlap//2 + stride,
                                                                overlap//2 : overlap//2 + stride, :]
            print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[:k1*stride-step_row, :k2*stride-step_col]

        return probs
    

# Visualizing every channel of a layer:
        
    
def Visualize_activations(layer_name, layer_activations):   
        
    # Number of feature maps in a row
    images_per_row = 8
    # Number of feature maps in the feature map    
    n_features = layer_activations.shape[-1]

    size = layer_activations.shape[1]

    ncols = n_features // images_per_row
    display_grid = np.zeros((size * ncols, images_per_row * size))
    
    for col in range(ncols):
        for row in range(images_per_row):
            # Colocate the activations of each filter into a horizontal grid
            channel_image = layer_activations[:,:, col * images_per_row + row]
            
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            
            display_grid[col * size : (col + 1) * size, row * size: (row + 1) * size] = channel_image
        
    scale = 1. / size
    plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
    
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect = 'auto',cmap = 'viridis')
                    



def Visualization_heatmaps(layer_name, layer_activations, net, patch_im_test):
    
    last_conv = net.get_layer(name=layer_name)
    # Gradients of the target class (deforestation) with regard to the output feature map of conv_16
    grads = K.gradients(net.output[:, :,:, 1], last_conv.output)[0]
    
    # Global average pooling 
    # Mean intensity of the gradient over a specific feature map
    # Axes to compute the mean axis = (0,1,2)            
    pooled_grads = K.mean(grads,axis = (0, 1, 2))
    
    # The keras function takes the image as an input and returns the pooled_ grads and the activation map of the conv_16 
    # Input: input of the model   Output: 2 outputs one for the gradients and the other for a convolution
    # This function runs the computation graph that we have created before.            
    iterate = K.function([net.input], [pooled_grads, last_conv.output[0]])
    # Return the output values as numpy arrays
    pooled_grads_value, conv_layer_output = iterate([patch_im_test])

    # Here we multiply each activation map with the corresponding gradient which acts as weigths determining how important
    # each channel is with regard to the target class            
    for i in range(64):
        conv_layer_output[:,:,i] *= pooled_grads_value[i]
        
    # The channel wise mean of the resulting feature map is the heatmap of the class activation                
    heatmap = np.mean(conv_layer_output, axis = -1)
    
    # Normalizing the heatmap between 0 and 1
    heatmap = np.maximum(heatmap, 0)  # Zeroing the smallest values
    heatmap /= np.max(heatmap)
    
    plt.matshow(heatmap, cmap ='viridis')
    
    return heatmap

def scaling(value, scale,total_pixels):
    # Here scaling between 0 to a predifined scale
    value = np.round((value - 1)/(total_pixels - 1)  * scale, 1) 

    return value

def get_class_weights(pixels_negative, pixels_positive, scale):
    
    total_pixels = pixels_negative + pixels_positive

    weight_positive = scaling(pixels_negative, scale, total_pixels)
    #print(weight_positive)

    weight_negative  = scaling(pixels_positive, scale,total_pixels)
    #print(weight_negative) 

    weigths = [weight_negative, weight_positive]

    return weigths
    


    
    
    
    
    
    
    
    
    
    
    
