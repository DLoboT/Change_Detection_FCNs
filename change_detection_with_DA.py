#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:37:21 2020

@author: daliana
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:01:12 2020

@author: daliana
"""

from keras.optimizers import Adam
from PIL import Image
import numpy as np
from time import sleep
import multiprocessing
import itertools
from sklearn.utils import shuffle
import math
import keras
from sklearn import preprocessing as pre
import time
import matplotlib.pyplot as plt
from keras import backend as K
from skimage.morphology import binary_dilation, disk, binary_erosion
from Arquitecturas.U_net import Unet
from Arquitecturas.segnet_unpooling import Segnet
from Arquitecturas.deeplabv3p import Deeplabv3p
from Arquitecturas.DenseNet import Tiramisu
from Arquitecturas.resunet_paper import build_res_unet
from arguments import Arguments
import tensorflow as tf
import sys

# from Arquitecturas.fusion import Tiramisu
# from reconstruction import Image_reconstruction, rgb2gray, gray2rgb, get_class_weights
# import matplotlib.image as mpimg


def Directory(dataset_name):
    
    if dataset_name == "Landsat8":
    
        dir_Images_past = "./Amazonas_Sentinel_Landsat/L8_20170802_EPSG4326.npy"
        dir_Image_present = "./Amazonas_Sentinel_Landsat/L8_20180906_EPSG4326.npy"
        
        dir_reference = "./Amazonas_Sentinel_Landsat/referencia_2018_landsat_EPSG4326.npy"
        dir_past_reference = "./Amazonas_Sentinel_Landsat/PAST_REFERENCE_LANDSAT_EPSG4326.npy"
    
    elif dataset_name == "Sentinel2":

        dir_Images_past = "./Amazonas_Sentinel_Landsat/S2A_20170724_EPSG4326.npy"
        dir_Image_present = "./Amazonas_Sentinel_Landsat/S2A_20180907_EPSG4326.npy"
        
        dir_reference = "./Amazonas_Sentinel_Landsat/referencia_2018_sentinel_EPSG4326.npy"
        dir_past_reference = "./Amazonas_Sentinel_Landsat/PAST_REFERENCE_SENTINEL_EPSG4326.npy"

    dataset = dir_Images_past, dir_Image_present, dir_reference, dir_past_reference

    return dataset


def load(j):
   
    im = np.load(j)

    return [im, j]


def load_im(dataset_name): 

    sleep(0.5)
    
    dataset = Directory(dataset_name)
    
    num_cores = multiprocessing.cpu_count()    
    pool = multiprocessing.Pool(num_cores)
    
    im = pool.map(load, dataset, chunksize = 1)    
    
    pool.close()
    pool.join()
    
    x = []
    files_name = []
    
    for i in range(len(im)):
        x_set, z = im[i]
        x.append(x_set)
        files_name.append(z)

    return x, files_name


def NDVI_band(image):
    # This is to pick up the band 8    
    band_8 = (image[4,:,:] - image[3,:,:])/(image[4,:,:] + image[3,:,:])
    band_8 = np.expand_dims(band_8, axis = 0)

    return band_8


def split_tail(rows, cols, no_tiles_h, no_tiles_w):   
   
    h = np.arange(0, rows, int(rows/no_tiles_h))
    w = np.arange(0, cols, int(cols/no_tiles_w))

    #Tiles coordinates
    tiles = list(itertools.product(h,w))
    
    return tiles   


def Normalization(im):
   
    ######### Normalization ######################
    rows, cols, c = im.shape 
    im = im.reshape((rows * cols, c))  

    scaler = pre.StandardScaler().fit(im)
    Norm_Image = np.float32(scaler.transform(im))
    Norm_Image = Norm_Image.reshape((rows, cols, c))
    return  Norm_Image
    

def Hot_encoding(ref):    
    ######## Hot encoding #########################
    rows, cols = ref.shape 
    classes=len(np.unique(ref))
    imgs_mask_cat = ref.reshape(-1)
    imgs_mask_cat = keras.utils.to_categorical(imgs_mask_cat, classes)
    gdt = imgs_mask_cat.reshape(rows, cols, classes) 
    return  gdt


def Using_buffer(ref, args):

    if args.dataset == "Landsat8":
        selem = disk(2)
        ero = disk(1)
    else: 
        selem = disk(6)
        ero = disk(3)
    erosion = np.uint(binary_erosion(ref, ero)) 
    dilation = np.uint(binary_dilation(ref, selem)) 
    buffer  = dilation - erosion
    ref[buffer == 1] = 2
    return ref


def create_mask(rows, cols, train_set, val_set, test_set, no_tiles_h, no_tiles_w  ):
    mask = np.ones((rows, cols))*0.2
    for i in train_set:
        mask[i[0]:i[0] + int(rows/no_tiles_h), i[1]:i[1] + int(cols/no_tiles_w)] = 1
    for i in val_set:
        mask[i[0]:i[0] + int(rows/no_tiles_h), i[1]:i[1] + int(cols/no_tiles_w)] = .5
    for i in test_set:
        mask[i[0]:i[0] + int(rows/no_tiles_h), i[1]:i[1] + int(cols/no_tiles_w)] = 0
    return mask


def Transform(arr, b):

    sufix = ''

    if b == 1:
        arr = np.rot90(arr, k = 1)
        sufix = '_rot90'
    elif b == 2:
        arr = np.rot90(arr, k = 2)
        sufix = '_rot180'
    elif b == 3:
        arr = np.rot90(arr, k = 3)
        sufix = '_rot270'
    elif b == 4:
        arr = np.flipud(arr)
        sufix = '_flipud'
    elif b == 5:
        arr = np.fliplr(arr)
        sufix = '_fliplr'
    elif b == 6:
        arr = np.transpose(arr, (1, 0, 2))
        sufix = '_transpose'
    elif b == 7:
        arr = np.rot90(arr, k = 2)
        arr = np.transpose(arr, (1, 0, 2))
        sufix = '_transverse'

    return arr


def Add_padding(reference):
    # Add Padding to the image to match with the patch size
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col) )
    pad_tuple_img = ( (overlap//2, overlap//2 + step_row), (overlap//2, overlap//2 + step_col), (0, 0) )
    
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')    
    img_pad = np.pad(Norm_image, pad_tuple_img, mode = 'symmetric')
    gdt_pad = np.pad(reference, pad_tuple_msk, mode = 'symmetric')
    
    return mask_pad, img_pad, gdt_pad


def split_patches(k1, k2):
    
    for i in range(k1):
        for j in range(k2):
            # Test
            if test_mask[i*stride:i*stride + args.patch_size, j*stride:j*stride + args.patch_size].all():
                test_patches.append((i*stride, j*stride))
            elif val_mask[i*stride:i*stride + args.patch_size, j*stride:j*stride + args.patch_size].all():
                # We only do data augmentation to the patches where there is a positive sample.
                if gdt_pad[i*stride:i*stride + args.patch_size, j*stride:j*stride + args.patch_size].any():
                    # Only patches with samples
                    for q in data_augmentation_index:                   
                        val_patches.append((i*stride, j*stride, q))
                else:
                    val_patches.append((i*stride, j*stride, 0))

            elif train_mask[i*stride:i*stride + args.patch_size, j*stride:j*stride + args.patch_size].all():
                # We only do data augmentation to the patches where there is a positive sample.
                if gdt_pad[i*stride:i*stride + args.patch_size, j*stride:j*stride + args.patch_size].any():
                    # Only patches with samples
                    for q in data_augmentation_index:                   
                        train_patches.append((i*stride, j*stride, q))
                else:
                    train_patches.append((i*stride, j*stride, 0))
                    
    return train_patches, val_patches, test_patches  


def patch_image(img_pad, gdt, data_batch):
        
    patch_im = []
    patch_gdt = []    

    # Loading the patches in the image
    for j in range(len(data_batch)):
        I_patch = img_pad[data_batch[j][0]: data_batch[j][0] + args.patch_size, data_batch[j][1]: data_batch[j][1] + args.patch_size,:]
        # Apply transformations to the image patches. 
        I_patch = Transform (I_patch, data_batch[j][2])  
        patch_im.append(I_patch)  
        
        gdt_patch = gdt[data_batch[j][0]: data_batch[j][0] + args.patch_size, data_batch[j][1]: data_batch[j][1] + args.patch_size, :]
        # Apply transformations to the reference patches. 
        gdt_patch = Transform (gdt_patch, data_batch[j][2]) 
        patch_gdt.append(gdt_patch) 
    
    patch_im = np.array(patch_im)
    
    patch_gdt = np.array(patch_gdt)

    return patch_im, patch_gdt

 
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        
        return loss


def network(Arq, reference, weights, patch_size, nChannels, args):
 
    opt = Adam(lr=args.lr)

    if Arq == 1:
        print("Start Segnet")
        net = Segnet(nClasses = len(np.unique(reference)), optimizer = None, input_width = patch_size , input_height = patch_size , nChannels =nChannels )
    elif Arq == 2:
        print("Start Unet")
        net = Unet(len(np.unique(reference)), patch_size, patch_size , nChannels) 
    elif Arq ==3:
        print("Start DeepLabv3p - xception")
        net = Deeplabv3p(input_tensor=None, infer = False,
                input_shape=(patch_size, patch_size, nChannels), classes= len(np.unique(reference)), backbone='xception', OS=8, alpha=1.)
    elif Arq ==4:
        print("Start DeepLabv3p - mobilenetv2")
        net = Deeplabv3p(input_tensor=None, infer = False,
                input_shape=(patch_size, patch_size, nChannels), classes= len(np.unique(reference)), backbone='mobilenetv2', OS=8, alpha=1.)
    elif Arq ==5:           
        print("Start DenseNet")
        net = Tiramisu(input_shape = (patch_size,patch_size,nChannels), n_classes = len(np.unique(reference)), n_filters_first_conv = 32, 
                      n_pool = 3, growth_rate = 8, n_layers_per_block = [4,4,4,4,4,4,4,4,4,4,4],  dropout_p = 0)
    else:
        print("Start ResUNet")
        net = build_res_unet(input_shape=(patch_size,patch_size,nChannels), nClasses = len(np.unique(reference)))   


    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,1] * y_pred[:,:,:,1], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,:,:,1], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,1] * y_pred[:,:,:,1], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[:,:,:,1], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    print(weights)
    net.compile(loss = [weighted_categorical_crossentropy(weights)], optimizer = opt, metrics=['acc', f1_m, precision_m, recall_m])

    #net.compile(loss=[categorical_focal_loss(gamma=3, alpha=0.8)], optimizer = opt, metrics=["accuracy"])
      
    return net

        
def train_test(img_pad, gdt, dataset, flag):   

    global net  
    n_batch = len(dataset) // args.batch_size

    loss = np.zeros((5)) 

    ########### Training per batch ####################
    for i in range(n_batch):
        # Data_batch is going to be the shape 
        # (x_coordinates, y coordinates, transformation_index) in the batch
        data_batch = dataset[i * args.batch_size : (i + 1) * args.batch_size]

        patch_im, patch_gdt = patch_image(img_pad, gdt, data_batch)
        
        # print(patch_im.shape)
        # print(patch_gdt.shape)

        if flag:
            loss += net.train_on_batch(patch_im, patch_gdt)
        else:
            loss += net.test_on_batch(patch_im, patch_gdt)
        
    if len(dataset) % args.batch_size:        
    
        data_batch = dataset[n_batch * args.batch_size : ] 
        
        patch_im, patch_gdt = patch_image(img_pad, gdt, data_batch)
        
        if flag:
            loss += net.train_on_batch(patch_im, patch_gdt)
        else:
            loss += net.test_on_batch(patch_im, patch_gdt)
    # Here, we have a remanent batch, so we have to add 1 to the n_batch    
        loss= loss/(n_batch + 1)
    else:
        loss= loss/n_batch

    return loss


def Train(img_pad, gdt, train_patches, val_patches):

    global net
    net = network(args.Arq, reference, args.weights, args.patch_size, nChannels, args) 
    print(net.get_weights()[0].reshape((-1))[0])
    net.summary()
    loss_train_plot, accuracy_train, f1_train, precision_train, recall_train, \
    loss_val_plot, accuracy_val, f1_val, precision_val, recall_val = [[] for i in range(10)]
    
    patience_cnt = 0
    max_f1_val = 0.0
    min_loss = 10000.0
              
    print('Start the training')
    start = time.time()

    for epoch in range(args.epochs):
        
        loss_train = np.zeros((5))
        loss_val = np.zeros((5))
        
        # Shuffling the train data 
        train_patches = shuffle(train_patches, random_state = 0)

        # Evaluating the network in the train set
        loss_train = train_test(img_pad, gdt, train_patches, flag = 1) 
        
        # To see the training curves
        loss_train_plot.append(loss_train[0])
        accuracy_train.append(100 * loss_train[1]) 
        f1_train.append(100 * loss_train[2]) 
        precision_train.append(100 * loss_train[3]) 
        recall_train.append(100 * loss_train[4]) 
        
        end = time.time()
        tm = end - start
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)

        if epoch == 0 and k==0: f = open("Results_" + output_file_name + ".txt", 'w')
        else: f = open("Results_" + output_file_name + ".txt", 'a')
        print_line = "Epoch: %d [TR -- loss: %f, Acc: %.2f%%, F1: %.2f%%, Prec: %.2f%%, Rec: %.2f%%, Time: %s]\n" %(epoch , loss_train[0], 100 * loss_train[1], 100 * loss_train[2], 100 * loss_train[3], 100 * loss_train[4], "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        f.write(print_line)
        print (print_line)

         ################################################
        # Evaluating the network in the validation set
        loss_val = train_test(img_pad, gdt, val_patches, flag = 0) 
        print_line = "Epoch: %d [VL -- loss: %f, Acc: %.2f%%, F1: %.2f%%, Prec: %.2f%%, Rec: %.2f%%]\n" %(epoch , loss_val[0], 100 * loss_val[1], 100 * loss_val[2], 100 * loss_val[3], 100 * loss_val[4])
        f.write(print_line)
        print (print_line)
        f.close()

        # To see the validation curves
        loss_val_plot.append(loss_val[0])
        accuracy_val.append(100 * loss_val[1])        
        f1_val.append(100 * loss_val[2])        
        precision_val.append(100 * loss_val[3])        
        recall_val.append(100 * loss_val[4])        

        # Performing Early stopping
        # if  loss_val[2] > max_f1_val:
        #     max_f1_val = loss_val[2]
        if  loss_val[0] < min_loss:
            min_loss = loss_val[0]
            patience_cnt = 0
            # Saving the best model for all runs.
            net.save('best_' + output_file_name + '_%d.h5'%(k))
        else:
            patience_cnt += 1

        if patience_cnt > args.patience:
            print("early stopping...")
            f = open("Results_" + output_file_name + ".txt", 'a')       
            f.write('----------- Time ---------\n')
            f.write('Fold:%d\n'%(k))
            f.write('Epoch:%d\n'%(epoch))
            f.write('{:0>2}:{:0>2}:{:05.2f}\n'.format(int(hours),int(minutes),seconds))
            f.write('--------------------------\n')           
            f.close()
            
            break
        
    return loss_train_plot, accuracy_train, loss_val_plot, accuracy_val, tm


if __name__=='__main__':
      
    args = Arguments()

    if args.Arq == 1:
        output_file_name = "Model_Segnet"
    elif args.Arq == 2:
        output_file_name = "Model_Unet"
    elif args.Arq == 3:
        output_file_name = "Model_DeepLab_xception"
    elif args.Arq == 4:
        output_file_name = "Model_DeepLab_mobilenetv2"
    elif args.Arq == 5:
        output_file_name = "Model_FCDenseNet"
    else:
        output_file_name = "Model_ResUNet"

    overlap = round(args.patch_size * args.overlap_percent)
    overlap -= overlap % 2
    stride = args.patch_size - overlap

    if args.Mask_P_M:
        no_tiles_h, no_tiles_w = 3, 5
    else: 
        no_tiles_h, no_tiles_w = 5, 5
    
    x, files_name = load_im(args.dataset)

    for i in range(len(x)):
        if i <= 1:

            if args.dataset == "Landsat8":
                # x[i] = x[i][:,1565:,:3670]
                x[i] = x[i][:,1510:,:3670]
            elif args.dataset == "Sentinel2":
               x[i] = x[i][:,3*1510:,:3*3670]
        else:
            if args.dataset == "Landsat8":
                # x[i]  = x[i][1565:,:3670]
                 x[i]  = x[i][1510:,:3670]
            elif args.dataset == "Sentinel2":         
                x[i]  = x[i][3*1510:,:3*3670]

    # Early Fusion: Concatenating the two dates.
    I = np.concatenate((x[0], x[1]), axis = 0) 
    I = I.transpose((1, 2, 0))
    nChannels = I.shape[-1]
    
    # print(x[0].shape)
    # print(x[1].shape)
    # print(np.isnan(x[0]).sum())
    # print(np.isnan(x[1]).sum())

    rows, cols, c = I.shape
    Norm_image = Normalization(I) 

    past_reference = x[3]
    act_reference = x[2] 
    
    print(past_reference.shape)
    print(act_reference.shape)
    
    print(np.unique(past_reference))
    print(np.unique(act_reference))
    sys.exit()

    reference = act_reference.copy()
    
    class_deforestation = 0
    class_background = 0

    l, counts = np.unique(x[2], return_counts=True)
    print(l, counts)
    class_deforestation += counts[1]
    class_background += counts[0]
    print('Class Deforestation_No.pixeles:%2f' %(class_deforestation))
    print('Class Background_No.pixeles:%2f' %(class_background))
    print('Percent Class Deforestation:%2f' %(class_deforestation * 100/(class_background + class_deforestation))) 
    print('Percent Class Background: %2f' %(class_background * 100/(class_background + class_deforestation)))  
    print('Proporcion:%2f' %(class_deforestation / class_background))  
    # print(args.weights)

    # Cancel buffer
    if args.cancel_buffer:
        reference = Using_buffer(reference, args)
        args.weights.append(0)
        print('Cancel Buffer')
    
    # Cancel past reference
    reference[past_reference == 1] = 2
    if len(args.weights) == 0:
        args.weights.append(0)
    print('Cancel Past Reference')

    if args.Mask_P_M: 
        tiles_Image = split_tail(rows, cols, no_tiles_h, no_tiles_w)
        #Train_tiles = np.array([2, 6, 13, 24, 28, 35, 37, 46, 47, 53, 58, 60, 64, 71, 75, 82, 86, 88, 93])
        #Valid_tiles = np.array([8, 11, 26, 49, 78])  # old data
        
        # Train_tiles = np.array([6, 13, 23, 24, 25, 35,47, 48, 49, 50, 60, 70, 71, 75, 95, 96, 97, 98])
        # Valid_tiles = np.array([8, 36, 37, 63, 85])
        
        # Train_tiles = np.array([21, 23, 24, 32, 34, 37, 44, 57, 65, 68, 87, 89, 90, 9, 72, 74, 16, 82, 73, 27, 39,  7, 15, 51, 86, 19, 28, 96, 94, 97, 17, 18, 31, 52,  5, 30, 81])
              
        # Train_tiles = np.array([21, 23, 24, 32, 34, 37, 44, 57, 65, 68, 87, 89, 90, 9, 72, 74, 16, 82, 73, 27, 39,  7, 15, 51, 86]) 
        # Valid_tiles = np.array([40, 43, 67, 76, 100, 4, 3, 83, 92, 36])

        # Train_tiles = np.array([25, 26, 27, 39, 49, 50, 56, 59, 60, 65, 67, 76, 78, 53, 21, 88, 10, 11, 15, 20, 18, 72, 86,  8,  4, 64]) 
        # Valid_tiles = np.array([12, 22, 74, 75, 89, 90, 91, 100, 19, 17, 73, 63,  9, 31, 82, 98])

        # Train_tiles = np.array([25, 26, 27, 39, 49, 50, 56, 59, 60, 65, 67, 76, 78]) 
        # Valid_tiles = np.array([12, 22, 74, 75, 89, 90, 91, 100, 28])
        # Test_tiles = np.array([1,  2,  3, 13, 14, 23, 24, 30, 33, 34, 35, 36, 37, 38, 40, 44, 45, 46, 47, 48, 55, 57, 58, 66, 68, 69, 70, 77, 79, 80])

        Train_tiles = np.array([13, 14, 15]) 
        Valid_tiles = np.array([9])
        Test_tiles = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])


        # Test_tiles = np.arange(100)  
        # Test_tiles = list(set(Test_tiles) - set(Train_tiles))
        # Test_tiles = list(set(Test_tiles) - set(Valid_tiles))

        train_set =[]
        val_set = []
        test_set = []
        
        for i in Train_tiles:
            train_set.append(tiles_Image[i-1])
        
        for i in Valid_tiles:
            val_set.append(tiles_Image[i-1])
            
        for i in Test_tiles:
            test_set.append(tiles_Image[i-1])

    tm_list = []
    print(args.N_run)
    
    mask = create_mask(rows, cols, train_set, val_set, test_set, no_tiles_h, no_tiles_w )
    plt.figure()
    imgplot = plt.imshow(mask)
    plt.savefig('mask.png')

    # Weights by (Patel,2020)
    # train_tile = act_reference[mask==1]
    # class_k, count = np.unique(train_tile, return_counts=True)
    # args.weights = [1] + args.weights
    # args.weights = [(np.log(sum(counts[:2])) - np.log(counts[0]))/ (np.log(sum(counts[:2])) - np.log(counts[1]))] + args.weights
    # args.weights = [element * 5 for element in args.weights]

    args.weights = [0.4, 2, 0]
    
    for k in range(args.N_run):

        # tf.reset_default_graph()
        # tf.set_random_seed(0)
        # np.random.seed(0)

        # This is the number of pixels you have to add in the padding step
        step_row = (stride - rows % stride) % stride
        step_col = (stride - cols % stride) % stride
        
        mask_pad, img_pad, gdt_pad = Add_padding(reference)
               
        # List for data augmentation.
        data_augmentation_index = [0, 1, 4, 5]
        
        train_patches, val_patches, test_patches = [], [], []
        k1, k2 = (rows + step_row)//stride, (cols + step_col)//stride
        print('Total number of patches: %d x %d' %(k1, k2))
        
        # Create the mask to train, validation and test.
        train_mask = np.zeros_like(mask_pad)
        val_mask = np.zeros_like(mask_pad)
        test_mask = np.zeros_like(mask_pad)
        train_mask[mask_pad == 1] = 1
        test_mask[mask_pad == 0] = 1
        val_mask[mask_pad == 0.5] = 1       
               
        # Split paches index            
        train_patches, val_patches, test_patches = split_patches(k1, k2)

        print(len(train_patches))
        print(len(val_patches))
        print(len(test_patches))

        gdt = Hot_encoding(gdt_pad)  
   
        loss_train_plot, accuracy_train, loss_val_plot, accuracy_val, tm = Train(img_pad, gdt, train_patches, val_patches)

        tm_list.append(tm)
        
        # Saving the curves of the loss for train and validation
        # np.savez("loss_train_%d_%d"%(args.Arq,k), loss_train_plot)
        # np.savez("acc_train_%d_%d"%(args.Arq,k),  accuracy_train)
        # np.savez("loss_val_%d_%d"%(args.Arq,k), loss_val_plot)
        # np.savez("acc_val_%d_%d"%(args.Arq,k), accuracy_val)
    
    tm_list = np.asarray(tm_list)
    f = open("Results_" + output_file_name + ".txt", 'a')
    f.write('----------- Mean Time ---------\n')
    hours, rem = divmod(tm_list.mean(), 3600)
    minutes, seconds = divmod(rem, 60)
    f.write('Time Mean: %d +- %d\n'%(tm_list.mean(), tm_list.std()))
    f.write('{:0>2}:{:0>2}:{:05.2f}\n'.format(int(hours),int(minutes),seconds))
    f.write('Time Median: %d\n'%(np.median(tm_list)))
    hours, rem = divmod(np.median(tm_list), 3600)
    minutes, seconds = divmod(rem, 60)
    f.write('{:0>2}:{:0>2}:{:05.2f}\n'.format(int(hours),int(minutes),seconds))
    f.close()
