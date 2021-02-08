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
from arguments import Arguments

import sys

# from Arquitecturas.fusion import Tiramisu
# from reconstruction import Image_reconstruction, rgb2gray, gray2rgb, get_class_weights
# import matplotlib.image as mpimg


def Directory(dataset_name):
    
    if dataset_name == "Landsat8":
    
        dir_Images_past = "./Amazonia_Legal/Organized/Images/18_07_2016_image.npy"
        dir_Image_present = "./Amazonia_Legal/Organized/Images/21_07_2017_image.npy"
        
        dir_reference = "./Amazonia_Legal/Organized/Reference/REFERENCE_2017_EPSG32620.npy"
        dir_past_reference ="./Amazonia_Legal/Organized/Reference/PAST_REFERENCE_FROM_1988_2017_EPSG32620_R232_67.npy"
    
    elif dataset_name == "Sentinel2":

        dir_Images_past = "./Amazonas_Sentinel/Imagen_2016_EPSG4674.npy"
        dir_Image_present = "./Amazonas_Sentinel/Imagen_2017_EPSG4674.npy"
        
        dir_reference = "./Amazonas_Sentinel/SENTINEL_REFERENCE_2017_EPSG4674.npy"
        dir_past_reference ="./Amazonas_Sentinel/SENTINEL_PAST_REFERENCE_FROM_1988_2016_EPSG4674.npy"

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


def Normalization(im, Arq):
   
    ######### Normalization ######################
    rows, cols, c = im.shape 
    im = im.reshape((rows * cols, c))  
    # if Arq == 3:
    #     scaler = pre.StandardScaler(with_std=False).fit(im)
    # else:
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


def Using_buffer(ref):
    #Landsat buffer
    selem = disk(4)
    ero = disk(2)
    erosion = np.uint(binary_erosion(ref, ero)) 
    dilation = np.uint(binary_dilation(ref, selem)) 
    buffer  = dilation - erosion
    ref[buffer == 1] = 2
    return ref


def create_mask(rows, cols, val_set, test_set, no_tiles_h, no_tiles_w  ):
    mask = np.ones((rows, cols))
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
        print("Start DeepLabv3p")
        net = Deeplabv3p(input_tensor=None, infer = False,
                input_shape=(patch_size, patch_size, nChannels), classes= len(np.unique(reference)), backbone='mobilenetv2', OS=8, alpha=1.)
    else:           
        print("Start DenseNet")
        net = Tiramisu(input_shape = (patch_size,patch_size,nChannels), n_classes = len(np.unique(reference)), n_filters_first_conv = 32, 
                      n_pool = 3, growth_rate = 8, n_layers_per_block = [4,4,4,4,4,4,4,4,4,4,4],  dropout_p = 0)
    
    net.compile(loss = [weighted_categorical_crossentropy(weights)], optimizer = opt, metrics=["accuracy"])
    
    #net.compile(loss=[categorical_focal_loss(gamma=3, alpha=0.8)], optimizer = opt, metrics=["accuracy"])
      
    return net

        
def train_test(img_pad, gdt, dataset, flag):   

    global net  
    n_batch = len(dataset) // args.batch_size

    loss = np.zeros((1 , 2)) 

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
    net.summary()

    loss_train_plot = []
    accuracy_train = []
    
    loss_val_plot = []
    accuracy_val = []
    
    patience_cnt = 0
    minimum = 10000
              
    print('Start the training')
    start = time.time()

    for epoch in range(args.epochs):
        
        loss_train = np.zeros((1 , 2))
        loss_val = np.zeros((1 , 2))
        
        # Shuffling the train data 
        train_patches = shuffle(train_patches, random_state = 0)

        # Evaluating the network in the train set
        loss_train = train_test(img_pad, gdt, train_patches, flag = 1) 
        
        # To see the training curves
        loss_train_plot.append(loss_train[0, 0])
        accuracy_train.append(100 * loss_train[0, 1]) 
        
        print("%d [training loss: %f, Train Acc: %.2f%%]" %(epoch, loss_train[0, 0], 100 * loss_train[0, 1]))
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
           
         ################################################
        # Evaluating the network in the validation set
        
        loss_val = train_test(img_pad, gdt, val_patches, flag = 0) 
        
        print("%d [Validation loss: %f, Validation Acc: %.2f%%]" %(epoch , loss_val[0 , 0], 100 * loss_val[0 , 1]))
        # To see the validation curves
        loss_val_plot.append(loss_val[0, 0])
        accuracy_val.append(100 * loss_val[0, 1])        

        # Performing Early stopping
        if  loss_val[0,0] < minimum:
            patience_cnt = 0
            minimum = loss_val[0,0]
            # Saving the best model for all runs.
            if args.Arq == 1:
                net.save('best_model_Segnet_%d.h5'%(k))
            elif args.Arq== 2 :
                net.save('best_model_Unet_%d.h5'%(k))
            elif args.Arq== 3 :
                net.save('best_model_Deep_%d.h5'%(k))
            else:
                net.save('best_model_Dense_%d.h5'%(k))
        else:
            patience_cnt += 1

        if patience_cnt > args.patience:
            print("early stopping...")
            break
        
    return loss_train_plot, accuracy_train, loss_val_plot, accuracy_val


if __name__=='__main__':
      
    args = Arguments()

    overlap = round(args.patch_size * args.overlap_percent)
    overlap -= overlap % 2
    stride = args.patch_size - overlap

    if args.Mask_P_M:
        no_tiles_h, no_tiles_w = 10, 10
    else: 
        no_tiles_h, no_tiles_w = 5, 5
    
    x, files_name = load_im(args.dataset)

    for i in range(len(x)):
        if i <= 1:
            if args.dataset == "Landsat8":
                x[i] = x[i][:,1:2551,1:5121]
            elif args.dataset == "Sentinel2":
                x[i] = x[i][:,20:10930,:11000]
        else:
            if args.dataset == "Landsat8":
                x[i]  = x[i][1:2551,1:5121]
            elif args.dataset == "Sentinel2":         
                x[i]  = x[i][20:10930,:11000]

    # Early Fusion: Concatenating the two dates.
    I = np.concatenate((x[0], x[1]), axis = 0) 
    I = I.transpose((1, 2, 0))
    nChannels = I.shape[-1]

    rows, cols, c = I.shape
    Norm_image = Normalization(I, args.Arq) 

    past_reference = x[3]
    act_reference = x[2] 

    reference = act_reference.copy()
    
    class_deforestation = 0
    class_background = 0

    _, counts = np.unique(x[2], return_counts=True)
    print(counts)
    class_deforestation += counts[1]
    class_background += counts[0]
    print('Class Deforestation_No.pixeles:%2f' %(class_deforestation))
    print('Class Background_No.pixeles:%2f' %(class_background))
    print('Percent Class Deforestation:%2f' %(class_deforestation * 100/(class_background + class_deforestation))) 
    print('Percent Class Background: %2f' %(class_background * 100/(class_background + class_deforestation)))  
    print('Proporcion:%2f' %(class_deforestation / class_background))  
    print(args.weights)

    # Cancel buffer
    if args.cancel_buffer:
        reference = Using_buffer(reference)
        if len(args.weights) == 2:
            args.weights.append(0)
        print('Cancel Buffer')
    
    # Cancel past reference
    reference[past_reference == 1] = 2
    if len(args.weights) == 2:
        args.weights.append(0)
    print('Cancel Past Reference')

    # plt.figure()
    # imgplot = plt.imshow(past_reference)
    # plt.show()   

    # Split Image in Tails
    tiles_Image = split_tail(rows, cols, no_tiles_h, no_tiles_w)
    if args.Mask_P_M: 
        tiles_Image = split_tail(rows, cols, no_tiles_h, no_tiles_w)
        Train_tiles = np.array([2, 6, 13, 24, 28, 35, 37, 46, 47, 53, 58, 60, 64, 71, 75, 82, 86, 88, 93])
        Valid_tiles = np.array([8, 11, 26, 49, 78])
        
        Test_tiles = np.arange(100)  
        Test_tiles = list(set(Test_tiles) - set(Train_tiles))
        Test_tiles = list(set(Test_tiles) - set(Valid_tiles))

        train_set =[]
        val_set = []
        test_set = []
        
        for i in Train_tiles:
            train_set.append(tiles_Image[i-1])
        
        for i in Valid_tiles:
            val_set.append(tiles_Image[i-1])
            
        for i in Test_tiles:
            test_set.append(tiles_Image[i-1])
    
    # if k_fold:
    #     n_folds = 2
    #     size = len(tiles_Image)//n_folds
    #     last = n_folds

    for k in range(args.N_run):
        
        # if k_fold:
        #     print("No-k fold %d" %(k))
        #     tiles_Image = shuffle(tiles_Image, random_state = k)
            
        #     test_set = tiles_Image[size * k: size * (k+1)]
        #     train_set = tiles_Image[: size * k] + tiles_Image[size * (k + 1) :]            
        #     val_set = train_set[: math.ceil(0.1 * len(train_set))]
    
        # if cross_validation: 
        #     print("Run number %d" %(k))
        #     tiles_Image = shuffle(tiles_Image, random_state = k)
            
        #     train_set = tiles_Image[: math.ceil(0.25 * len(tiles_Image))]
        #     val_set = train_set[: math.ceil(0.05 * len(train_set))]
        #     train_set = train_set[math.ceil(0.05 * len(train_set)):]
        #     test_set = tiles_Image[math.ceil(0.25 * len(tiles_Image)) :]

        mask = create_mask(rows, cols, val_set, test_set, no_tiles_h, no_tiles_w )
        # plt.figure()
        # imgplot = plt.imshow(mask)
        # plt.show()             

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
        val_mask = (1 - train_mask) * (1 - test_mask)   
       
        # Split paches index            
        train_patches, val_patches, _ = split_patches(k1, k2)
        print(len(train_patches))

        gdt = Hot_encoding(gdt_pad)  
   
        loss_train_plot, accuracy_train, loss_val_plot, accuracy_val = Train(img_pad, gdt, train_patches, val_patches)
        
        # Saving the curves of the loss for train and validation
        # np.savez("loss_train_%d_%d"%(args.Arq,k), loss_train_plot)
        # np.savez("acc_train_%d_%d"%(args.Arq,k),  accuracy_train)
        # np.savez("loss_val_%d_%d"%(args.Arq,k), loss_val_plot)
        # np.savez("acc_val_%d_%d"%(args.Arq,k), accuracy_val)
                    
     

